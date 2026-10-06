"""Chromium engine adapter: real Chromium over CDP.

This is Afnan's Chromium-based browser engine: it launches a
Chromium-family browser (Chromium, Chrome or Edge) as a local
subprocess and drives it through the Chrome DevTools Protocol
(CDP) — the same protocol Afnan's own future customized
Chromium build will speak.  It implements
:class:`BrowserEngineAdapter`, so the stack above it never
changes::

    Afnan Agent → Browser Tools → BrowserController
        → AfnanBrowserRuntime → ChromiumAdapter → Chromium

The adapter uses only the Python standard library (a minimal
WebSocket client is included) — there is no Playwright and no
third-party driver dependency here.  Playwright remains
available as the fallback/development adapter in
``backend.py``; both adapters produce the same normalized
results, so the controller cannot tell them apart.

Profiles map to Chromium user-data directories: each Afnan
profile gets its own browser process with its own storage
directory, so cookies/storage can never leak across profiles.
The debugging port binds to 127.0.0.1 only; profile data stays
on disk in the profile directory and is never read back into
AgentState, logs or screenshots.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from afnan_ai.browser.base import BrowserErrorCode, BrowserException
from afnan_ai.browser.engine import BrowserEngineAdapter
from afnan_ai.redaction import redact_text

#: Environment variable naming an explicit Chromium executable.
EXECUTABLE_ENV = "AFNAN_CHROMIUM_EXECUTABLE"

_CANDIDATES = {
    "chromium": (
        "chromium", "chromium-browser", "google-chrome",
        "google-chrome-stable", "chrome",
    ),
    "chrome": (
        "google-chrome", "google-chrome-stable", "chrome",
        "chromium", "chromium-browser",
    ),
    "edge": (
        "microsoft-edge", "microsoft-edge-stable", "msedge",
    ),
}


def _windows_install_paths(browser: str) -> list[Path]:
    """Standard Windows install locations (not on PATH)."""
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    program_files_x86 = os.environ.get(
        "ProgramFiles(x86)", r"C:\Program Files (x86)"
    )
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    paths = []
    if browser in ("chromium", "chrome"):
        paths += [
            Path(program_files) / "Google" / "Chrome"
            / "Application" / "chrome.exe",
            Path(program_files_x86) / "Google" / "Chrome"
            / "Application" / "chrome.exe",
            Path(local_app_data) / "Chromium" / "Application"
            / "chrome.exe",
        ]
    if browser in ("chromium", "edge"):
        paths += [
            Path(program_files) / "Microsoft" / "Edge"
            / "Application" / "msedge.exe",
            Path(program_files_x86) / "Microsoft" / "Edge"
            / "Application" / "msedge.exe",
        ]
    return paths


def _playwright_cache_dirs() -> list[Path]:
    """Playwright browser cache locations across platforms."""
    dirs = [Path.home() / ".cache" / "ms-playwright"]
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            dirs.append(Path(local_app_data) / "ms-playwright")
    return [d for d in dirs if d.is_dir()]

_SUPPORTED = ("chromium", "chrome", "edge")


# ----------------------------------------------------------------------
# Minimal WebSocket client (RFC 6455) — enough for CDP.
# ----------------------------------------------------------------------


class _WebSocket:
    """A blocking text-frame WebSocket client (stdlib only)."""

    def __init__(self, url: str, timeout: float = 15.0):
        parsed = urlparse(url)
        if parsed.scheme not in ("ws", "wss"):
            raise ValueError(f"Unsupported WebSocket URL {url!r}")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._file = self._sock.makefile("rb")
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self._sock.sendall(request.encode("ascii"))
        status = self._file.readline(4096).decode("latin-1")
        if " 101 " not in status:
            self.close()
            raise ConnectionError(
                f"WebSocket upgrade refused: {status.strip()}"
            )
        while True:  # consume headers up to the blank line
            line = self._file.readline(4096)
            if line in (b"\r\n", b"\n", b""):
                break

    # -- framing -----------------------------------------------------
    def send_text(self, text: str) -> None:
        payload = text.encode("utf-8")
        mask = os.urandom(4)
        header = bytearray([0x81])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header += length.to_bytes(2, "big")
        else:
            header.append(0x80 | 127)
            header += length.to_bytes(8, "big")
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._sock.sendall(bytes(header) + masked)

    def _read_exact(self, count: int) -> bytes:
        data = self._file.read(count)
        if data is None or len(data) < count:
            raise ConnectionError("WebSocket closed by browser")
        return data

    def recv_text(self, timeout: float) -> str | None:
        """Receive one text message; None on timeout."""
        self._sock.settimeout(timeout)
        fragments: list[bytes] = []
        try:
            while True:
                first = self._read_exact(2)
                fin = bool(first[0] & 0x80)
                opcode = first[0] & 0x0F
                length = first[1] & 0x7F
                if length == 126:
                    length = int.from_bytes(self._read_exact(2), "big")
                elif length == 127:
                    length = int.from_bytes(self._read_exact(8), "big")
                if first[1] & 0x80:  # masked (servers should not)
                    self._read_exact(4)
                payload = self._read_exact(length) if length else b""
                if opcode == 0x8:
                    raise ConnectionError("WebSocket closed by browser")
                if opcode == 0x9:  # ping -> pong
                    self._send_control(0xA, payload)
                    continue
                if opcode in (0x0, 0x1, 0x2):
                    fragments.append(payload)
                    if fin:
                        return b"".join(fragments).decode(
                            "utf-8", "replace"
                        )
        except (socket.timeout, TimeoutError):
            return None

    def _send_control(self, opcode: int, payload: bytes) -> None:
        mask = os.urandom(4)
        header = bytes([0x80 | opcode, 0x80 | len(payload)]) + mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        try:
            self._sock.sendall(header + masked)
        except OSError:
            pass

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


class _CDPError(Exception):
    """The browser answered a CDP command with an error."""


class _CDPConnection:
    """Synchronous CDP client over one browser WebSocket."""

    def __init__(self, websocket: _WebSocket):
        self._ws = websocket
        self._next_id = 0
        self._events: list[dict[str, Any]] = []

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        self._next_id += 1
        call_id = self._next_id
        payload: dict[str, Any] = {
            "id": call_id, "method": method, "params": params or {},
        }
        if session_id:
            payload["sessionId"] = session_id
        self._ws.send_text(json.dumps(payload))
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"CDP call {method} timed out")
            message = self._ws.recv_text(remaining)
            if message is None:
                continue
            try:
                data = json.loads(message)
            except ValueError:
                continue
            if data.get("id") == call_id:
                if "error" in data:
                    raise _CDPError(
                        str(
                            (data.get("error") or {}).get(
                                "message", method
                            )
                        )
                    )
                return data.get("result") or {}
            if "method" in data:
                self._events.append(data)

    def poll(self, timeout: float = 0.05) -> None:
        """Read one pending message (event) if any arrives."""
        try:
            message = self._ws.recv_text(timeout)
        except (ConnectionError, OSError):
            return
        if message:
            try:
                data = json.loads(message)
            except ValueError:
                return
            if "method" in data:
                self._events.append(data)

    def drain_events(self) -> list[dict[str, Any]]:
        events, self._events = self._events, []
        return events

    def close(self) -> None:
        self._ws.close()


# ----------------------------------------------------------------------
# Handles (opaque to every layer above the adapter)
# ----------------------------------------------------------------------


class _ChromiumPage:
    __slots__ = ("profile", "target_id", "session_id")

    def __init__(self, profile: str, target_id: str, session_id: str):
        self.profile = profile
        self.target_id = target_id
        self.session_id = session_id


class _ChromiumElement:
    __slots__ = ("page", "object_id")

    def __init__(self, page: _ChromiumPage, object_id: str):
        self.page = page
        self.object_id = object_id


class _ProfileState:
    """One profile = one Chromium process + user-data dir."""

    def __init__(self, name: str, user_data_dir: str = ""):
        self.name = name
        self.user_data_dir = user_data_dir
        self.options: dict[str, Any] = {}
        self.process: subprocess.Popen | None = None
        self.cdp: _CDPConnection | None = None
        self.external = False
        self.owns_dir = False
        self.pages: dict[str, _ChromiumPage] = {}
        self.network: dict[str, list[dict[str, Any]]] = {}
        self.dialogs: dict[str, list[dict[str, Any]]] = {}
        self.pending_requests: dict[tuple[str, str], str] = {}


# ----------------------------------------------------------------------
# The adapter
# ----------------------------------------------------------------------


class ChromiumAdapter(BrowserEngineAdapter):
    """Drive a real Chromium-family browser via CDP.

    Requires a Chromium, Chrome or Edge binary on the machine
    (or ``AFNAN_CHROMIUM_EXECUTABLE`` pointing at one); without
    one, :meth:`start` raises a structured
    ``browser_unavailable`` error.  No third-party packages are
    needed.
    """

    name = "chromium"

    def __init__(
        self,
        executable_path: str | None = None,
        user_data_dir: str | None = None,
    ):
        self._executable = executable_path
        self._user_data_dir = user_data_dir
        self._profiles: dict[str, _ProfileState] = {
            "default": _ProfileState(
                "default", user_data_dir or ""
            )
        }
        self._active_profile = "default"
        self._browser_name = "chromium"
        self._headless = True
        self._started = False

    # -- executable discovery -------------------------------------------

    @staticmethod
    def find_executable(browser: str = "chromium") -> str | None:
        """Locate a Chromium-family executable, if installed."""
        env = os.environ.get(EXECUTABLE_ENV)
        if env and Path(env).is_file():
            return env
        for candidate in _CANDIDATES.get(browser, _CANDIDATES["chromium"]):
            found = shutil.which(candidate)
            if found:
                return found
        # Windows: standard install locations are not on PATH.
        for path in _windows_install_paths(browser):
            if path.is_file():
                return str(path)
        # Playwright's downloaded browsers work too.
        for cache in _playwright_cache_dirs():
            patterns = (
                "*/chrome-linux/chrome",
                "*/chrome-linux/headless_shell",
                "*/chrome-mac/*/Chromium",
                "*/chromium-*/chrome-linux/chrome",
                "*/chrome-win/chrome.exe",
                "*/chrome-win/headless_shell.exe",
            )
            for pattern in patterns:
                for path in sorted(cache.glob(pattern)):
                    if path.is_file():
                        return str(path)
        return None

    def _executable_for(self, browser: str) -> str:
        if self._executable and Path(self._executable).is_file():
            return self._executable
        found = self.find_executable(browser)
        if found is None:
            raise BrowserException(
                "No Chromium-family browser was found. Install "
                "Chromium, Chrome or Edge, or set "
                f"{EXECUTABLE_ENV} to its executable path.",
                code=BrowserErrorCode.BROWSER_UNAVAILABLE,
                details={"browser": browser},
            )
        return found

    # -- lifecycle --------------------------------------------------------

    def start(self, browser: str, headless: bool) -> None:
        browser = str(browser or "chromium").lower()
        if browser not in _SUPPORTED:
            raise BrowserException(
                f"ChromiumAdapter drives Chromium-family browsers, "
                f"not {browser!r}. Use the Playwright adapter for "
                "Firefox/WebKit.",
                code=BrowserErrorCode.BROWSER_UNAVAILABLE,
                details={"browser": browser},
            )
        if self._started:
            raise BrowserException(
                "Browser is already running",
                code=BrowserErrorCode.OPERATION_FAILED,
            )
        self._browser_name = browser
        self._headless = bool(headless)
        self._started = True
        try:
            self._launch(self._profiles["default"])
        except Exception:
            self._started = False
            raise

    def connect(self, endpoint: str) -> None:
        """Attach to a running Chromium (http://host:port or ws://…)."""
        endpoint = str(endpoint or "").strip()
        ws_url = endpoint
        if endpoint.startswith(("http://", "https://")):
            try:
                with urllib.request.urlopen(
                    endpoint.rstrip("/") + "/json/version", timeout=5
                ) as response:
                    info = json.loads(response.read().decode("utf-8"))
                ws_url = str(info.get("webSocketDebuggerUrl") or "")
            except Exception as e:
                raise BrowserException(
                    f"Could not read Chromium version info at "
                    f"{endpoint!r}: {e}",
                    code=BrowserErrorCode.CONNECTION_FAILED,
                    details={"endpoint": endpoint},
                ) from e
        if not ws_url.startswith("ws"):
            raise BrowserException(
                f"No Chromium DevTools endpoint at {endpoint!r}",
                code=BrowserErrorCode.CONNECTION_FAILED,
                details={"endpoint": endpoint},
            )
        state = self._profiles["default"]
        state.external = True
        try:
            state.cdp = self._open_cdp(ws_url)
            state.cdp.call("Target.setDiscoverTargets", {"discover": True})
        except Exception as e:
            raise BrowserException(
                f"Could not connect to Chromium at {endpoint!r}: {e}",
                code=BrowserErrorCode.CONNECTION_FAILED,
                details={"endpoint": endpoint},
            ) from e
        self._started = True
        self._active_profile = "default"

    def stop(self) -> None:
        for state in list(self._profiles.values()):
            self._shutdown_state(state)
        default_dir = self._user_data_dir or ""
        self._profiles = {
            "default": _ProfileState("default", default_dir)
        }
        self._active_profile = "default"
        self._started = False

    def restart(self) -> None:
        """Engine-level restart (the runtime also offers its own)."""
        self.stop()
        self.start(self._browser_name, self._headless)

    def is_alive(self) -> bool:
        if not self._started:
            return False
        states = [
            s for s in self._profiles.values() if s.cdp is not None
        ]
        if not states:
            return False
        for state in states:
            if state.external:
                try:
                    state.cdp.call("Browser.getVersion", {}, timeout=3)
                    return True
                except Exception:
                    continue
            elif state.process is not None and state.process.poll() is None:
                return True
        return False

    def capabilities(self) -> dict[str, bool]:
        return {
            "tabs": True,
            "accessibility": True,
            "screenshots": True,
            "downloads": False,
            "uploads": True,
            "persistent_profiles": True,
            "iframe": False,
            "shadow_dom": False,
            "network_observation": True,
            "dialogs": True,
        }

    # -- process management -------------------------------------------------

    def _free_port(self) -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _open_cdp(self, ws_url: str) -> _CDPConnection:
        return _CDPConnection(_WebSocket(ws_url))

    def _spawn_browser_process(self, state: _ProfileState) -> str:
        """Launch the Chromium process; return its browser WS URL."""
        executable = self._executable_for(self._browser_name)
        if not state.user_data_dir:
            state.user_data_dir = tempfile.mkdtemp(
                prefix=f"afnan-{state.name}-"
            )
            state.owns_dir = True
        headless_flags: list[str | None] = (
            ["--headless=new", "--headless"] if self._headless else [None]
        )
        last_error = ""
        for headless_flag in headless_flags:
            port = self._free_port()
            args = [
                executable,
                f"--remote-debugging-port={port}",
                "--remote-debugging-address=127.0.0.1",
                f"--user-data-dir={state.user_data_dir}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-dev-shm-usage",
                "--mute-audio",
            ]
            if state.options.get("user_agent"):
                args.append(f"--user-agent={state.options['user_agent']}")
            if state.options.get("locale"):
                args.append(f"--lang={state.options['locale']}")
            if headless_flag:
                args.append(headless_flag)
            args.append("about:blank")
            process = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            state.process = process
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    last_error = (
                        f"Chromium exited with code "
                        f"{process.returncode} (profile "
                        f"{state.name!r} may be locked or corrupted)"
                    )
                    break
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/json/version",
                        timeout=2,
                    ) as response:
                        info = json.loads(
                            response.read().decode("utf-8")
                        )
                    ws_url = str(info.get("webSocketDebuggerUrl") or "")
                    if ws_url:
                        return ws_url
                except Exception:
                    time.sleep(0.15)
            else:
                last_error = "Chromium did not open its debug port"
            try:
                process.terminate()
            except Exception:
                pass
            state.process = None
        raise BrowserException(
            f"Could not launch Chromium for profile "
            f"{state.name!r}: {last_error}",
            code=BrowserErrorCode.BROWSER_UNAVAILABLE,
            details={"profile": state.name},
        )

    def _launch(self, state: _ProfileState) -> None:
        ws_url = self._spawn_browser_process(state)
        try:
            state.cdp = self._open_cdp(ws_url)
            state.cdp.call("Browser.getVersion", {}, timeout=10)
        except Exception as e:
            self._shutdown_state(state)
            raise BrowserException(
                f"Could not talk to the launched Chromium: {e}",
                code=BrowserErrorCode.BROWSER_UNAVAILABLE,
                details={"profile": state.name},
            ) from e

    def _shutdown_state(self, state: _ProfileState) -> None:
        if state.cdp is not None and not state.external:
            try:
                state.cdp.call("Browser.close", {}, timeout=3)
            except Exception:
                pass
        if state.cdp is not None:
            state.cdp.close()
            state.cdp = None
        if state.process is not None:
            try:
                if state.process.poll() is None:
                    state.process.terminate()
                    try:
                        state.process.wait(timeout=3)
                    except Exception:
                        state.process.kill()
            except Exception:
                pass
            state.process = None
        state.pages.clear()
        state.network.clear()
        state.dialogs.clear()
        state.pending_requests.clear()
        if state.owns_dir and state.user_data_dir:
            shutil.rmtree(state.user_data_dir, ignore_errors=True)
            state.owns_dir = False

    # -- CDP plumbing --------------------------------------------------------

    def _cdp(
        self,
        state: _ProfileState,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        if state.cdp is None:
            raise BrowserException(
                "Browser is not running",
                code=BrowserErrorCode.BROWSER_NOT_STARTED,
            )
        try:
            return state.cdp.call(
                method, params, session_id=session_id, timeout=timeout
            )
        except TimeoutError as e:
            raise BrowserException(
                f"Timed out waiting for Chromium ({method})",
                code=BrowserErrorCode.TIMEOUT,
            ) from e
        except _CDPError as e:
            raise BrowserException(
                f"Chromium error ({method}): {e}",
                code=BrowserErrorCode.OPERATION_FAILED,
            ) from e
        except (ConnectionError, OSError) as e:
            raise BrowserException(
                f"Lost connection to Chromium: {e}",
                code=BrowserErrorCode.BROWSER_CRASHED,
            ) from e

    def _page_state(self, page: _ChromiumPage) -> _ProfileState:
        return self._profiles[page.profile]

    def _pump_events(self, state: _ProfileState) -> list[dict[str, Any]]:
        """Process pending CDP events (dialogs, network, targets)."""
        if state.cdp is None:
            return []
        try:
            state.cdp.poll(0.02)
        except Exception:
            return []
        events = state.cdp.drain_events()
        for event in events:
            try:
                self._handle_event(state, event)
            except Exception:
                pass
        return events

    def _handle_event(
        self, state: _ProfileState, event: dict[str, Any]
    ) -> None:
        method = event.get("method")
        params = event.get("params") or {}
        session_id = event.get("sessionId")
        page = next(
            (
                p for p in state.pages.values()
                if p.session_id == session_id
            ),
            None,
        )
        if method == "Page.javascriptDialogOpening" and session_id:
            # Record (redacted) and dismiss: a page must never
            # hang the agent on a dialog nobody can see; nothing
            # is ever auto-confirmed.
            if page is not None:
                seen = state.dialogs.setdefault(page.target_id, [])
                seen.append({
                    "type": str(params.get("type") or "alert"),
                    "message": redact_text(
                        str(params.get("message") or "")
                    )[:200],
                    "action": "dismissed",
                })
                del seen[:-20]
            try:
                state.cdp.call(  # type: ignore[union-attr]
                    "Page.handleJavaScriptDialog",
                    {"accept": False},
                    session_id=session_id,
                    timeout=5,
                )
            except Exception:
                pass
        elif method == "Network.responseReceived" and page is not None:
            response = params.get("response") or {}
            status = int(response.get("status") or 0)
            if status >= 400:
                self._record_network(state, page, {
                    "url": str(response.get("url") or ""),
                    "method": "",
                    "status": status,
                    "failure": None,
                    "resource_type": str(params.get("type") or ""),
                })
        elif method == "Network.requestWillBeSent" and page is not None:
            request_id = str(params.get("requestId") or "")
            request = params.get("request") or {}
            state.pending_requests[(page.target_id, request_id)] = str(
                request.get("url") or ""
            )
        elif method == "Network.loadingFailed" and page is not None:
            request_id = str(params.get("requestId") or "")
            url = state.pending_requests.pop(
                (page.target_id, request_id), ""
            )
            self._record_network(state, page, {
                "url": url,
                "method": "",
                "status": None,
                "failure": str(params.get("errorText") or "failed"),
                "resource_type": str(params.get("type") or ""),
            })
        elif method == "Target.targetDestroyed":
            target_id = str(params.get("targetId") or "")
            state.pages.pop(target_id, None)

    @staticmethod
    def _record_network(
        state: _ProfileState,
        page: _ChromiumPage,
        entry: dict[str, Any],
    ) -> None:
        events = state.network.setdefault(page.target_id, [])
        events.append(entry)
        del events[:-200]  # bounded history per page

    # -- profiles (isolated processes) -------------------------------------

    def create_profile_context(
        self, name: str, options: dict[str, Any]
    ) -> str:
        """Register an isolated profile (own process + storage).

        The profile's browser process starts when the profile
        becomes active; a persistent profile's cookies/storage
        live in ``options["user_data_dir"]``.
        """
        state = self._profiles.get(name)
        if state is None:
            state = _ProfileState(name)
            self._profiles[name] = state
        user_data_dir = (options or {}).get("user_data_dir")
        if user_data_dir and not state.user_data_dir and state.cdp is None:
            state.user_data_dir = str(user_data_dir)
            state.owns_dir = False
        for key in ("user_agent", "locale", "timezone_id",
                    "color_scheme", "viewport"):
            if (options or {}).get(key) is not None:
                state.options[key] = options[key]
        return name

    def set_active_profile(self, name: str) -> None:
        state = self._profiles.get(name)
        if state is None:
            raise BrowserException(
                f"Unknown browser profile {name!r}",
                code=BrowserErrorCode.PROFILE_ERROR,
                details={"profile": name},
            )
        if name == self._active_profile and state.cdp is not None:
            return
        if self._started and state.cdp is None:
            # Profile isolation in Chromium means a separate
            # process with its own user-data directory; other
            # profiles' processes (and their tabs) keep running.
            self._launch(state)
        self._active_profile = name

    # -- pages ---------------------------------------------------------------

    def _require_page_state(self) -> _ProfileState:
        if not self._started:
            raise BrowserException(
                "Browser is not running",
                code=BrowserErrorCode.BROWSER_NOT_STARTED,
            )
        state = self._profiles[self._active_profile]
        if state.cdp is None:
            self._launch(state)
        return state

    def new_page(self) -> Any:
        state = self._require_page_state()
        result = self._cdp(
            state, "Target.createTarget", {"url": "about:blank"}
        )
        target_id = str(result.get("targetId") or "")
        attach = self._cdp(
            state,
            "Target.attachToTarget",
            {"targetId": target_id, "flatten": True},
        )
        session_id = str(attach.get("sessionId") or "")
        page = _ChromiumPage(state.name, target_id, session_id)
        state.pages[target_id] = page
        for domain in ("Page", "Runtime", "DOM", "Network",
                       "Accessibility"):
            try:
                self._cdp(state, f"{domain}.enable", {}, session_id=session_id)
            except BrowserException:
                pass
        viewport = state.options.get("viewport")
        if isinstance(viewport, dict) and viewport.get("width"):
            try:
                self._cdp(
                    state,
                    "Emulation.setDeviceMetricsOverride",
                    {
                        "width": int(viewport["width"]),
                        "height": int(viewport.get("height") or 720),
                        "deviceScaleFactor": 1,
                        "mobile": False,
                    },
                    session_id=session_id,
                )
            except BrowserException:
                pass
        return page

    def close_page(self, handle: Any) -> None:
        state = self._page_state(handle)
        self._cdp(
            state, "Target.closeTarget", {"targetId": handle.target_id}
        )
        state.pages.pop(handle.target_id, None)

    def list_pages(self) -> list[Any]:
        pages: list[Any] = []
        for state in self._profiles.values():
            if state.cdp is None:
                continue
            try:
                targets = self._cdp(
                    state, "Target.getTargets", {}, timeout=10
                ).get("targetInfos") or []
            except BrowserException:
                continue
            live: set[str] = set()
            for target in targets:
                if target.get("type") != "page":
                    continue
                target_id = str(target.get("targetId") or "")
                live.add(target_id)
                if target_id not in state.pages:
                    # A page the controller did not create
                    # (popup): attach so it becomes manageable.
                    try:
                        attach = self._cdp(
                            state,
                            "Target.attachToTarget",
                            {"targetId": target_id, "flatten": True},
                        )
                        page = _ChromiumPage(
                            state.name,
                            target_id,
                            str(attach.get("sessionId") or ""),
                        )
                        state.pages[target_id] = page
                        for domain in ("Page", "Runtime"):
                            try:
                                self._cdp(
                                    state, f"{domain}.enable", {},
                                    session_id=page.session_id,
                                )
                            except BrowserException:
                                pass
                    except BrowserException:
                        continue
            for target_id in list(state.pages):
                if target_id not in live:
                    state.pages.pop(target_id, None)
            pages.extend(state.pages.values())
        return pages

    # -- navigation ------------------------------------------------------------

    def goto(self, handle: Any, url: str) -> None:
        state = self._page_state(handle)
        result = self._cdp(
            state, "Page.navigate", {"url": url},
            session_id=handle.session_id, timeout=45,
        )
        error = result.get("errorText")
        if error:
            raise BrowserException(
                f"Chromium could not load {url!r}: {error}",
                code=BrowserErrorCode.NAVIGATION_FAILED,
                details={"url": url},
            )
        # Wait (condition-based) for the load event, bounded.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            events = self._pump_events(state)
            if any(
                e.get("method") == "Page.loadEventFired"
                and e.get("sessionId") == handle.session_id
                for e in events
            ):
                break
            time.sleep(0.05)

    def go_back(self, handle: Any) -> None:
        self._history_step(handle, -1)

    def go_forward(self, handle: Any) -> None:
        self._history_step(handle, +1)

    def _history_step(self, handle: Any, delta: int) -> None:
        state = self._page_state(handle)
        history = self._cdp(
            state, "Page.getNavigationHistory", {},
            session_id=handle.session_id,
        )
        entries = history.get("entries") or []
        index = int(history.get("currentIndex") or 0) + delta
        if not entries or index < 0 or index >= len(entries):
            raise BrowserException(
                "No page in that direction of the tab history",
                code=BrowserErrorCode.NAVIGATION_FAILED,
            )
        self._cdp(
            state,
            "Page.navigateToHistoryEntry",
            {"entryId": entries[index]["id"]},
            session_id=handle.session_id,
        )

    def reload(self, handle: Any) -> None:
        state = self._page_state(handle)
        self._cdp(
            state, "Page.reload", {}, session_id=handle.session_id
        )

    def page_url(self, handle: Any) -> str:
        return str(self._evaluate(handle, "location.href") or "")

    def page_title(self, handle: Any) -> str:
        return str(self._evaluate(handle, "document.title") or "")

    # -- JS evaluation plumbing -------------------------------------------------

    def _evaluate(
        self, page: _ChromiumPage, expression: str, *, by_value: bool = True
    ) -> Any:
        state = self._page_state(page)
        result = self._cdp(
            state,
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": by_value,
                "awaitPromise": False,
            },
            session_id=page.session_id,
        )
        if result.get("exceptionDetails"):
            details = result["exceptionDetails"]
            text = str(
                (details.get("exception") or {}).get("description")
                or details.get("text")
                or "JavaScript error"
            )
            raise BrowserException(
                f"Chromium page script failed: {text}",
                code=BrowserErrorCode.OPERATION_FAILED,
            )
        payload = result.get("result") or {}
        return payload.get("value") if by_value else payload

    def _elements_from(
        self, page: _ChromiumPage, expression: str
    ) -> list[_ChromiumElement]:
        descriptor = self._evaluate(page, expression, by_value=False)
        object_id = (descriptor or {}).get("objectId")
        if not object_id:
            return []
        state = self._page_state(page)
        props = self._cdp(
            state,
            "Runtime.getProperties",
            {"objectId": object_id, "ownProperties": True},
            session_id=page.session_id,
        )
        indexed: list[tuple[int, str]] = []
        for prop in props.get("result") or []:
            name = str(prop.get("name") or "")
            value = prop.get("value") or {}
            if name.isdigit() and value.get("objectId"):
                indexed.append((int(name), str(value["objectId"])))
        try:
            self._cdp(
                state,
                "Runtime.releaseObject",
                {"objectId": object_id},
                session_id=page.session_id,
            )
        except BrowserException:
            pass
        return [
            _ChromiumElement(page, oid) for _, oid in sorted(indexed)
        ]

    def _call_on(
        self,
        element: _ChromiumElement,
        declaration: str,
        arguments: list[Any] | None = None,
    ) -> Any:
        page = element.page
        state = self._page_state(page)
        result = self._cdp(
            state,
            "Runtime.callFunctionOn",
            {
                "objectId": element.object_id,
                "functionDeclaration": declaration,
                "returnByValue": True,
                "arguments": [
                    {"value": a} for a in (arguments or [])
                ],
            },
            session_id=page.session_id,
        )
        if result.get("exceptionDetails"):
            details = result["exceptionDetails"]
            text = str(
                (details.get("exception") or {}).get("description")
                or details.get("text")
                or "element action failed"
            )
            lowered = text.lower()
            if "cannot find context" in lowered or "no object" in lowered:
                raise BrowserException(
                    "Element is no longer attached to the page: "
                    f"{text}",
                    code=BrowserErrorCode.STALE_ELEMENT,
                )
            raise BrowserException(
                f"Chromium element action failed: {text}",
                code=BrowserErrorCode.OPERATION_FAILED,
            )
        return (result.get("result") or {}).get("value")

    # -- element interaction ------------------------------------------------
    # Locator strategies mirror the Playwright adapter's priority:
    # selector -> test_id -> label -> placeholder -> role(+name) -> text.
    _LOCATOR_FN = r"""
(locator, limit) => {
  const all = (sel) => {
    try { return Array.from(document.querySelectorAll(sel)); }
    catch (e) { return []; }
  };
  const text = (el) => ((el.innerText || el.textContent || ''))
    .trim().replace(/\s+/g, ' ');
  const interactiveSel = "a,button,input,select,textarea,"
    + "[role='button'],[role='link'],[role='textbox'],"
    + "[role='checkbox'],[contenteditable='true']";
  const roleOf = (el) => {
    const r = el.getAttribute('role');
    if (r) return r.toLowerCase();
    const tag = el.tagName.toLowerCase();
    if (tag === 'button') return 'button';
    if (tag === 'a' && el.hasAttribute('href')) return 'link';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'select') return 'combobox';
    if (tag === 'input') {
      const t = (el.getAttribute('type') || 'text').toLowerCase();
      if (t === 'checkbox') return 'checkbox';
      if (t === 'radio') return 'radio';
      if (t === 'submit' || t === 'button') return 'button';
      return 'textbox';
    }
    return '';
  };
  const nameOf = (el) => (el.getAttribute('aria-label')
    || text(el) || el.getAttribute('placeholder')
    || el.value || '').trim();
  const strategies = [];
  if (locator.selector) {
    strategies.push(() => all(locator.selector));
  }
  if (locator.test_id) {
    const wanted = String(locator.test_id).replace(/"/g, '\\"');
    strategies.push(() => all('[data-testid="' + wanted + '"]'));
  }
  if (locator.label) {
    strategies.push(() => {
      const wanted = locator.label;
      const fields = all('input,textarea,select');
      const direct = fields.filter(
        (el) => (el.getAttribute('aria-label') || '') === wanted);
      const labels = all('label').filter((l) => text(l) === wanted);
      const viaLabel = [];
      for (const label of labels) {
        const target = (label.htmlFor
          ? document.getElementById(label.htmlFor) : null)
          || label.querySelector('input,textarea,select');
        if (target) viaLabel.push(target);
      }
      return [...direct, ...viaLabel];
    });
  }
  if (locator.placeholder) {
    strategies.push(() => all('input,textarea').filter(
      (el) => (el.getAttribute('placeholder') || '')
        === locator.placeholder));
  }
  if (locator.role) {
    strategies.push(() => all(interactiveSel + ',[role]').filter(
      (el) => roleOf(el) === String(locator.role).toLowerCase()
        && (!locator.name || nameOf(el) === locator.name)));
  }
  if (locator.text) {
    strategies.push(() => all(interactiveSel).filter(
      (el) => text(el).includes(locator.text)));
  }
  for (const fn of strategies) {
    const found = fn();
    if (found.length) return found.slice(0, limit);
  }
  return [];
}
"""

    def query_elements(
        self, handle: Any, locator: dict[str, Any], limit: int
    ) -> list[Any]:
        expression = (
            f"({self._LOCATOR_FN})"
            f"({json.dumps(locator or {})}, {max(1, int(limit))})"
        )
        return self._elements_from(handle, expression)

    def interactive_elements(
        self, handle: Any, limit: int
    ) -> list[Any]:
        selector = (
            "a, button, input, select, textarea, "
            "[role='button'], [role='link'], [role='textbox'], "
            "[contenteditable='true']"
        )
        expression = (
            "Array.from(document.querySelectorAll("
            f"{json.dumps(selector)})).slice(0, {max(1, int(limit))})"
        )
        return self._elements_from(handle, expression)

    _INFO_FN = r"""
function() {
  const el = this;
  const attrs = {};
  const names = ['id','name','type','role','aria-label',
    'placeholder','href','value','title'];
  for (const name of names) {
    const v = el.getAttribute ? el.getAttribute(name) : null;
    if (v !== null && v !== undefined) attrs[name] = String(v);
  }
  const text = ((el.innerText || el.textContent || ''))
    .trim().replace(/\s+/g, ' ').slice(0, 200);
  const editable = !!el.isContentEditable
    || ['INPUT','TEXTAREA','SELECT'].includes(el.tagName);
  let visible = false;
  try {
    const r = el.getBoundingClientRect();
    const s = window.getComputedStyle(el);
    visible = r.width > 0 && r.height > 0
      && s.visibility !== 'hidden' && s.display !== 'none';
  } catch (e) {}
  return {
    tag: (el.tagName || '').toLowerCase(),
    text: text,
    attributes: attrs,
    editable: editable,
    value: (el.value !== undefined && el.value !== null)
      ? String(el.value).slice(0, 200) : '',
    visible: visible,
    enabled: !el.disabled
  };
}
"""

    def element_info(self, handle: Any, element: Any) -> dict[str, Any]:
        info = self._call_on(element, self._INFO_FN)
        return info if isinstance(info, dict) else {}

    def click_element(
        self, handle: Any, element: Any, timeout_ms: int
    ) -> None:
        self._call_on(
            element,
            "function() { this.scrollIntoView({block: 'center',"
            " inline: 'nearest'}); this.click(); }",
        )
        self._pump_events(self._page_state(handle))

    _FILL_FN = r"""
function(text) {
  const el = this;
  el.focus();
  if (el.isContentEditable) {
    el.innerText = text;
  } else {
    const proto = el instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : (el instanceof HTMLInputElement
        ? HTMLInputElement.prototype : null);
    const descriptor = proto
      && Object.getOwnPropertyDescriptor(proto, 'value');
    if (descriptor && descriptor.set) {
      descriptor.set.call(el, text);
    } else {
      el.value = text;
    }
  }
  el.dispatchEvent(new Event('input', {bubbles: true}));
  el.dispatchEvent(new Event('change', {bubbles: true}));
  return true;
}
"""

    def fill_element(
        self, handle: Any, element: Any, text: str, timeout_ms: int
    ) -> None:
        self._call_on(element, self._FILL_FN, [text])

    def clear_element(
        self, handle: Any, element: Any, timeout_ms: int
    ) -> None:
        self._call_on(element, self._FILL_FN, [""])

    def select_option(
        self, handle: Any, element: Any, value: str, timeout_ms: int
    ) -> None:
        selected = self._call_on(
            element,
            r"""
function(value) {
  const el = this;
  el.focus();
  el.value = value;
  const ok = el.value === value;
  el.dispatchEvent(new Event('input', {bubbles: true}));
  el.dispatchEvent(new Event('change', {bubbles: true}));
  return ok;
}
""",
            [value],
        )
        if not selected:
            raise BrowserException(
                f"No option with value {value!r} in the select",
                code=BrowserErrorCode.INVALID_ELEMENT,
            )

    def set_input_files(
        self, handle: Any, element: Any, path: str, timeout_ms: int
    ) -> None:
        state = self._page_state(handle)
        node = self._cdp(
            state,
            "DOM.requestNode",
            {"objectId": element.object_id},
            session_id=handle.session_id,
        )
        self._cdp(
            state,
            "DOM.setFileInputFiles",
            {"files": [str(path)], "nodeId": node.get("nodeId")},
            session_id=handle.session_id,
        )

    def press_key(
        self,
        handle: Any,
        element: Any | None,
        key: str,
        timeout_ms: int,
    ) -> None:
        if element is not None:
            self._call_on(element, "function() { this.focus(); }")
        state = self._page_state(handle)
        text = key if len(key) == 1 else {"Enter": "\r", "Tab": "\t"}.get(key, "")
        for event_type in ("keyDown", "keyUp"):
            params: dict[str, Any] = {"type": event_type, "key": key}
            if text:
                params["text"] = text
            self._cdp(
                state, "Input.dispatchKeyEvent", params,
                session_id=handle.session_id,
            )

    def scroll_page(self, handle: Any, dx: int, dy: int) -> None:
        self._evaluate(
            handle, f"window.scrollBy({int(dx)}, {int(dy)})"
        )

    def scroll_to_element(self, handle: Any, element: Any) -> None:
        self._call_on(
            element,
            "function() { this.scrollIntoView({block: 'center'}); }",
        )

    # -- page observation ------------------------------------------------------

    def page_text(self, handle: Any) -> str:
        text = self._evaluate(
            handle,
            "document.body ? (document.body.innerText"
            " || document.body.textContent || '') : ''",
        )
        return str(text or "")

    _CONTENT_JS = r"""() => {
      const clean = (el) => ((el && el.innerText) || '')
        .replace(/\s+/g, ' ').trim();
      const headings = [...document.querySelectorAll('h1,h2,h3,h4,h5,h6')]
        .map(h => ({level: parseInt(h.tagName[1], 10), text: clean(h)}))
        .filter(h => h.text);
      const paragraphs = [...document.querySelectorAll('p')]
        .map(p => clean(p)).filter(Boolean);
      const lists = [...document.querySelectorAll('ul, ol')]
        .map(l => [...l.querySelectorAll(':scope > li')]
          .map(li => clean(li)).filter(Boolean))
        .filter(l => l.length);
      const links = [...document.querySelectorAll('a[href]')]
        .map(a => ({text: clean(a), url: a.href}))
        .filter(l => l.url && !l.url.startsWith('javascript:'));
      const tables = [...document.querySelectorAll('table')]
        .map(t => ({
          caption: clean(t.querySelector('caption')),
          rows: [...t.querySelectorAll('tr')]
            .map(tr => [...tr.querySelectorAll('th, td')]
              .map(c => clean(c)))
        }))
        .filter(t => t.rows.length);
      return {
        title: document.title || '',
        url: location.href,
        headings, paragraphs, lists, links, tables,
        text: clean(document.body)
      };
    }"""

    def page_content(self, handle: Any) -> dict[str, Any]:
        result = self._evaluate(handle, self._CONTENT_JS)
        return result if isinstance(result, dict) else {}

    _PROBE_JS = r"""() => {
      const frameworks = [];
      if (window.__NEXT_DATA__) frameworks.push('nextjs');
      if (window.React || document.querySelector(
        '[data-reactroot], [data-reactid]')) frameworks.push('react');
      if (window.Vue || document.querySelector('[data-v-app]'))
        frameworks.push('vue');
      if (window.ng) frameworks.push('angular');
      const body = document.body;
      const text = body ? (body.innerText || '') : '';
      const html = body ? body.innerHTML : '';
      let hash = 0;
      for (let i = 0; i < html.length; i++) {
        hash = ((hash << 5) - hash + html.charCodeAt(i)) | 0;
      }
      return {
        url: location.href,
        title: document.title,
        ready_state: document.readyState,
        frameworks: frameworks,
        text_length: text.length,
        element_count: document.querySelectorAll('*').length,
        content_hash: String(hash)
      };
    }"""

    def page_probe(self, handle: Any) -> dict[str, Any]:
        result = self._evaluate(handle, self._PROBE_JS)
        return result if isinstance(result, dict) else {}

    def accessibility_snapshot(self, handle: Any) -> Any:
        """The page's accessibility tree via the CDP
        Accessibility domain — Chromium's current AX API (no
        deprecated snapshot APIs), converted to the same
        {role, name, children} tree the controller normalizes."""
        state = self._page_state(handle)
        result = self._cdp(
            state, "Accessibility.getFullAXTree", {},
            session_id=handle.session_id,
        )
        nodes = result.get("nodes") or []
        by_id: dict[str, dict[str, Any]] = {
            str(n.get("nodeId")): n for n in nodes
        }
        children_of: dict[str, list[dict[str, Any]]] = {}
        for node in nodes:
            parent = node.get("parentId")
            if parent:
                children_of.setdefault(str(parent), []).append(node)

        def build(node: dict[str, Any]) -> dict[str, Any]:
            role = str(
                (node.get("role") or {}).get("value") or "generic"
            ).lower()
            built: dict[str, Any] = {
                "role": role,
                "name": str(
                    (node.get("name") or {}).get("value") or ""
                ),
            }
            value = (node.get("value") or {}).get("value")
            if value not in (None, ""):
                built["value"] = value
            kids = [
                by_id[child]
                for child in node.get("childIds") or []
                if str(child) in by_id
            ] or children_of.get(str(node.get("nodeId")), [])
            children = [build(child) for child in kids]
            if children:
                built["children"] = children
            return built

        root = next(
            (n for n in nodes if not n.get("parentId")),
            nodes[0] if nodes else None,
        )
        if root is None:
            return {"role": "document", "name": "", "children": []}
        return build(root)

    def network_events(self, handle: Any) -> list[dict[str, Any]]:
        state = self._page_state(handle)
        self._pump_events(state)
        return [
            dict(e)
            for e in state.network.get(handle.target_id, [])
        ]

    def dialogs(self, handle: Any) -> list[dict[str, Any]]:
        state = self._page_state(handle)
        self._pump_events(state)
        return [
            dict(d)
            for d in state.dialogs.get(handle.target_id, [])
        ]

    def screenshot(self, handle: Any) -> bytes:
        state = self._page_state(handle)
        # fromSurface can hang in some headless environments;
        # fall back to a plain capture when it does.
        result = None
        last_error = ""
        for params in (
            {"format": "png", "fromSurface": True},
            {"format": "png"},
        ):
            try:
                result = self._cdp(
                    state,
                    "Page.captureScreenshot",
                    params,
                    session_id=handle.session_id,
                    timeout=30,
                )
                break
            except BrowserException as e:
                last_error = str(e)
        if result is None:
            raise BrowserException(
                f"Chromium screenshot failed: {last_error}",
                code=BrowserErrorCode.OPERATION_FAILED,
            )
        data = result.get("data")
        if not data:
            raise BrowserException(
                "Chromium returned an empty screenshot",
                code=BrowserErrorCode.OPERATION_FAILED,
            )
        return base64.b64decode(str(data))

    # -- computer-use primitives (CDP Input/DOM) --------------------------
    def element_box(
        self, handle: Any, element: Any
    ) -> dict[str, Any] | None:
        box = self._call_on(
            element,
            "function() { const r = this.getBoundingClientRect();"
            " return {x: r.x, y: r.y, width: r.width,"
            " height: r.height}; }",
        )
        return box if isinstance(box, dict) else None

    def _dispatch_mouse(
        self, handle: Any, event_type: str, x: float, y: float,
        *, click_count: int = 0,
    ) -> None:
        state = self._page_state(handle)
        params: dict[str, Any] = {
            "type": event_type,
            "x": float(x), "y": float(y), "button": "left",
        }
        if click_count:
            params["clickCount"] = int(click_count)
        self._cdp(
            state, "Input.dispatchMouseEvent", params,
            session_id=handle.session_id,
        )

    def mouse_click(
        self, handle: Any, x: float, y: float, click_count: int = 1
    ) -> None:
        count = max(1, int(click_count))
        self._dispatch_mouse(
            handle, "mousePressed", x, y, click_count=count
        )
        self._dispatch_mouse(
            handle, "mouseReleased", x, y, click_count=count
        )
        self._pump_events(self._page_state(handle))

    def mouse_move(self, handle: Any, x: float, y: float) -> None:
        self._dispatch_mouse(handle, "mouseMoved", x, y)

    def mouse_drag(
        self, handle: Any, x1: float, y1: float, x2: float, y2: float
    ) -> None:
        self._dispatch_mouse(handle, "mouseMoved", x1, y1)
        self._dispatch_mouse(handle, "mousePressed", x1, y1, click_count=1)
        self._dispatch_mouse(handle, "mouseMoved", x2, y2)
        self._dispatch_mouse(handle, "mouseReleased", x2, y2, click_count=1)

    def focus_element(
        self, handle: Any, element: Any, timeout_ms: int
    ) -> None:
        self._call_on(element, "function() { this.focus(); }")

    def set_checked(
        self, handle: Any, element: Any, checked: bool, timeout_ms: int
    ) -> None:
        changed = self._call_on(
            element,
            "function(want) { if (this.checked !== want) {"
            " this.click(); } return this.checked === want; }",
            [bool(checked)],
        )
        if not changed:
            raise BrowserException(
                "Could not set the element's checked state",
                code=BrowserErrorCode.OPERATION_FAILED,
            )

    def wait_for(
        self, handle: Any, spec: dict[str, Any], timeout_ms: int
    ) -> None:
        kind = (spec or {}).get("kind")

        def check() -> bool:
            if kind == "element_present":
                return bool(
                    self.query_elements(
                        handle, spec.get("locator") or {}, 1
                    )
                )
            if kind == "element_hidden":
                elements = self.query_elements(
                    handle, spec.get("locator") or {}, 1
                )
                if not elements:
                    return True
                info = self.element_info(handle, elements[0])
                return not info.get("visible", False)
            if kind == "text_present":
                return str(spec.get("text") or "") in self.page_text(handle)
            if kind == "url_contains":
                return str(spec.get("value") or "") in self.page_url(handle)
            if kind == "title_contains":
                return str(spec.get("value") or "") in self.page_title(handle)
            raise BrowserException(
                f"Unknown wait condition {kind!r}",
                code=BrowserErrorCode.INVALID_LOCATOR,
            )

        deadline = time.monotonic() + max(1, int(timeout_ms)) / 1000.0
        while True:
            if check():
                return
            if time.monotonic() >= deadline:
                raise BrowserException(
                    f"Timed out waiting for {kind}",
                    code=BrowserErrorCode.TIMEOUT,
                )
            self._pump_events(self._page_state(handle))
            time.sleep(0.1)

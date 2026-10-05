"""Afnan Browser Runtime — owner of browser lifecycle and state.

The runtime sits between the BrowserController and a concrete
:class:`BrowserEngineAdapter`::

    Afnan Agent → Browser Tools → BrowserController
        → AfnanBrowserRuntime → BrowserEngineAdapter
        → PlaywrightAdapter → Chromium

The runtime owns: browser lifecycle (start/stop/restart),
engine-level session state (which tabs exist, where they are),
persistent profiles, an event stream the agent and logging can
consume, capability discovery, and crash detection/recovery.
No engine types cross this boundary — handles stay opaque,
state is the serializable models in ``models.py``, and failures
are structured BrowserExceptions with Afnan error codes.

Swapping Playwright for a future Afnan Chromium adapter means
writing one new adapter; nothing above this layer changes.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from afnan_ai.browser.base import BrowserErrorCode, BrowserException
from afnan_ai.browser.engine import BrowserEngineAdapter
from afnan_ai.browser.models import (
    BrowserProfile,
    BrowserResult,
    BrowserSession,
    BrowserTab,
)
from afnan_ai.log_config import get_logger
from afnan_ai.redaction import redact_text, redact_value

logger = get_logger(__name__)

#: Event types the runtime emits (see ``subscribe``/``events``).
EVENT_TYPES = (
    "browser_started", "browser_stopped", "tab_created",
    "tab_closed", "tab_selected", "navigation_started",
    "navigation_completed", "page_changed", "popup_detected",
    "download_started", "download_completed", "browser_error",
    "browser_recovered",
)

_SENSITIVE_PREF_PARTS = (
    "password", "passwd", "token", "secret", "cookie", "auth",
    "credential", "api_key",
)


def _sanitize_preferences(
    preferences: dict[str, Any] | None,
) -> dict[str, Any]:
    """Drop credential-shaped keys from profile preferences."""
    clean: dict[str, Any] = {}
    for key, value in (preferences or {}).items():
        lowered = str(key).lower()
        if any(part in lowered for part in _SENSITIVE_PREF_PARTS):
            continue
        clean[str(key)] = value
    return clean


class AfnanBrowserRuntime:
    """Owns the browser engine and its recoverable state."""

    def __init__(
        self,
        adapter: BrowserEngineAdapter | None = None,
        *,
        runtime_dir: str | Path | None = None,
    ):
        if adapter is None:
            from afnan_ai.browser.backend import PlaywrightAdapter

            adapter = PlaywrightAdapter()
        self.adapter = adapter
        self.runtime_dir = (
            Path(runtime_dir).expanduser() if runtime_dir else None
        )
        self.session = BrowserSession(
            session_id=uuid.uuid4().hex[:12],
            adapter_name=adapter.name,
        )
        self.previous_session: BrowserSession | None = None
        self._tabs: dict[int, BrowserTab] = {}
        self._tab_counter = 0
        self._profiles: dict[str, BrowserProfile] = {}
        self._events: list[dict[str, Any]] = []
        self._subscribers: list[Callable[[dict[str, Any]], Any]] = []
        self._seen_downloads: dict[str, str] = {}
        self._browser_name = "chromium"
        self._headless = True
        self._load_profiles()

    # -- identity -----------------------------------------------------

    @property
    def name(self) -> str:
        return self.adapter.name

    @property
    def running(self) -> bool:
        return self.session.status == "running"

    # -- events --------------------------------------------------------

    def emit(self, event_type: str, **data: Any) -> dict[str, Any]:
        event = {
            "type": event_type,
            "at": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session.session_id,
            "data": redact_value(dict(data)),
        }
        self._events.append(event)
        if len(self._events) > 200:
            del self._events[:-200]
        logger.info("browser event: %s", event_type)
        for subscriber in list(self._subscribers):
            try:
                subscriber(dict(event))
            except Exception:
                pass  # a listener must never break the runtime
        return event

    def subscribe(
        self, callback: Callable[[dict[str, Any]], Any]
    ) -> Callable[[dict[str, Any]], Any]:
        self._subscribers.append(callback)
        return callback

    def events(self, limit: int = 50) -> list[dict[str, Any]]:
        return [dict(e) for e in self._events[-max(1, int(limit)):]]

    # -- lifecycle -----------------------------------------------------

    def start(
        self, browser: str = "chromium", headless: bool = True
    ) -> BrowserResult:
        if self.running:
            return BrowserResult(
                True, "start",
                message="Browser is already running",
                data={"session": self.session.to_dict()},
            )
        self._browser_name = browser
        self._headless = headless
        try:
            self.adapter.start(browser, headless)
        except BrowserException as e:
            self.emit(
                "browser_error", code=e.error.code.value,
                message=e.error.message,
            )
            if e.error.code == BrowserErrorCode.BROWSER_UNAVAILABLE:
                raise  # engine missing is its own distinct code
            raise BrowserException(
                f"Browser failed to start: {e.error.message}",
                code=BrowserErrorCode.STARTUP_FAILED,
                details={"adapter": self.adapter.name,
                         "cause": e.error.code.value},
            ) from e
        except Exception as e:
            self.emit("browser_error", code="startup_failed",
                      message=str(e))
            raise BrowserException(
                f"Browser failed to start: {e}",
                code=BrowserErrorCode.STARTUP_FAILED,
                details={"adapter": self.adapter.name},
            ) from e
        self.session.status = "running"
        self.session.adapter_name = self.adapter.name
        self.session.touch()
        self.emit(
            "browser_started", browser=browser,
            adapter=self.adapter.name,
        )
        return BrowserResult(
            True, "start",
            data={"session": self.session.to_dict()},
        )

    def connect(self, endpoint: str) -> BrowserResult:
        try:
            self.adapter.connect(endpoint)
        except BrowserException:
            raise
        except Exception as e:
            raise BrowserException(
                f"Could not connect to browser: {e}",
                code=BrowserErrorCode.CONNECTION_FAILED,
            ) from e
        self.session.status = "running"
        self.session.adapter_name = self.adapter.name
        self.session.touch()
        self.emit("browser_started", connected=True)
        return BrowserResult(
            True, "connect",
            data={"session": self.session.to_dict()},
        )

    def stop(self) -> BrowserResult:
        try:
            self.adapter.stop()
        except Exception:
            pass  # stopping must never raise
        self._tabs.clear()
        self.session.tabs = []
        self.session.window.tab_ids = []
        self.session.status = "stopped"
        self.session.touch()
        self.save_session()
        self.emit("browser_stopped")
        return BrowserResult(True, "stop")

    def restart(self) -> BrowserResult:
        self.stop()
        return self.start(self._browser_name, self._headless)

    # -- crash detection & recovery -------------------------------------

    def check_health(self) -> bool:
        """True when the engine is responsive.

        A dead engine flips the session to ``crashed`` (and
        preserves its tab records) instead of failing silently.
        """
        if not self.running:
            return False
        try:
            alive = bool(self.adapter.is_alive())
        except Exception:
            alive = False
        if not alive and self.session.status != "crashed":
            self.session.status = "crashed"
            self.session.touch()
            self.save_session()
            self.emit(
                "browser_error", code="browser_crashed",
                message="Browser process stopped responding",
            )
        return alive

    def recover(self) -> BrowserResult:
        """Restart a crashed engine and report recoverable tabs.

        The previous session's tabs are *identified* (URL, title,
        purpose) so the agent can decide what to reopen — the
        task is never blindly restarted.
        """
        recoverable = [t.to_dict() for t in self.session.tabs]
        previous_id = self.session.session_id
        try:
            self.adapter.stop()
        except Exception:
            pass
        self._tabs.clear()
        self.session.tabs = []
        self.session.window.tab_ids = []
        self.session.status = "stopped"
        result = self.start(self._browser_name, self._headless)
        self.emit(
            "browser_recovered",
            previous_session_id=previous_id,
            recoverable_tabs=len(recoverable),
        )
        return BrowserResult(
            True, "recover",
            message=(
                "Browser restarted; "
                f"{len(recoverable)} tab(s) are recoverable"
            ),
            data={
                "previous_session_id": previous_id,
                "recoverable_tabs": recoverable,
                "session": result.data["session"],
            },
        )

    # -- engine delegation (controller-facing surface) ------------------

    def _call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the adapter, preserving its errors.

        Structured BrowserExceptions pass through untouched (the
        controller's error fidelity is unchanged); an unexpected
        exception while the engine is dead marks the session
        crashed and surfaces as ``browser_crashed`` — anything
        else propagates to the caller's own structured handling,
        exactly as before the runtime existed.
        """
        try:
            return fn(*args, **kwargs)
        except BrowserException:
            raise
        except Exception as e:
            if self.session.status != "stopped" and (
                not self._alive_quietly()
            ):
                if self.session.status != "crashed":
                    self.session.status = "crashed"
                    self.session.touch()
                    self.save_session()
                self.emit(
                    "browser_error", code="browser_crashed",
                    message=str(e),
                )
                raise BrowserException(
                    f"Browser engine crashed: {e}",
                    code=BrowserErrorCode.BROWSER_CRASHED,
                ) from e
            raise

    def _alive_quietly(self) -> bool:
        try:
            return bool(self.adapter.is_alive())
        except Exception:
            return False

    def new_page(self) -> Any:
        handle = self._call(self.adapter.new_page)
        tab = self._register_tab(handle)
        self.emit(
            "tab_created", tab_id=tab.tab_id,
            profile_id=tab.profile_id,
        )
        return handle

    def close_page(self, handle: Any) -> None:
        tab = self._tabs.pop(id(handle), None)
        self._call(self.adapter.close_page, handle)
        self._sync_session_tabs()
        self.emit(
            "tab_closed", tab_id=tab.tab_id if tab else ""
        )

    def list_pages(self) -> list[Any]:
        pages = self._call(self.adapter.list_pages)
        known = set(self._tabs)
        for handle in pages:
            if id(handle) not in known:
                tab = self._register_tab(handle)
                self.emit(
                    "popup_detected", tab_id=tab.tab_id
                )
        return pages

    def goto(self, handle: Any, url: str) -> None:
        self.emit("navigation_started", url=redact_text(url))
        self._call(self.adapter.goto, handle, url)
        self._refresh_tab(handle)
        tab = self._tabs.get(id(handle))
        self.emit(
            "navigation_completed",
            url=tab.url if tab else redact_text(url),
            title=tab.title if tab else "",
        )

    def go_back(self, handle: Any) -> None:
        self._call(self.adapter.go_back, handle)
        self._refresh_tab(handle)

    def go_forward(self, handle: Any) -> None:
        self._call(self.adapter.go_forward, handle)
        self._refresh_tab(handle)

    def reload(self, handle: Any) -> None:
        self._call(self.adapter.reload, handle)
        self._refresh_tab(handle)

    def page_url(self, handle: Any) -> str:
        return self._call(self.adapter.page_url, handle)

    def page_title(self, handle: Any) -> str:
        return self._call(self.adapter.page_title, handle)

    def page_text(self, handle: Any) -> str:
        return self._call(self.adapter.page_text, handle)

    def interactive_elements(
        self, handle: Any, limit: int
    ) -> list[Any]:
        return self._call(
            self.adapter.interactive_elements, handle, limit
        )

    def query_elements(
        self, handle: Any, locator: dict[str, Any], limit: int
    ) -> list[Any]:
        return self._call(
            self.adapter.query_elements, handle, locator, limit
        )

    def element_info(self, handle: Any, element: Any) -> dict[str, Any]:
        return self._call(self.adapter.element_info, handle, element)

    def click_element(
        self, handle: Any, element: Any, timeout_ms: int
    ) -> None:
        return self._call(
            self.adapter.click_element, handle, element, timeout_ms
        )

    def fill_element(
        self, handle: Any, element: Any, text: str, timeout_ms: int
    ) -> None:
        return self._call(
            self.adapter.fill_element, handle, element, text,
            timeout_ms,
        )

    def clear_element(
        self, handle: Any, element: Any, timeout_ms: int
    ) -> None:
        return self._call(
            self.adapter.clear_element, handle, element, timeout_ms
        )

    def select_option(
        self, handle: Any, element: Any, value: str, timeout_ms: int
    ) -> None:
        return self._call(
            self.adapter.select_option, handle, element, value,
            timeout_ms,
        )

    def press_key(
        self, handle: Any, element: Any, key: str, timeout_ms: int
    ) -> None:
        return self._call(
            self.adapter.press_key, handle, element, key, timeout_ms
        )

    def scroll_page(self, handle: Any, dx: int, dy: int) -> None:
        return self._call(self.adapter.scroll_page, handle, dx, dy)

    def scroll_to_element(self, handle: Any, element: Any) -> None:
        return self._call(
            self.adapter.scroll_to_element, handle, element
        )

    def set_input_files(
        self, handle: Any, element: Any, path: str, timeout_ms: int
    ) -> None:
        return self._call(
            self.adapter.set_input_files, handle, element, path,
            timeout_ms,
        )

    def screenshot(self, handle: Any) -> bytes:
        return self._call(self.adapter.screenshot, handle)

    def wait_for(
        self, handle: Any, spec: dict[str, Any], timeout_ms: int
    ) -> None:
        return self._call(
            self.adapter.wait_for, handle, spec, timeout_ms
        )

    def page_probe(self, handle: Any) -> dict[str, Any]:
        return self._call(self.adapter.page_probe, handle)

    def page_content(self, handle: Any) -> dict[str, Any]:
        return self._call(self.adapter.page_content, handle)

    def accessibility_snapshot(self, handle: Any) -> Any:
        return self._call(self.adapter.accessibility_snapshot, handle)

    def network_events(self, handle: Any) -> list[dict[str, Any]]:
        return self._call(self.adapter.network_events, handle)

    def dialogs(self, handle: Any) -> list[dict[str, Any]]:
        return self._call(self.adapter.dialogs, handle)

    def downloads(self) -> list[dict[str, Any]]:
        records = self._call(self.adapter.downloads)
        for record in records:
            rid = str(record.get("id") or "")
            state = str(record.get("state") or "")
            previous = self._seen_downloads.get(rid)
            if previous is None:
                self.emit(
                    "download_started",
                    filename=record.get("filename", ""),
                )
            if state in ("completed", "failed") and previous != state:
                self.emit(
                    "download_completed",
                    filename=record.get("filename", ""),
                    state=state,
                )
            self._seen_downloads[rid] = state
        return records

    def create_profile_context(
        self, name: str, options: dict[str, Any]
    ) -> str:
        options = dict(options or {})
        profile = self._profiles.get(name)
        if profile is not None and profile.storage_dir:
            # Persistent profile: the engine stores cookies /
            # storage in the profile's own directory.
            options.setdefault("user_data_dir", profile.storage_dir)
        return self._call(
            self.adapter.create_profile_context, name, options
        )

    def set_active_profile(self, name: str) -> None:
        self._call(self.adapter.set_active_profile, name)
        self.session.profile_id = name
        self.session.touch()

    # -- tab/session state ----------------------------------------------

    def _register_tab(self, handle: Any) -> BrowserTab:
        self._tab_counter += 1
        tab = BrowserTab(
            tab_id=f"rt_{self._tab_counter}",
            profile_id=self.session.profile_id,
        )
        for other in self._tabs.values():
            other.active = False
        tab.active = True
        self._tabs[id(handle)] = tab
        self._refresh_tab(handle)
        self._sync_session_tabs()
        return tab

    def _refresh_tab(self, handle: Any) -> None:
        tab = self._tabs.get(id(handle))
        if tab is None:
            return
        try:
            tab.url = self.adapter.page_url(handle)
        except Exception:
            pass
        try:
            tab.title = self.adapter.page_title(handle)
        except Exception:
            pass
        tab.updated_at = datetime.now(timezone.utc).isoformat()
        self.session.touch()

    def _sync_session_tabs(self) -> None:
        self.session.tabs = list(self._tabs.values())
        self.session.window.tab_ids = [
            t.tab_id for t in self.session.tabs
        ]
        self.session.window.profile_id = self.session.profile_id
        self.session.touch()

    def tab_records(self) -> list[dict[str, Any]]:
        """Serializable engine-level tab state (safe for
        AgentState/checkpoints — no handles, redacted URLs)."""
        return [t.to_dict() for t in self.session.tabs]

    # -- profiles (persistent registry) -----------------------------------

    def _profiles_path(self) -> Path | None:
        if self.runtime_dir is None:
            return None
        return self.runtime_dir / "profiles.json"

    def _load_profiles(self) -> None:
        path = self._profiles_path()
        if path is None or not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data.get("profiles") or []:
                profile = BrowserProfile.from_dict(entry)
                if profile.name:
                    self._profiles[profile.name] = profile
        except Exception:
            logger.warning("could not load browser profiles registry")

    def _save_profiles(self) -> None:
        path = self._profiles_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "profiles": [
                    p.to_dict() for p in self._profiles.values()
                ]
            }
            path.write_text(
                json.dumps(redact_value(payload), indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.warning("could not save browser profiles registry")

    def create_profile(
        self,
        name: str,
        preferences: dict[str, Any] | None = None,
        *,
        persistent: bool = True,
    ) -> BrowserProfile:
        """Create (and persist) an isolated browser profile."""
        name = str(name or "").strip()
        if not name or "/" in name or "\\" in name or name in (".", ".."):
            raise BrowserException(
                f"Invalid browser profile name {name!r}",
                code=BrowserErrorCode.PROFILE_ERROR,
            )
        if name in self._profiles:
            raise BrowserException(
                f"Browser profile {name!r} already exists",
                code=BrowserErrorCode.PROFILE_ERROR,
                details={"profile": name},
            )
        storage_dir = ""
        if persistent and self.runtime_dir is not None:
            storage = self.runtime_dir / "profiles" / name
            storage.mkdir(parents=True, exist_ok=True)
            storage_dir = str(storage)
        profile = BrowserProfile(
            profile_id=name,
            name=name,
            storage_dir=storage_dir,
            preferences=_sanitize_preferences(preferences),
            persistent=persistent,
        )
        self._profiles[name] = profile
        self._save_profiles()
        return profile

    def list_profiles(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self._profiles.values()]

    def get_profile(self, name: str) -> BrowserProfile | None:
        return self._profiles.get(name)

    def open_profile(self, name: str) -> BrowserProfile:
        """Make *name* the engine's active profile."""
        profile = self._profiles.get(name)
        if profile is None and name != "default":
            raise BrowserException(
                f"Unknown browser profile {name!r}",
                code=BrowserErrorCode.PROFILE_ERROR,
                details={"profile": name},
            )
        self.create_profile_context(name, {})
        self.set_active_profile(name)
        return profile or BrowserProfile(
            profile_id="default", name="default"
        )

    def delete_profile(self, name: str) -> None:
        profile = self._profiles.pop(name, None)
        if profile is None:
            raise BrowserException(
                f"Unknown browser profile {name!r}",
                code=BrowserErrorCode.PROFILE_ERROR,
                details={"profile": name},
            )
        self._save_profiles()
        if profile.storage_dir and self.runtime_dir is not None:
            storage = Path(profile.storage_dir)
            profiles_root = (self.runtime_dir / "profiles").resolve()
            try:
                if storage.resolve().is_relative_to(profiles_root):
                    shutil.rmtree(storage, ignore_errors=True)
            except Exception:
                pass

    # -- session persistence ----------------------------------------------

    def _session_path(self) -> Path | None:
        if self.runtime_dir is None:
            return None
        return self.runtime_dir / "session.json"

    def save_session(self) -> str | None:
        """Persist the recoverable session state (redacted)."""
        path = self._session_path()
        if path is None:
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    redact_value(self.session.to_dict()), indent=2
                ),
                encoding="utf-8",
            )
            return str(path)
        except Exception:
            logger.warning("could not save browser session")
            return None

    def load_session(self) -> BrowserSession | None:
        """Load the previously persisted session, if any.

        The loaded record becomes ``previous_session`` — the
        foundation for post-restart recovery (its tabs are the
        recoverable ones); the live session is untouched.
        """
        path = self._session_path()
        if path is None or not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.previous_session = BrowserSession.from_dict(data)
            return self.previous_session
        except Exception:
            logger.warning("could not load browser session")
            return None

    # -- capabilities -------------------------------------------------------

    def capabilities(self) -> dict[str, bool]:
        """Afnan-level capabilities (never engine features)."""
        caps = {
            "tabs": True,
            "sessions": True,
            "events": True,
            "crash_recovery": True,
            "persistent_profiles": False,
            "accessibility": False,
            "screenshots": False,
            "downloads": False,
            "uploads": False,
            "semantic_locator": False,
            "iframe": False,
            "shadow_dom": False,
            "network_observation": False,
            "visual_interaction": False,
        }
        try:
            caps.update(self.adapter.capabilities())
        except Exception:
            pass
        return caps

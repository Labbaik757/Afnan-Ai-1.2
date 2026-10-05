"""BrowserController — programmatic browser control for Afnan.

The controller owns a browser *session*: launching/connecting,
tabs (create/select/close), navigation (URL, back/forward/reload),
reading the current page state (URL + title) and shutdown.  It
works the same on Windows, macOS and Linux because every
browser-specific action goes through a
:class:`~afnan_ai.browser.backend.BrowserBackend` (Playwright by
default); the controller itself contains no OS- or driver-specific
code, and the core Agent never sees a browser API — it only runs
the browser Tools built on this controller.

Failure model: operations either return a plain dict describing
the new state, or raise :class:`BrowserException` with a
structured ``{code, message, details}`` error — browser
unavailable, connection failed, browser not started, invalid tab
or invalid URL are all expected, reported failures, never crashes
and never silent successes.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from afnan_ai.browser.accessibility import derive_nodes, normalize_snapshot
from afnan_ai.browser.backend import (
    SUPPORTED_BROWSERS,
    BrowserBackend,
)
from afnan_ai.browser.base import (
    BrowserErrorCode,
    BrowserException,
    ElementInfo,
    PageState,
    TabInfo,
)
from afnan_ai.browser.challenge import detect_challenge
from afnan_ai.browser.downloads import DownloadManager
from afnan_ai.browser.network import summarize_network
from afnan_ai.browser.ratelimit import (
    ActionPacer,
    RateLimitDetector,
    RateLimitPolicy,
)
from afnan_ai.browser.runtime import AfnanBrowserRuntime
from afnan_ai.browser.security import (
    ActionRisk,
    ApprovalGate,
    Sensitivity,
    classify_action,
)
from afnan_ai.browser.semantics import rank as rank_semantic
from afnan_ai.browser.session import SessionManager
from afnan_ai.log_config import get_logger
from afnan_ai.redaction import MASK, redact_text

logger = get_logger(__name__)

#: locator strategies, most stable/specific first
_LOCATOR_KEYS = ("selector", "test_id", "label", "placeholder", "role", "text", "name")


@dataclass
class _Tab:
    tab_id: str
    handle: Any
    url: str = ""
    title: str = ""


class BrowserController:
    """Control one browser session through a backend.

    Constructing a controller never launches anything (and never
    fails because a backend is missing); call :meth:`launch` or
    :meth:`connect` first.
    """

    DEFAULT_TIMEOUT_MS = 5000

    def __init__(
        self,
        backend: BrowserBackend | None = None,
        *,
        browser: str = "chromium",
        headless: bool = False,
        approval_gate: ApprovalGate | None = None,
        challenge_guard: bool = True,
        challenge_handler=None,
        challenge_wait_ms: int = 120_000,
        rate_policy: RateLimitPolicy | None = None,
        runtime: AfnanBrowserRuntime | None = None,
    ):
        # The controller talks to the Afnan Browser Runtime,
        # never to an engine adapter directly: the runtime owns
        # lifecycle/session state and routes through the adapter
        # (Playwright today, a future Afnan Chromium adapter
        # later) without this class changing.
        if runtime is not None:
            self.runtime = runtime
        elif backend is not None:
            self.runtime = AfnanBrowserRuntime(adapter=backend)
        else:
            self.runtime = AfnanBrowserRuntime()
        self._backend = self.runtime.adapter
        # Sensitive-action approval gate (security layer): a
        # default gate is always present, so purchases, sends,
        # destructive clicks, uploads etc. never run without a
        # human approver; pass a gate to configure it.
        self.approval_gate = approval_gate or ApprovalGate()
        self._default_browser = browser
        self._default_headless = headless
        self._running = False
        self._browser_name: str | None = None
        self._endpoint: str | None = None
        self._tabs: dict[str, _Tab] = {}
        self._active_tab_id: str | None = None
        self._counter = 0
        # element references issued by find_elements, per page
        # generation: a ref dies (stale_element) when its page
        # navigates or reloads
        self._elements: dict[str, dict[str, Any]] = {}
        self._element_counter = 0
        self._generations: dict[str, int] = {}
        # last observation fingerprint per tab (page-change detection)
        self._observations: dict[str, str] = {}
        # Confidence of controller-issued element refs that came
        # from a natural-language (semantic) search.  Actions on
        # sub-0.8 targets are gated through human approval.
        self._semantic_confidence: dict[str, float] = {}
        # task-level purpose per tab (multi-tab task management)
        self._tab_purposes: dict[str, str] = {}
        # navigation history (session manager): bounded, redacted
        self._history: list[dict[str, Any]] = []
        self._history_seq = 0
        # CAPTCHA guard: actions stop with human_required on
        # human-check pages; detection only, never a bypass
        self._challenge_guard = challenge_guard
        # Optional human-in-the-loop challenge flow: when a
        # challenge is detected, the handler is asked (approval);
        # if the human solves the check in the browser, the
        # action resumes automatically once the page clears.
        self.challenge_handler = challenge_handler
        self._challenge_wait_ms = challenge_wait_ms
        # rate-limit / anti-bot awareness: page-signal detection
        # plus optional pacing of the controller's own actions
        self._rate_policy = rate_policy or RateLimitPolicy()
        self._pacer = ActionPacer(self._rate_policy)
        # browser profiles: isolated contexts, one active at a
        # time; each profile's tabs are stashed on switch so
        # cookies/storage/tabs never leak across profiles
        self._profiles: dict[str, dict[str, Any]] = {
            "default": {"name": "default", "preferences": {}}
        }
        self._active_profile = "default"
        self._profile_tabs: dict[
            str, tuple[dict[str, _Tab], str | None]
        ] = {}
        # download + session managers (browser-layer services)
        self.download_manager = DownloadManager(self.runtime)
        self.session_manager = SessionManager(self)

    # -- introspection ----------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def browser_name(self) -> str | None:
        return self._browser_name

    @property
    def active_tab_id(self) -> str | None:
        return self._active_tab_id

    @property
    def backend_name(self) -> str:
        return self.runtime.name

    def session_history(self) -> list[dict[str, Any]]:
        """Recorded navigation history (redacted URLs)."""
        return [dict(entry) for entry in self._history]

    def _record_history(self, action: str, tab: _Tab) -> None:
        """Append a navigation event to the session history.

        URLs are redacted at record time so tokens never reach the
        history, logs or LLM context.
        """
        self._history_seq += 1
        self._history.append(
            {
                "seq": self._history_seq,
                "at": datetime.now(timezone.utc).isoformat(),
                "tab_id": tab.tab_id,
                "action": action,
                "url": redact_text(tab.url or ""),
                "title": tab.title or "",
                "purpose": self._tab_purposes.get(tab.tab_id, ""),
                "profile": self._active_profile,
            }
        )
        # bounded: keep the most recent 500 events
        if len(self._history) > 500:
            del self._history[:-500]

    # -- lifecycle ------------------------------------------------------------
    def launch(
        self,
        browser: str | None = None,
        headless: bool | None = None,
    ) -> dict[str, Any]:
        """Launch the browser (idempotent) and report its state."""
        if self._running:
            return self._session_info(launched=False)
        name = (browser or self._default_browser or "chromium").lower()
        if name not in SUPPORTED_BROWSERS:
            raise BrowserException(
                f"Unsupported browser {name!r}. Supported: "
                + ", ".join(SUPPORTED_BROWSERS),
                code=BrowserErrorCode.BROWSER_UNAVAILABLE,
                details={"browser": name, "supported": list(SUPPORTED_BROWSERS)},
            )
        try:
            self.runtime.start(
                name,
                self._default_headless if headless is None else headless,
            )
        except BrowserException:
            raise
        except Exception as e:
            raise BrowserException(
                f"Browser backend failed to launch: {e}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"browser": name},
            ) from e
        self._running = True
        self._browser_name = name
        self._endpoint = None
        logger.info("browser launched: %s", name)
        return self._session_info(launched=True)

    def connect(self, endpoint: str) -> dict[str, Any]:
        """Connect to an already-running browser (CDP endpoint)."""
        if self._running:
            return self._session_info(launched=False)
        endpoint = (endpoint or "").strip()
        if not endpoint:
            raise BrowserException(
                "A CDP endpoint (e.g. http://localhost:9222) is "
                "required to connect",
                code=BrowserErrorCode.CONNECTION_FAILED,
            )
        try:
            self.runtime.connect(endpoint)
        except BrowserException:
            raise
        except Exception as e:
            raise BrowserException(
                f"Browser backend failed to connect: {e}",
                code=BrowserErrorCode.CONNECTION_FAILED,
                details={"endpoint": endpoint},
            ) from e
        self._running = True
        self._browser_name = "chromium"
        self._endpoint = endpoint
        logger.info("browser connected: %s", endpoint)
        return self._session_info(launched=True)

    def shutdown(self) -> dict[str, Any]:
        """Close all tabs and the browser.  Safe to call anytime."""
        was_running = self._running
        self._tabs.clear()
        self._elements.clear()
        self._generations.clear()
        self._active_tab_id = None
        self._running = False
        self._endpoint = None
        try:
            self.runtime.stop()
        except Exception as e:  # shutdown must not crash the caller
            logger.warning("browser backend stop failed: %s", e)
        logger.info("browser shut down (was running: %s)", was_running)
        return {"closed": True, "was_running": was_running}

    # -- tabs -----------------------------------------------------------------
    def new_tab(
        self, url: str | None = None, *, purpose: str | None = None
    ) -> dict[str, Any]:
        """Create a tab (optionally navigating it) and select it.

        ``purpose`` records *why* the task opened this tab (e.g.
        "pricing research"), so multi-tab work can always tell
        which tab belongs to which line of work.
        """
        self._require_running()
        try:
            handle = self.runtime.new_page()
        except BrowserException:
            raise
        except Exception as e:
            raise BrowserException(
                f"Could not open a new tab: {e}",
                code=BrowserErrorCode.OPERATION_FAILED,
            ) from e
        self._counter += 1
        tab = _Tab(tab_id=f"tab_{self._counter}", handle=handle)
        self._tabs[tab.tab_id] = tab
        self._generations[tab.tab_id] = 0
        self._active_tab_id = tab.tab_id
        if purpose:
            self._tab_purposes[tab.tab_id] = str(purpose)
        if url:
            self._goto(tab, url)
        self._refresh(tab)
        self._record_history("new_tab", tab)
        self.runtime.emit(
            "tab_created", tab_id=tab.tab_id,
            purpose=self._tab_purposes.get(tab.tab_id, ""),
            url=tab.url,
        )
        logger.info("browser tab created: %s", tab.tab_id)
        return self._tab_info(tab).to_dict()

    def set_tab_purpose(self, tab_id: str, purpose: str) -> dict[str, Any]:
        """Label a tab with its task-level purpose."""
        tab = self._resolve_tab(tab_id)
        self._tab_purposes[tab.tab_id] = str(purpose)
        self._refresh(tab, quiet=True)
        return self._tab_info(tab).to_dict()

    def tab_purpose(self, tab_id: str) -> str:
        tab = self._resolve_tab(tab_id)
        return self._tab_purposes.get(tab.tab_id, "")

    def list_tabs(self) -> list[dict[str, Any]]:
        """Snapshot of all open tabs (empty when none are open;
        popup pages appear here after adoption)."""
        self.sync_tabs()
        for tab in self._tabs.values():
            self._refresh(tab, quiet=True)
        return [self._tab_info(t).to_dict() for t in self._tabs.values()]

    def sync_tabs(self) -> list[dict[str, Any]]:
        """Adopt pages the driver knows about but the controller
        did not create (popups / new-window links) as regular tabs,
        so they can be listed, selected and observed like any tab."""
        adopted = []
        try:
            handles = self.runtime.list_pages()
        except Exception:
            handles = []
        for handle in handles or []:
            if any(handle is tab.handle for tab in self._tabs.values()):
                continue
            self._counter += 1
            tab = _Tab(tab_id=f"tab_{self._counter}", handle=handle)
            self._tabs[tab.tab_id] = tab
            self._generations[tab.tab_id] = 0
            self._refresh(tab, quiet=True)
            adopted.append(self._tab_info(tab).to_dict())
            self.runtime.emit(
                "popup_detected", tab_id=tab.tab_id, url=tab.url
            )
            logger.info("browser adopted popup tab: %s", tab.tab_id)
        return adopted

    def select_tab(self, tab_id: str) -> dict[str, Any]:
        """Make *tab_id* the active tab."""
        tab = self._resolve_tab(tab_id, allow_default=False)
        self._active_tab_id = tab.tab_id
        self._refresh(tab)
        self._record_history("select_tab", tab)
        self.runtime.emit(
            "tab_selected", tab_id=tab.tab_id, url=tab.url
        )
        return self._tab_info(tab).to_dict()

    def close_tab(self, tab_id: str | None = None) -> dict[str, Any]:
        """Close a tab (the active one by default)."""
        tab = self._resolve_tab(tab_id)
        try:
            self.runtime.close_page(tab.handle)
        except Exception as e:
            raise BrowserException(
                f"Could not close tab {tab.tab_id!r}: {e}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"tab_id": tab.tab_id},
            ) from e
        del self._tabs[tab.tab_id]
        self._generations.pop(tab.tab_id, None)
        self._tab_purposes.pop(tab.tab_id, None)
        self._record_history("close_tab", tab)
        self._elements = {
            ref: rec
            for ref, rec in self._elements.items()
            if rec["tab_id"] != tab.tab_id
        }
        if self._active_tab_id == tab.tab_id:
            self._active_tab_id = (
                next(reversed(self._tabs)) if self._tabs else None
            )
        self.runtime.emit("tab_closed", tab_id=tab.tab_id)
        logger.info("browser tab closed: %s", tab.tab_id)
        return {"closed_tab": tab.tab_id, "active_tab": self._active_tab_id}

    # -- navigation ---------------------------------------------------------------
    def navigate(
        self, url: str, tab_id: str | None = None
    ) -> dict[str, Any]:
        """Navigate a tab (active by default) to *url*."""
        self._require_running()
        if not self._tabs:
            # Friendly agent behaviour: navigating with no tab open
            # creates one instead of failing.
            self.new_tab()
        tab = self._resolve_tab(tab_id)
        self._goto(tab, url)
        return self.current_page(tab.tab_id)

    def back(self, tab_id: str | None = None) -> dict[str, Any]:
        return self._history_move("back", tab_id)

    def forward(self, tab_id: str | None = None) -> dict[str, Any]:
        return self._history_move("forward", tab_id)

    def reload(self, tab_id: str | None = None) -> dict[str, Any]:
        tab = self._resolve_tab(tab_id)
        try:
            self.runtime.reload(tab.handle)
        except Exception as e:
            raise BrowserException(
                f"Could not reload tab {tab.tab_id!r}: {e}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"tab_id": tab.tab_id},
            ) from e
        self._generations[tab.tab_id] = (
            self._generations.get(tab.tab_id, 0) + 1
        )
        return self.current_page(tab.tab_id)

    # -- page state -------------------------------------------------------------
    def current_page(
        self, tab_id: str | None = None
    ) -> dict[str, Any]:
        """Current page state (tab id, URL, title) of a tab."""
        tab = self._resolve_tab(tab_id)
        self._refresh(tab)
        return PageState(
            tab_id=tab.tab_id, url=tab.url, title=tab.title
        ).to_dict()

    def title(self, tab_id: str | None = None) -> str:
        return self.current_page(tab_id)["title"]

    def current_url(self, tab_id: str | None = None) -> str:
        return self.current_page(tab_id)["url"]

    # -- element interaction ------------------------------------------------
    # Every interaction validates its target first: the element
    # must exist, belong to the current page (not stale), and be
    # usable for the action (visible/enabled/editable/...).  A
    # *target* is {"ref": "el_1"} (from find_elements) or
    # {"locator": {...}} / a plain locator dict with one or more of
    # selector / test_id / label / placeholder / role(+name) / text.
    def find_elements(
        self,
        locator: dict[str, Any] | None = None,
        *,
        tab_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find elements matching *locator* and return their
        identities + basic properties (each with a usable ref)."""
        tab = self._resolve_tab(tab_id)
        locator = self._validate_locator(locator)
        handles = self._query(tab, locator, limit)
        if not handles:
            raise BrowserException(
                f"No element matches locator {locator} on tab "
                f"{tab.tab_id!r}",
                code=BrowserErrorCode.ELEMENT_NOT_FOUND,
                details={"locator": locator, "tab_id": tab.tab_id},
            )
        return [self._register_element(tab, h).to_dict() for h in handles]

    def inspect_element(
        self,
        target: dict[str, Any] | None = None,
        *,
        tab_id: str | None = None,
    ) -> dict[str, Any]:
        """Read one element's identity and basic properties."""
        tab, record, info = self._prepare_interaction(tab_id, target)
        return info.to_dict()

    def click(
        self,
        target: dict[str, Any] | None = None,
        *,
        tab_id: str | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        """Validate and click the target element."""
        tab, record, info = self._prepare_interaction(tab_id, target)
        self._require_usable(info, "click")
        self._challenge_check(tab)
        self._ratelimit_check(tab)
        sensitivity = self._gate_check("browser_click", tab, info)
        self._do(
            lambda: self.runtime.click_element(
                tab.handle, record["handle"], self._timeout(timeout_ms)
            ),
            "click",
            record,
        )
        result = self._interaction_result(tab, "click", info)
        if sensitivity:
            result["sensitivity"] = sensitivity
        return result

    def type_text(
        self,
        target: dict[str, Any] | None,
        text: str,
        *,
        tab_id: str | None = None,
        clear_first: bool = False,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        """Type *text* into an input/textarea (optionally clearing
        it first)."""
        tab, record, info = self._prepare_interaction(tab_id, target)
        self._require_usable(info, "type into")
        self._challenge_check(tab)
        self._ratelimit_check(tab)
        if not info.editable:
            raise BrowserException(
                f"Element {info.ref} (<{info.tag}>) is not an "
                "editable field; cannot type into it",
                code=BrowserErrorCode.INVALID_ELEMENT,
                details={"ref": info.ref, "tag": info.tag},
            )
        sensitivity = self._gate_check(
            "browser_type", tab, info,
            arguments={
                "text": text,
                "field": info.attributes.get("id")
                or info.attributes.get("name", ""),
                "field_type": info.attributes.get("type", ""),
            },
        )
        if clear_first:
            self._do(
                lambda: self.runtime.clear_element(
                    tab.handle, record["handle"], self._timeout(timeout_ms)
                ),
                "clear",
                record,
            )
        self._do(
            lambda: self.runtime.fill_element(
                tab.handle, record["handle"], text, self._timeout(timeout_ms)
            ),
            "type into",
            record,
        )
        result = self._interaction_result(tab, "type", info)
        # A password's value is a secret: it is never echoed back
        # into results (and therefore never into AgentState/logs)
        is_password = (
            str(info.attributes.get("type", "")).lower() == "password"
        )
        result["value"] = MASK if is_password else text
        if sensitivity:
            result["sensitivity"] = sensitivity
        return result

    def clear_field(
        self,
        target: dict[str, Any] | None = None,
        *,
        tab_id: str | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        """Clear an input/textarea's current value."""
        tab, record, info = self._prepare_interaction(tab_id, target)
        self._require_usable(info, "clear")
        self._challenge_check(tab)
        self._ratelimit_check(tab)
        if not info.editable:
            raise BrowserException(
                f"Element {info.ref} (<{info.tag}>) is not an "
                "editable field; cannot clear it",
                code=BrowserErrorCode.INVALID_ELEMENT,
                details={"ref": info.ref, "tag": info.tag},
            )
        self._gate_check("browser_clear", tab, info)
        self._do(
            lambda: self.runtime.clear_element(
                tab.handle, record["handle"], self._timeout(timeout_ms)
            ),
            "clear",
            record,
        )
        return self._interaction_result(tab, "clear", info)

    def select_option(
        self,
        target: dict[str, Any] | None,
        value: str,
        *,
        tab_id: str | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        """Select *value* in a <select> dropdown element."""
        tab, record, info = self._prepare_interaction(tab_id, target)
        self._require_usable(info, "select an option in")
        self._challenge_check(tab)
        self._ratelimit_check(tab)
        if info.tag != "select":
            raise BrowserException(
                f"Element {info.ref} is a <{info.tag}>, not a "
                "<select>; cannot select an option in it",
                code=BrowserErrorCode.INVALID_ELEMENT,
                details={"ref": info.ref, "tag": info.tag},
            )
        self._gate_check("browser_select_option", tab, info)
        self._do(
            lambda: self.runtime.select_option(
                tab.handle, record["handle"], value,
                self._timeout(timeout_ms),
            ),
            "select option in",
            record,
        )
        result = self._interaction_result(tab, "select_option", info)
        result["value"] = value
        return result

    def press_key(
        self,
        key: str,
        target: dict[str, Any] | None = None,
        *,
        tab_id: str | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        """Press a keyboard key (e.g. "Enter", "Tab", "Escape",
        "Control+A") on an element, or on the page when no target
        is given."""
        if not key or not str(key).strip():
            raise BrowserException(
                "A non-empty key (e.g. 'Enter', 'Tab', 'Escape') "
                "is required",
                code=BrowserErrorCode.OPERATION_FAILED,
            )
        tab = self._resolve_tab(tab_id)
        self._challenge_check(tab)
        self._ratelimit_check(tab)
        element_handle = None
        info = None
        if target:
            _, record, info = self._prepare_interaction(tab_id, target)
            element_handle = record["handle"]
            self._gate_check(
                "browser_press_key", tab, info,
                arguments={"key": str(key)},
            )
        self._do(
            lambda: self.runtime.press_key(
                tab.handle, element_handle, str(key),
                self._timeout(timeout_ms),
            ),
            "press key",
            {"ref": info.ref if info else None},
        )
        result = {"action": "press_key", "key": str(key)}
        if info is not None:
            result["element"] = info.to_dict()
        result["page"] = self.current_page(tab.tab_id)
        return result

    def upload_file(
        self,
        target: dict[str, Any] | None,
        file_path: str,
        *,
        tab_id: str | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        """Upload a local file through an <input type=file>.

        Uploads are a sensitive action (a local file leaves the
        machine), so the approval gate runs before the file is
        touched; validation failures (wrong element, missing
        file) are structured errors, never partial uploads.
        """
        tab, record, info = self._prepare_interaction(tab_id, target)
        self._require_usable(info, "upload to")
        self._challenge_check(tab)
        self._ratelimit_check(tab)
        if info.tag != "input" or (
            str(info.attributes.get("type", "")).lower() != "file"
        ):
            raise BrowserException(
                f"Element {info.ref} (<{info.tag}> "
                f"type={info.attributes.get('type')!r}) is not a "
                "file input; cannot upload through it",
                code=BrowserErrorCode.INVALID_ELEMENT,
                details={"ref": info.ref, "tag": info.tag},
            )
        path = Path(str(file_path)).expanduser()
        if not path.is_file():
            raise BrowserException(
                f"Upload failed: file not found: {path}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"path": str(path)},
            )
        sensitivity = self._gate_check(
            "browser_upload_file", tab, info,
            arguments={"path": str(path)},
        )
        self._do(
            lambda: self.runtime.set_input_files(
                tab.handle, record["handle"], str(path),
                self._timeout(timeout_ms),
            ),
            "upload file to",
            record,
        )
        # Completion verification: the input must actually show
        # the uploaded file.  A driver that reports a different
        # file means the upload did not land — fail structurally
        # instead of claiming success.
        shown = ""
        try:
            shown = self._read_info(tab, record).value or ""
        except Exception:
            shown = ""
        if shown and path.name not in shown:
            raise BrowserException(
                f"Upload of {path.name!r} could not be verified: "
                f"the file input shows {shown!r}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"ref": info.ref, "file": path.name},
            )
        result = self._interaction_result(tab, "upload", info)
        result["uploaded_file"] = path.name
        result["upload_verified"] = bool(shown and path.name in shown)
        if sensitivity:
            result["sensitivity"] = sensitivity
        return result

    def scroll_page(
        self,
        *,
        dx: int = 0,
        dy: int = 0,
        target: dict[str, Any] | None = None,
        tab_id: str | None = None,
    ) -> dict[str, Any]:
        """Scroll the page by (dx, dy), or scroll an element into
        view when *target* is given."""
        if target:
            tab, record, info = self._prepare_interaction(tab_id, target)
            self._do(
                lambda: self.runtime.scroll_to_element(
                    tab.handle, record["handle"]
                ),
                "scroll to",
                record,
            )
            result = self._interaction_result(tab, "scroll_to", info)
            return result
        tab = self._resolve_tab(tab_id)
        self._do(
            lambda: self.runtime.scroll_page(tab.handle, int(dx), int(dy)),
            "scroll page",
            {"ref": None},
        )
        return {
            "action": "scroll",
            "dx": int(dx),
            "dy": int(dy),
            "page": self.current_page(tab.tab_id),
        }

    # -- page observation ---------------------------------------------------
    def observe(
        self,
        tab_id: str | None = None,
        *,
        max_elements: int = 25,
        text_limit: int = 2000,
    ) -> dict[str, Any]:
        """Read the current page's structured state: URL, title,
        visible text and the interactive elements (each with a
        reusable ref for later click/type actions), plus whether
        the page changed since the last observation of this tab.

        A failed observation raises — it is never reported as a
        successful but empty observation.
        """
        self.sync_tabs()
        tab = self._resolve_tab(tab_id)
        page = self.current_page(tab.tab_id)
        try:
            text = self.runtime.page_text(tab.handle)
        except BrowserException:
            raise
        except Exception as e:
            raise BrowserException(
                f"Could not read page text: {e}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"tab_id": tab.tab_id},
            ) from e
        try:
            handles = self.runtime.interactive_elements(
                tab.handle, max(1, min(int(max_elements), 100))
            )
        except BrowserException:
            raise
        except Exception as e:
            raise BrowserException(
                f"Could not list interactive elements: {e}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"tab_id": tab.tab_id},
            ) from e
        elements = [self._register_element(tab, h) for h in handles]

        # Dialogs the page raised (recorded + dismissed by the
        # driver): surfaced as observation, never acted on.
        try:
            dialogs = self.runtime.dialogs(tab.handle)
        except Exception:
            dialogs = []

        text = text or ""
        fingerprint = hashlib.sha256(
            json.dumps(
                [
                    page["url"],
                    page["title"],
                    text,
                    [
                        (
                            el.tag,
                            el.attributes.get("id"),
                            el.attributes.get("name"),
                            el.text,
                        )
                        for el in elements
                    ],
                ],
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        changed = self._observations.get(tab.tab_id) != fingerprint
        self._observations[tab.tab_id] = fingerprint
        if changed:
            self.runtime.emit(
                "page_changed", tab_id=tab.tab_id, url=page["url"]
            )

        logger.info(
            "browser observed tab %s: %d elements, changed=%s",
            tab.tab_id,
            len(elements),
            changed,
        )
        return {
            "kind": "observation",
            "tab_id": tab.tab_id,
            "url": page["url"],
            "title": page["title"],
            "text": text[:text_limit],
            "text_truncated": len(text) > text_limit,
            "elements": [el.to_dict() for el in elements],
            "element_count": len(elements),
            "dialogs": dialogs,
            "page_changed": changed,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "observation": {
                "type": "page_observation",
                "summary": (
                    f"Observed page {page['title']!r} at "
                    f"{page['url']}: {len(elements)} interactive "
                    f"elements, page changed: {changed}"
                ),
                "url": page["url"],
                "title": page["title"],
                "page_changed": changed,
            },
        }

    def accessibility_tree(
        self, *, tab_id: str | None = None
    ) -> dict[str, Any]:
        """The page's accessibility tree, normalized.

        The browser's own accessibility snapshot is preferred
        (it names elements the way users perceive them); when
        the driver cannot provide one, an equivalent tree is
        derived from the DOM's interactive elements, whose nodes
        carry live element refs.  Either way the shape is the
        same, so semantic tooling never cares which source
        answered.
        """
        tab = self._resolve_tab(tab_id)
        nodes: list[dict[str, Any]] = []
        source = "accessibility"
        try:
            raw = self.runtime.accessibility_snapshot(tab.handle)
            nodes = normalize_snapshot(raw)
        except BrowserException:
            nodes = []
        except Exception:
            nodes = []
        if not nodes:
            observed = self.observe(tab_id=tab.tab_id)
            nodes = derive_nodes(observed["elements"])
            source = "dom"
        for node in nodes:
            node["tab_id"] = tab.tab_id
        interactive = sum(
            1 for n in nodes if n["role"] not in ("generic", "text")
        )
        return {
            "tab_id": tab.tab_id,
            "source": source,
            "nodes": nodes,
            "count": len(nodes),
            "observation": {
                "type": "accessibility_tree",
                "summary": (
                    f"Accessibility tree ({source}) for tab "
                    f"{tab.tab_id}: {len(nodes)} nodes, "
                    f"{interactive} named/interactive"
                ),
                "url": self.current_page(tab.tab_id)["url"],
                "source": source,
                "count": len(nodes),
            },
        }

    def find_semantic(
        self,
        description: str,
        *,
        tab_id: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Rank page elements against a natural-language
        description (e.g. "Login button", "Email field").

        Accessibility first: candidates come from the page's
        accessibility tree / DOM-derived nodes.  Low-confidence
        matches carry no element reference at all; mid-confidence
        matches are remembered, and any later action on them is
        routed through the human approval gate (see
        ``_gate_check``) instead of executing blindly.
        """
        tree = self.accessibility_tree(tab_id=tab_id)
        ranked = rank_semantic(description, tree["nodes"])
        matches: list[dict[str, Any]] = []
        for match in ranked[: max(1, int(limit))]:
            entry = dict(match)
            if entry["tier"] == "low":
                entry["element_ref"] = None
            elif entry.get("element_ref"):
                self._semantic_confidence[entry["element_ref"]] = float(
                    entry["confidence"]
                )
            matches.append(entry)
        best = matches[0] if matches else None
        uncertain = bool(best and best["tier"] != "actionable")
        return {
            "description": description,
            "tab_id": tree["tab_id"],
            "matches": matches,
            "best_confidence": best["confidence"] if best else 0.0,
            "uncertain": uncertain,
            "note": (
                "Low-confidence matches have no element_ref and "
                "must not be acted on automatically; verify or "
                "ask first."
                if uncertain else
                "Top match is confident enough to act on via its "
                "element_ref or locator."
            ),
            "observation": {
                "type": "semantic_search",
                "summary": (
                    f"Semantic search for {description!r}: "
                    f"{len(matches)} match(es), best confidence "
                    f"{best['confidence'] if best else 0.0}"
                ),
                "description": description,
                "best_confidence": best["confidence"] if best else 0.0,
            },
        }

    def page_document(
        self, *, tab_id: str | None = None
    ) -> dict[str, Any]:
        """Raw structured page document for content extraction.

        Uses the driver's structured extraction; when that is
        unavailable, falls back to the flat page text (marked
        ``structured=False``) so extraction degrades instead of
        failing.
        """
        tab = self._resolve_tab(tab_id)
        doc: dict[str, Any] = {}
        try:
            doc = self.runtime.page_content(tab.handle) or {}
        except BrowserException:
            doc = {}
        except Exception:
            doc = {}
        page = self.current_page(tab.tab_id)
        if doc:
            return {**doc, "structured": True,
                    "url": doc.get("url") or page["url"],
                    "title": doc.get("title") or page["title"]}
        observed = self.observe(tab_id=tab.tab_id)
        return {
            "title": page["title"],
            "url": page["url"],
            "text": observed["text"],
            "headings": [],
            "paragraphs": [],
            "lists": [],
            "links": [],
            "tables": [],
            "structured": False,
        }

    def wait_for(
        self,
        condition: str,
        *,
        locator: dict[str, Any] | None = None,
        text: str | None = None,
        value: str | None = None,
        tab_id: str | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        """Wait — condition-based, never a fixed sleep — until the
        page reaches a state: an element appears/disappears, text
        is present, or the URL/title contains a value.  Times out
        with a structured ``timeout`` error."""
        kind = str(condition or "").strip()
        if kind in ("element_present", "element_hidden"):
            spec = {
                "kind": kind,
                "locator": self._validate_locator(locator),
            }
        elif kind == "text_present":
            if not text and isinstance(locator, dict):
                text = locator.get("text")
            if not text:
                raise BrowserException(
                    "text_present waits need a 'text' value",
                    code=BrowserErrorCode.INVALID_LOCATOR,
                )
            spec = {"kind": kind, "text": str(text)}
        elif kind in ("url_contains", "title_contains"):
            if not value:
                raise BrowserException(
                    f"{kind} waits need a 'value' to look for",
                    code=BrowserErrorCode.INVALID_LOCATOR,
                )
            spec = {"kind": kind, "value": str(value)}
        else:
            raise BrowserException(
                f"Unknown wait condition {condition!r}; use "
                "element_present, element_hidden, text_present, "
                "url_contains or title_contains",
                code=BrowserErrorCode.INVALID_LOCATOR,
            )
        tab = self._resolve_tab(tab_id)
        timeout = self._timeout(timeout_ms)
        try:
            self.runtime.wait_for(tab.handle, spec, timeout)
        except BrowserException:
            raise
        except Exception as e:
            raise BrowserException(
                f"Wait for {kind} failed: {e}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"condition": kind, "tab_id": tab.tab_id},
            ) from e
        return {
            "satisfied": True,
            "condition": kind,
            "page": self.current_page(tab.tab_id),
        }

    # -- SPA / dynamic-page awareness -------------------------------------

    def spa_state(self, tab_id: str | None = None) -> dict[str, Any]:
        """Probe the page's dynamic state (SPA-aware).

        Returns URL, title, ``document.readyState``, detected
        frameworks (React / Next.js / Vue / Angular), text length,
        element count and a content hash — everything needed to tell
        whether client-side rendering has settled.  Falls back to a
        probe composed from ordinary page reads when the driver has
        no probe of its own.
        """
        tab = self._resolve_tab(tab_id)
        probe: dict[str, Any] | None = None
        try:
            raw = self.runtime.page_probe(tab.handle)
            if isinstance(raw, dict) and raw:
                probe = {
                    "url": str(raw.get("url") or tab.url),
                    "title": str(raw.get("title") or tab.title),
                    "ready_state": str(
                        raw.get("ready_state") or "unknown"
                    ),
                    "frameworks": list(raw.get("frameworks") or []),
                    "text_length": int(raw.get("text_length") or 0),
                    "element_count": int(raw.get("element_count") or 0),
                    "content_hash": str(raw.get("content_hash") or ""),
                }
        except Exception:
            probe = None
        if probe is None:
            text = ""
            try:
                text = self.runtime.page_text(tab.handle) or ""
            except Exception:
                text = ""
            element_count = 0
            try:
                element_count = len(
                    self.runtime.interactive_elements(tab.handle)
                    or []
                )
            except Exception:
                element_count = 0
            self._refresh(tab, quiet=True)
            probe = {
                "url": tab.url,
                "title": tab.title,
                "ready_state": "unknown",
                "frameworks": [],
                "text_length": len(text),
                "element_count": element_count,
                "content_hash": hashlib.sha256(
                    text.encode("utf-8", "replace")
                ).hexdigest()[:12],
            }
        probe["tab_id"] = tab.tab_id
        return probe

    def wait_for_stable(
        self,
        tab_id: str | None = None,
        *,
        timeout_ms: int = 5000,
        stable_polls: int = 2,
    ) -> dict[str, Any]:
        """Wait until a dynamic page stops changing.

        Polls the SPA probe until URL, title, text length, element
        count and content hash stay identical for ``stable_polls``
        consecutive probes — covering client-side routing, async
        rendering and DOM mutations without any blind fixed sleep.
        Raises a structured ``timeout`` error when the page keeps
        changing past the deadline.
        """
        if timeout_ms is None or timeout_ms <= 0:
            raise BrowserException(
                "timeout_ms must be a positive number of "
                "milliseconds.",
                code=BrowserErrorCode.TIMEOUT,
            )
        tab = self._resolve_tab(tab_id)
        deadline = time.monotonic() + timeout_ms / 1000.0
        last_signature: tuple | None = None
        stable_count = 0
        probes = 0
        probe: dict[str, Any] = {}
        while True:
            probe = self.spa_state(tab.tab_id)
            probes += 1
            signature = (
                probe["url"],
                probe["title"],
                probe["text_length"],
                probe["element_count"],
                probe["content_hash"],
            )
            if signature == last_signature:
                stable_count += 1
            else:
                stable_count = 0
                last_signature = signature
            if stable_count >= stable_polls:
                return {
                    **probe,
                    "stable": True,
                    "probes": probes,
                }
            if time.monotonic() >= deadline:
                raise BrowserException(
                    "Page did not become stable (SPA/dynamic "
                    f"content) — timed out after {timeout_ms}ms "
                    "waiting for a stable state.",
                    code=BrowserErrorCode.TIMEOUT,
                    details={
                        "tab_id": tab.tab_id,
                        "last_state": probe,
                        "probes": probes,
                    },
                )
            time.sleep(0.05)

    def screenshot(
        self,
        tab_id: str | None = None,
        *,
        output_dir: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        """Capture a PNG screenshot of the current page to disk
        and return a structured observation of it.  An empty or
        failed capture is an error, never a fake success."""
        tab = self._resolve_tab(tab_id)
        try:
            data = self.runtime.screenshot(tab.handle)
        except BrowserException:
            raise
        except Exception as e:
            raise BrowserException(
                f"Screenshot failed: {e}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"tab_id": tab.tab_id},
            ) from e
        if not data:
            raise BrowserException(
                "Screenshot failed: the browser returned no image data",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"tab_id": tab.tab_id},
            )
        directory = Path(output_dir) if output_dir else Path("screenshots")
        directory.mkdir(parents=True, exist_ok=True)
        if filename:
            name = Path(filename).name
            if not name.lower().endswith(".png"):
                name += ".png"
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            name = f"screenshot_{stamp}.png"
        path = directory / name
        path.write_bytes(data)
        page = self.current_page(tab.tab_id)
        logger.info("browser screenshot saved: %s", path)
        return {
            "kind": "observation",
            "path": str(path),
            "size_bytes": len(data),
            "tab_id": tab.tab_id,
            "url": page["url"],
            "title": page["title"],
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "observation": {
                "type": "screenshot",
                "summary": (
                    f"Screenshot of {page['title']!r} at "
                    f"{page['url']} saved to {path} "
                    f"({len(data)} bytes)"
                ),
                "path": str(path),
                "url": page["url"],
            },
        }

    # -- element internals ------------------------------------------------
    @staticmethod
    def _validate_locator(
        locator: dict[str, Any] | None,
    ) -> dict[str, Any]:
        locator = dict(locator or {})
        strategies = {
            k: locator[k]
            for k in ("selector", "test_id", "label", "placeholder", "role", "text")
            if locator.get(k)
        }
        if locator.get("role") and locator.get("name"):
            strategies["name"] = locator["name"]
        if locator.get("frame"):
            # look inside an iframe (CSS also pierces open shadow
            # DOM, so shadow content needs no extra key)
            strategies["frame"] = locator["frame"]
        if not strategies:
            raise BrowserException(
                "A locator needs at least one of: selector, "
                "test_id, label, placeholder, role or text",
                code=BrowserErrorCode.INVALID_LOCATOR,
                details={"locator": locator},
            )
        return strategies

    @staticmethod
    def _split_target(
        target: dict[str, Any] | None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        """(ref, locator) from a target dict; either may be None."""
        if not target:
            return None, None
        ref = target.get("ref")
        if ref:
            return str(ref), None
        if isinstance(target.get("locator"), dict):
            return None, dict(target["locator"])
        # a bare locator dict is accepted too
        if any(k in target for k in _LOCATOR_KEYS):
            return None, dict(target)
        return None, None

    def _query(self, tab: _Tab, locator: dict[str, Any], limit: int):
        try:
            return self.runtime.query_elements(
                tab.handle, locator, max(1, min(int(limit), 50))
            )
        except BrowserException:
            raise
        except Exception as e:
            raise BrowserException(
                f"Element lookup failed: {e}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"locator": locator, "tab_id": tab.tab_id},
            ) from e

    def _register_element(self, tab: _Tab, handle) -> ElementInfo:
        self._element_counter += 1
        ref = f"el_{self._element_counter}"
        record = {
            "ref": ref,
            "tab_id": tab.tab_id,
            "handle": handle,
            "generation": self._generations.get(tab.tab_id, 0),
        }
        self._elements[ref] = record
        return self._read_info(tab, record)

    def _read_info(self, tab: _Tab, record: dict[str, Any]) -> ElementInfo:
        try:
            data = self.runtime.element_info(tab.handle, record["handle"])
        except BrowserException:
            raise
        except Exception as e:
            raise BrowserException(
                f"Could not read element {record['ref']}: {e}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"ref": record["ref"]},
            ) from e
        attributes = dict(data.get("attributes", {}))
        value = str(data.get("value", ""))
        if str(attributes.get("type", "")).lower() == "password":
            # A password field's content is a secret: never carry
            # it into element records, observations or AgentState.
            value = MASK
            if "value" in attributes:
                attributes["value"] = MASK
        return ElementInfo(
            ref=record["ref"],
            tab_id=tab.tab_id,
            tag=str(data.get("tag", "")),
            text=str(data.get("text", "")),
            attributes=attributes,
            visible=bool(data.get("visible", True)),
            enabled=bool(data.get("enabled", True)),
            editable=bool(data.get("editable", False)),
            value=value,
        )

    def _resolve_element(self, tab: _Tab, target):
        """Validate a target and return (tab, element record).

        Refs from an older page generation are rejected as stale;
        locators are resolved fresh against the live page.
        """
        ref, locator = self._split_target(target)
        if ref is not None:
            record = self._elements.get(ref)
            if record is None or record["tab_id"] != tab.tab_id:
                raise BrowserException(
                    f"Unknown element reference {ref!r} for tab "
                    f"{tab.tab_id!r}; find the element again",
                    code=BrowserErrorCode.INVALID_ELEMENT,
                    details={"ref": ref, "tab_id": tab.tab_id},
                )
            if record["generation"] != self._generations.get(tab.tab_id, 0):
                raise BrowserException(
                    f"Element {ref} belongs to an older version of "
                    "the page (the page changed since it was "
                    "found); find it again before interacting",
                    code=BrowserErrorCode.STALE_ELEMENT,
                    details={"ref": ref, "tab_id": tab.tab_id},
                )
            return tab, record
        if locator is not None:
            locator = self._validate_locator(locator)
            handles = self._query(tab, locator, 1)
            if not handles:
                raise BrowserException(
                    f"No element matches locator {locator} on tab "
                    f"{tab.tab_id!r}",
                    code=BrowserErrorCode.ELEMENT_NOT_FOUND,
                    details={"locator": locator, "tab_id": tab.tab_id},
                )
            record = {
                "ref": f"el_{self._element_counter + 1}",
                "tab_id": tab.tab_id,
                "handle": handles[0],
                "generation": self._generations.get(tab.tab_id, 0),
            }
            self._element_counter += 1
            self._elements[record["ref"]] = record
            return tab, record
        raise BrowserException(
            "An interaction target is required: a ref from "
            "browser_find_elements or a locator (selector / role / "
            "text / label / placeholder / test_id)",
            code=BrowserErrorCode.INVALID_LOCATOR,
        )

    def _prepare_interaction(self, tab_id, target):
        tab = self._resolve_tab(tab_id)
        tab, record = self._resolve_element(tab, target)
        info = self._read_info(tab, record)
        return tab, record, info

    @staticmethod
    def _require_usable(info: ElementInfo, action: str) -> None:
        if not info.visible:
            raise BrowserException(
                f"Element {info.ref} (<{info.tag}>) is not "
                f"visible; refusing to {action} it",
                code=BrowserErrorCode.INVALID_ELEMENT,
                details={"ref": info.ref, "tag": info.tag},
            )
        if not info.enabled:
            raise BrowserException(
                f"Element {info.ref} (<{info.tag}>) is disabled; "
                f"refusing to {action} it",
                code=BrowserErrorCode.INVALID_ELEMENT,
                details={"ref": info.ref, "tag": info.tag},
            )

    def _challenge_check(self, tab: _Tab) -> None:
        """Stop with ``human_required`` on human-check pages.

        Detection only — the challenge is never solved, bypassed or
        worked around.  When a challenge handler is registered it
        is asked for approval; on approval the controller waits
        for the human to solve the check in the browser and then
        lets the action resume automatically.  Otherwise the task
        pauses for a human (``human_required``).  Detection
        problems never block ordinary actions.
        """
        if not self._challenge_guard:
            return
        try:
            detection = detect_challenge(self, tab.tab_id)
        except Exception:
            return
        if not detection.get("detected"):
            return
        handler = self.challenge_handler
        if handler is not None:
            try:
                approved = bool(handler(dict(detection)))
            except Exception:
                approved = False
            if approved:
                cleared, _last = self._wait_clear(
                    tab, self._challenge_wait_ms
                )
                if cleared:
                    logger.info(
                        "browser challenge cleared by human on "
                        "tab %s; resuming",
                        tab.tab_id,
                    )
                    return
        raise BrowserException(
            "A human check (CAPTCHA / verification) is present "
            "on this page; human intervention is required "
            "before continuing (human_required).",
            code=BrowserErrorCode.HUMAN_REQUIRED,
            details={
                "tab_id": tab.tab_id,
                "challenge": detection,
            },
        )

    def _wait_clear(
        self, tab: _Tab, timeout_ms: int
    ) -> tuple[bool, dict[str, Any]]:
        """Poll until no challenge is detected (condition-based).

        Returns (cleared, last_detection); never raises.
        """
        deadline = time.monotonic() + max(0, timeout_ms) / 1000.0
        last: dict[str, Any] = {"detected": True}
        while True:
            try:
                last = detect_challenge(self, tab.tab_id)
            except Exception:
                return True, {"detected": False}
            if not last.get("detected"):
                return True, last
            if time.monotonic() >= deadline:
                return False, last
            time.sleep(0.25)

    def wait_for_challenge_clear(
        self, tab_id: str | None = None, *, timeout_ms: int = 120_000
    ) -> dict[str, Any]:
        """Wait for a human to solve the page's human check.

        The human solves the CAPTCHA/verification in the browser
        themselves; this only watches (condition-based polling)
        until the challenge disappears.  Raises a structured
        ``human_required`` error when the wait times out with the
        challenge still present.
        """
        if timeout_ms is None or timeout_ms <= 0:
            raise BrowserException(
                "timeout_ms must be a positive number of "
                "milliseconds.",
                code=BrowserErrorCode.TIMEOUT,
            )
        tab = self._resolve_tab(tab_id)
        cleared, last = self._wait_clear(tab, timeout_ms)
        if not cleared:
            raise BrowserException(
                "The human check is still present after waiting "
                f"{timeout_ms}ms; human intervention is still "
                "required (human_required).",
                code=BrowserErrorCode.HUMAN_REQUIRED,
                details={"tab_id": tab.tab_id, "challenge": last},
            )
        return {
            "cleared": True,
            "human_required": False,
            "tab_id": tab.tab_id,
            "page": self.current_page(tab.tab_id),
        }

    def set_challenge_handler(
        self, handler: Any, *, wait_ms: int | None = None
    ) -> None:
        """Register (or clear, with None) the challenge approver.

        The handler receives the detection dict and returns True
        when the human will solve the check in the browser; the
        blocked action then waits (up to ``wait_ms``) and resumes
        automatically once the challenge clears.
        """
        self.challenge_handler = handler
        if wait_ms is not None:
            self._challenge_wait_ms = int(wait_ms)

    # -- network / rate-limit awareness --------------------------------

    def network_status(
        self, tab_id: str | None = None
    ) -> dict[str, Any]:
        """Diagnostic summary of the page's network activity.

        Failed requests, timeouts, blocked resources and HTTP
        errors the page hit — observation only; nothing here can
        fire a network request of its own.
        """
        tab = self._resolve_tab(tab_id)
        try:
            events = self.runtime.network_events(tab.handle)
        except Exception:
            events = []
        summary = summarize_network(events or [], page_url=tab.url)
        summary["tab_id"] = tab.tab_id
        return summary

    def capabilities(self) -> dict[str, Any]:
        """Afnan-level browser capabilities (via the runtime).

        The agent sees what the *Afnan* browser stack can do —
        never raw engine features.  Runtime/engine capabilities
        are merged with the controller's own (semantic location,
        structured observation).
        """
        caps = dict(self.runtime.capabilities())
        caps.update({
            "semantic_locator": True,
            "accessibility": True,
            "page_observation": True,
            "multi_tab": True,
            "visual_interaction": False,
        })
        return {
            "adapter": self.runtime.name,
            "session_id": self.runtime.session.session_id,
            "profile_id": self.runtime.session.profile_id,
            "capabilities": caps,
            "observation": {
                "type": "browser_capabilities",
                "summary": (
                    "Browser capabilities via AfnanBrowserRuntime "
                    f"({self.runtime.name}): "
                    + ", ".join(
                        k for k, v in sorted(caps.items()) if v
                    )
                ),
            },
        }

    def events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Recent browser runtime events (started, tabs,
        navigation, downloads, errors...)."""
        return self.runtime.events(limit)

    def recover_browser(self) -> dict[str, Any]:
        """Restart a crashed browser engine via the runtime and
        report which tabs are recoverable (never auto-reopens
        them; the task decides)."""
        result = self.runtime.recover()
        if result.success:
            self._tabs.clear()
            self._active_tab_id = None
            self._elements.clear()
        return result.to_dict()

    def rate_limit_status(
        self, tab_id: str | None = None
    ) -> dict[str, Any]:
        """Is this page/host rate-limiting or blocking us?

        Combines page-signal detection (title/text/URL), network
        diagnostics (HTTP 429) and the controller's own action
        pacing.  Detection only — the answer never triggers a
        retry.
        """
        tab = self._resolve_tab(tab_id)
        detection: dict[str, Any] = {"detected": False}
        try:
            detection = RateLimitDetector().detect(
                self.observe(tab.tab_id),
                self.network_status(tab.tab_id),
            )
        except Exception:
            detection = {"detected": False}
        from urllib.parse import urlparse

        host = urlparse(tab.url).netloc
        pacing = self._pacer.status(host)
        retry_after = detection.get("retry_after_s")
        if retry_after is None and not pacing.get("allowed", True):
            retry_after = int(pacing.get("retry_after_s") or 0)
        return {
            "limited": bool(
                detection.get("detected")
                or not pacing.get("allowed", True)
            ),
            "detection": detection,
            "pacing": pacing,
            "retry_after_s": retry_after,
            "tab_id": tab.tab_id,
        }

    def rate_limit_backoff(
        self, tab_id: str | None = None, *, max_wait_ms: int = 5000
    ) -> dict[str, Any]:
        """One controlled backoff: wait once, then re-check.

        Never loops and never retries the failed action itself —
        it waits at most ``max_wait_ms`` and reports the fresh
        status so the caller can decide (continue, pause or ask
        a human).
        """
        tab = self._resolve_tab(tab_id)
        status = self.rate_limit_status(tab.tab_id)
        if not status["limited"]:
            return {**status, "waited_ms": 0}
        wait_ms = min(
            int((status.get("retry_after_s") or 1) * 1000),
            max(0, int(max_wait_ms)),
        )
        if wait_ms > 0:
            time.sleep(wait_ms / 1000.0)
        after = self.rate_limit_status(tab.tab_id)
        return {**after, "waited_ms": wait_ms}

    def _ratelimit_check(self, tab: _Tab) -> None:
        """Stop with ``rate_limited`` when the site is throttling
        or blocking, or when our own action pacing limit for this
        host is exhausted.  Never retries anything itself."""
        from urllib.parse import urlparse

        host = urlparse(tab.url).netloc
        if self._rate_policy.max_actions_per_minute is not None:
            allowed, retry_after = self._pacer.check(host)
            if not allowed:
                raise BrowserException(
                    "Action rate limit reached for "
                    f"{host or 'this page'}: too many actions in "
                    "a short window. Pause with a controlled "
                    "backoff instead of retrying (rate_limited).",
                    code=BrowserErrorCode.RATE_LIMITED,
                    details={
                        "tab_id": tab.tab_id,
                        "host": host,
                        "retry_after_s": round(retry_after, 2),
                    },
                )
            self._pacer.note(host)
        try:
            detection = RateLimitDetector().detect(
                self.observe(tab.tab_id),
                self.network_status(tab.tab_id),
            )
        except Exception:
            return
        if detection.get("detected"):
            raise BrowserException(
                "The site is rate-limiting or blocking automated "
                "access on this page. Do not retry aggressively: "
                "pause with a controlled backoff or hand the task "
                "to a human (rate_limited).",
                code=BrowserErrorCode.RATE_LIMITED,
                details={
                    "tab_id": tab.tab_id,
                    "detection": detection,
                },
            )

    # -- profiles (isolated contexts) -----------------------------------

    @property
    def current_profile(self) -> str:
        return self._active_profile

    def profile_info(self, name: str) -> dict[str, Any]:
        if name not in self._profiles:
            raise BrowserException(
                f"Unknown browser profile {name!r}.",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"profile": name},
            )
        return self._profile_info(name)

    @staticmethod
    def _sanitize_preferences(
        preferences: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Profile preferences, minus anything credential-shaped.

        Profiles carry display/context preferences only; secrets
        never belong in a profile record.
        """
        clean: dict[str, Any] = {}
        for key, value in (preferences or {}).items():
            lowered = str(key).lower()
            if any(
                marker in lowered
                for marker in (
                    "password", "passwd", "token", "secret",
                    "cookie", "auth", "credential", "api_key",
                )
            ):
                continue
            clean[str(key)] = value
        return clean

    def create_profile(
        self,
        name: str,
        preferences: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an isolated browser profile (own context:
        separate cookies, storage and tabs)."""
        name = str(name or "").strip()
        if not name:
            raise BrowserException(
                "A profile name is required.",
                code=BrowserErrorCode.OPERATION_FAILED,
            )
        if name in self._profiles:
            raise BrowserException(
                f"Browser profile {name!r} already exists.",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"profile": name},
            )
        clean = self._sanitize_preferences(preferences)
        # driver context first: if the backend cannot isolate
        # profiles, nothing is registered here either
        self.runtime.create_profile_context(name, clean)
        self._profiles[name] = {
            "name": name,
            "preferences": clean,
        }
        logger.info("browser profile created: %s", name)
        return self._profile_info(name)

    def select_profile(self, name: str) -> dict[str, Any]:
        """Switch the active profile, stashing the current
        profile's tabs so nothing leaks across profiles."""
        name = str(name or "").strip()
        if name not in self._profiles:
            raise BrowserException(
                f"Unknown browser profile {name!r}.",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"profile": name},
            )
        if name == self._active_profile:
            return self._profile_info(name)
        # backend first: a failed switch leaves state untouched
        self.runtime.set_active_profile(name)
        self._profile_tabs[self._active_profile] = (
            self._tabs,
            self._active_tab_id,
        )
        stashed = self._profile_tabs.pop(name, None)
        if stashed is not None:
            self._tabs, self._active_tab_id = stashed
        else:
            self._tabs, self._active_tab_id = {}, None
        self._active_profile = name
        logger.info("browser profile selected: %s", name)
        return self._profile_info(name)

    def list_profiles(self) -> list[dict[str, Any]]:
        return [
            self._profile_info(name) for name in self._profiles
        ]

    def _profile_info(self, name: str) -> dict[str, Any]:
        if name == self._active_profile:
            tab_count = len(self._tabs)
        else:
            stashed = self._profile_tabs.get(name)
            tab_count = len(stashed[0]) if stashed else 0
        record = self._profiles[name]
        return {
            "name": name,
            "active": name == self._active_profile,
            "tabs": tab_count,
            "preferences": dict(record.get("preferences") or {}),
        }

    def _gate_check(
        self,
        tool_name: str,
        tab: _Tab,
        info: ElementInfo | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Classify an interaction and run the approval gate.

        Returns the sensitivity record for sensitive-but-allowed
        actions (None for ordinary ones).  Raises a structured
        BrowserException — approval_required / approval_denied —
        *before* anything executes when a sensitive action may
        not run.
        """
        risk = classify_action(
            tool_name,
            element=info.to_dict() if info is not None else None,
            arguments=arguments,
        )
        if not risk.sensitive and info is not None:
            confidence = self._semantic_confidence.get(info.ref)
            if confidence is not None and confidence < 0.8:
                # The target came from a natural-language search
                # and the match is not confident: a human must
                # confirm before anything irreversible happens
                # to the wrong element.
                risk = ActionRisk(
                    sensitivity=Sensitivity.SENSITIVE,
                    category="uncertain_target",
                    reason=(
                        "Element was located from a "
                        "natural-language description with "
                        f"confidence {confidence:.2f} (< 0.80); "
                        "confirm it is the right target before "
                        "acting"
                    ),
                )
        if not risk.sensitive:
            return None
        page = self.current_page(tab.tab_id)
        decision = self.approval_gate.check(
            risk,
            tool_name=tool_name,
            arguments=arguments,
            url=page["url"],
            title=page["title"],
        )
        if not decision.allowed:
            code = (
                BrowserErrorCode.APPROVAL_DENIED
                if "Denied by human" in decision.detail
                else BrowserErrorCode.APPROVAL_REQUIRED
            )
            raise BrowserException(
                f"{decision.detail}: {tool_name} "
                f"({risk.category}) was not executed",
                code=code,
                details={
                    "category": risk.category,
                    "reason": risk.reason,
                    "tool": tool_name,
                },
            )
        return decision.to_dict()

    def _do(self, operation, action: str, record) -> None:
        try:
            operation()
        except BrowserException:
            raise
        except Exception as e:
            raise BrowserException(
                f"Browser failed to {action}: {e}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"ref": record.get("ref") if record else None},
            ) from e

    def _interaction_result(
        self, tab: _Tab, action: str, info: ElementInfo
    ) -> dict[str, Any]:
        logger.info(
            "browser %s on %s (tab %s)", action, info.ref, tab.tab_id
        )
        # Refresh the element info after the action, but never let
        # that fail the action itself: a click that navigates the
        # page detaches the clicked element, so fall back to the
        # validated pre-action snapshot.
        element = info.to_dict()
        record = self._elements.get(info.ref)
        if record is not None:
            try:
                element = self._read_info(tab, record).to_dict()
            except BrowserException:
                pass
        return {
            "action": action,
            "element": element,
            "page": self.current_page(tab.tab_id),
        }

    def _timeout(self, timeout_ms: int | None) -> int:
        if timeout_ms is None:
            return self.DEFAULT_TIMEOUT_MS
        return max(1, int(timeout_ms))

    # -- internals -------------------------------------------------------------
    def _session_info(self, *, launched: bool) -> dict[str, Any]:
        return {
            "running": self._running,
            "launched": launched,
            "browser": self._browser_name,
            "backend": self.runtime.name,
            "endpoint": self._endpoint,
            "tabs": len(self._tabs),
            "active_tab": self._active_tab_id,
        }

    def _require_running(self) -> None:
        if not self._running:
            raise BrowserException(
                "Browser is not running; launch it first "
                "(browser_launch)",
                code=BrowserErrorCode.BROWSER_NOT_STARTED,
            )

    def _resolve_tab(
        self, tab_id: str | None, *, allow_default: bool = True
    ) -> _Tab:
        self._require_running()
        if tab_id is None:
            if not allow_default or self._active_tab_id is None:
                raise BrowserException(
                    "No tab is open; create one first (browser_new_tab)",
                    code=BrowserErrorCode.INVALID_TAB,
                    details={"tabs": list(self._tabs)},
                )
            return self._tabs[self._active_tab_id]
        tab = self._tabs.get(tab_id)
        if tab is None:
            raise BrowserException(
                f"Unknown tab {tab_id!r}. Open tabs: "
                + (", ".join(self._tabs) or "(none)"),
                code=BrowserErrorCode.INVALID_TAB,
                details={
                    "tab_id": tab_id,
                    "tabs": list(self._tabs),
                },
            )
        return tab

    def _goto(self, tab: _Tab, url: str) -> None:
        url = self._validate_url(url)
        try:
            self.runtime.goto(tab.handle, url)
        except BrowserException:
            raise
        except Exception as e:
            raise BrowserException(
                f"Navigation to {url!r} failed: {e}",
                code=BrowserErrorCode.NAVIGATION_FAILED,
                details={"url": url, "tab_id": tab.tab_id},
            ) from e
        # the page changed: element refs from before are now stale
        self._generations[tab.tab_id] = (
            self._generations.get(tab.tab_id, 0) + 1
        )
        self._refresh(tab)
        self._record_history("navigate", tab)

    @staticmethod
    def _validate_url(url: str) -> str:
        url = (url or "").strip()
        if not url:
            raise BrowserException(
                "A non-empty URL is required",
                code=BrowserErrorCode.INVALID_URL,
            )
        if "://" not in url:
            url = f"https://{url}"
        return url

    def _history_move(
        self, kind: str, tab_id: str | None
    ) -> dict[str, Any]:
        tab = self._resolve_tab(tab_id)
        method = self.runtime.go_back if kind == "back" else self.runtime.go_forward
        try:
            method(tab.handle)
        except Exception as e:
            raise BrowserException(
                f"Could not go {kind} in tab {tab.tab_id!r}: {e}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"tab_id": tab.tab_id},
            ) from e
        self._generations[tab.tab_id] = (
            self._generations.get(tab.tab_id, 0) + 1
        )
        self._refresh(tab, quiet=True)
        self._record_history(kind, tab)
        return self.current_page(tab.tab_id)

    def _refresh(self, tab: _Tab, *, quiet: bool = False) -> None:
        try:
            tab.url = self.runtime.page_url(tab.handle) or tab.url
            tab.title = self.runtime.page_title(tab.handle) or tab.title
        except Exception as e:
            if quiet:
                return
            raise BrowserException(
                f"Could not read page state for tab {tab.tab_id!r}: {e}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"tab_id": tab.tab_id},
            ) from e

    def _tab_info(self, tab: _Tab) -> TabInfo:
        return TabInfo(
            tab_id=tab.tab_id,
            url=tab.url,
            title=tab.title,
            active=tab.tab_id == self._active_tab_id,
            purpose=self._tab_purposes.get(tab.tab_id, ""),
        )

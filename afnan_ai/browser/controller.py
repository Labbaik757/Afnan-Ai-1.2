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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from afnan_ai.browser.accessibility import derive_nodes, normalize_snapshot
from afnan_ai.browser.backend import (
    SUPPORTED_BROWSERS,
    BrowserBackend,
    PlaywrightBackend,
)
from afnan_ai.browser.base import (
    BrowserErrorCode,
    BrowserException,
    ElementInfo,
    PageState,
    TabInfo,
)
from afnan_ai.browser.security import (
    ApprovalGate,
    classify_action,
)
from afnan_ai.log_config import get_logger
from afnan_ai.redaction import MASK

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
    ):
        self._backend = backend or PlaywrightBackend()
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
        # task-level purpose per tab (multi-tab task management)
        self._tab_purposes: dict[str, str] = {}

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
        return self._backend.name

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
            self._backend.start(
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
            self._backend.connect(endpoint)
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
            self._backend.stop()
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
            handle = self._backend.new_page()
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
            handles = self._backend.list_pages()
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
            logger.info("browser adopted popup tab: %s", tab.tab_id)
        return adopted

    def select_tab(self, tab_id: str) -> dict[str, Any]:
        """Make *tab_id* the active tab."""
        tab = self._resolve_tab(tab_id, allow_default=False)
        self._active_tab_id = tab.tab_id
        self._refresh(tab)
        return self._tab_info(tab).to_dict()

    def close_tab(self, tab_id: str | None = None) -> dict[str, Any]:
        """Close a tab (the active one by default)."""
        tab = self._resolve_tab(tab_id)
        try:
            self._backend.close_page(tab.handle)
        except Exception as e:
            raise BrowserException(
                f"Could not close tab {tab.tab_id!r}: {e}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"tab_id": tab.tab_id},
            ) from e
        del self._tabs[tab.tab_id]
        self._generations.pop(tab.tab_id, None)
        self._tab_purposes.pop(tab.tab_id, None)
        self._elements = {
            ref: rec
            for ref, rec in self._elements.items()
            if rec["tab_id"] != tab.tab_id
        }
        if self._active_tab_id == tab.tab_id:
            self._active_tab_id = (
                next(reversed(self._tabs)) if self._tabs else None
            )
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
            self._backend.reload(tab.handle)
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
        sensitivity = self._gate_check("browser_click", tab, info)
        self._do(
            lambda: self._backend.click_element(
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
                lambda: self._backend.clear_element(
                    tab.handle, record["handle"], self._timeout(timeout_ms)
                ),
                "clear",
                record,
            )
        self._do(
            lambda: self._backend.fill_element(
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
        if not info.editable:
            raise BrowserException(
                f"Element {info.ref} (<{info.tag}>) is not an "
                "editable field; cannot clear it",
                code=BrowserErrorCode.INVALID_ELEMENT,
                details={"ref": info.ref, "tag": info.tag},
            )
        self._gate_check("browser_clear", tab, info)
        self._do(
            lambda: self._backend.clear_element(
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
        if info.tag != "select":
            raise BrowserException(
                f"Element {info.ref} is a <{info.tag}>, not a "
                "<select>; cannot select an option in it",
                code=BrowserErrorCode.INVALID_ELEMENT,
                details={"ref": info.ref, "tag": info.tag},
            )
        self._gate_check("browser_select_option", tab, info)
        self._do(
            lambda: self._backend.select_option(
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
            lambda: self._backend.press_key(
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
            lambda: self._backend.set_input_files(
                tab.handle, record["handle"], str(path),
                self._timeout(timeout_ms),
            ),
            "upload file to",
            record,
        )
        result = self._interaction_result(tab, "upload", info)
        result["uploaded_file"] = path.name
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
                lambda: self._backend.scroll_to_element(
                    tab.handle, record["handle"]
                ),
                "scroll to",
                record,
            )
            result = self._interaction_result(tab, "scroll_to", info)
            return result
        tab = self._resolve_tab(tab_id)
        self._do(
            lambda: self._backend.scroll_page(tab.handle, int(dx), int(dy)),
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
            text = self._backend.page_text(tab.handle)
        except BrowserException:
            raise
        except Exception as e:
            raise BrowserException(
                f"Could not read page text: {e}",
                code=BrowserErrorCode.OPERATION_FAILED,
                details={"tab_id": tab.tab_id},
            ) from e
        try:
            handles = self._backend.interactive_elements(
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
            raw = self._backend.accessibility_snapshot(tab.handle)
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
            doc = self._backend.page_content(tab.handle) or {}
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
            self._backend.wait_for(tab.handle, spec, timeout)
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
            data = self._backend.screenshot(tab.handle)
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
            return self._backend.query_elements(
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
            data = self._backend.element_info(tab.handle, record["handle"])
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
            "backend": self._backend.name,
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
            self._backend.goto(tab.handle, url)
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
        method = self._backend.go_back if kind == "back" else self._backend.go_forward
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
        return self.current_page(tab.tab_id)

    def _refresh(self, tab: _Tab, *, quiet: bool = False) -> None:
        try:
            tab.url = self._backend.page_url(tab.handle) or tab.url
            tab.title = self._backend.page_title(tab.handle) or tab.title
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

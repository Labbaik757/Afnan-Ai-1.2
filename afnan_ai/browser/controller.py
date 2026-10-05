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

from dataclasses import dataclass
from typing import Any

from afnan_ai.browser.backend import (
    SUPPORTED_BROWSERS,
    BrowserBackend,
    PlaywrightBackend,
)
from afnan_ai.browser.base import (
    BrowserErrorCode,
    BrowserException,
    PageState,
    TabInfo,
)
from afnan_ai.log_config import get_logger

logger = get_logger(__name__)


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

    def __init__(
        self,
        backend: BrowserBackend | None = None,
        *,
        browser: str = "chromium",
        headless: bool = False,
    ):
        self._backend = backend or PlaywrightBackend()
        self._default_browser = browser
        self._default_headless = headless
        self._running = False
        self._browser_name: str | None = None
        self._endpoint: str | None = None
        self._tabs: dict[str, _Tab] = {}
        self._active_tab_id: str | None = None
        self._counter = 0

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
    def new_tab(self, url: str | None = None) -> dict[str, Any]:
        """Create a tab (optionally navigating it) and select it."""
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
        self._active_tab_id = tab.tab_id
        if url:
            self._goto(tab, url)
        self._refresh(tab)
        logger.info("browser tab created: %s", tab.tab_id)
        return self._tab_info(tab).to_dict()

    def list_tabs(self) -> list[dict[str, Any]]:
        """Snapshot of all open tabs (empty when none are open)."""
        for tab in self._tabs.values():
            self._refresh(tab, quiet=True)
        return [self._tab_info(t).to_dict() for t in self._tabs.values()]

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
        )

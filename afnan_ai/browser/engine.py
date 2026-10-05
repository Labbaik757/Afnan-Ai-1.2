"""Browser Engine Adapter: the engine boundary of Afnan AI.

An *engine adapter* is the only layer allowed to talk to a
concrete browser engine (today: Chromium through Playwright).
The :class:`AfnanBrowserRuntime` owns browser lifecycle and
state and only ever calls this interface, so the engine can be
replaced — Playwright today, a future Afnan Chromium adapter —
without touching the runtime, controller, tool or agent code.

No Playwright (or any engine) types appear in this interface:
handles are opaque objects owned by the adapter, and failures
are structured :class:`BrowserException`s.

This module used to live inside ``backend.py`` as
``BrowserBackend``; that name remains as a compatibility alias.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from afnan_ai.browser.base import BrowserErrorCode, BrowserException


class BrowserEngineAdapter(ABC):
    """Interface a concrete browser engine adapter implements.

    Handles are opaque backend objects (a Playwright Page, a fake
    test page, ...).  Implementations raise BrowserException for
    expected failures; the controller also wraps unexpected
    exceptions into structured errors, so a backend never leaks a
    bare driver exception through the controller.
    """

    name: str = "backend"

    @abstractmethod
    def start(self, browser: str, headless: bool) -> None:
        """Launch a browser process (or raise BrowserException)."""

    @abstractmethod
    def connect(self, endpoint: str) -> None:
        """Connect to an already-running browser (CDP endpoint)."""

    @abstractmethod
    def new_page(self) -> Any:
        """Open a new tab/page and return its handle."""

    @abstractmethod
    def close_page(self, handle: Any) -> None:
        """Close one tab/page."""

    @abstractmethod
    def goto(self, handle: Any, url: str) -> None:
        """Navigate a tab to *url*."""

    @abstractmethod
    def go_back(self, handle: Any) -> None:
        """Go back in a tab's history."""

    @abstractmethod
    def go_forward(self, handle: Any) -> None:
        """Go forward in a tab's history."""

    @abstractmethod
    def reload(self, handle: Any) -> None:
        """Reload a tab."""

    @abstractmethod
    def page_url(self, handle: Any) -> str:
        """Return the tab's current URL."""

    @abstractmethod
    def page_title(self, handle: Any) -> str:
        """Return the tab's current page title."""

    @abstractmethod
    def stop(self) -> None:
        """Close the browser and release the driver."""

    # -- element interaction -------------------------------------------------
    # These have safe defaults (a structured "not supported" error)
    # so simple backends/fakes stay valid; real backends override
    # them.  ``locator`` is a dict of lookup strategies, e.g.
    # {"selector": "#login"} or {"role": "button", "name": "Sign in"}
    # or {"text": "Sign in"} — most specific first.
    def _unsupported(self, operation: str):
        raise BrowserException(
            f"Browser engine adapter {self.name!r} does not "
            f"support {operation}",
            code=BrowserErrorCode.UNSUPPORTED_OPERATION,
        )

    def query_elements(
        self, handle: Any, locator: dict[str, Any], limit: int
    ) -> list[Any]:
        self._unsupported("element lookup")

    def element_info(self, handle: Any, element: Any) -> dict[str, Any]:
        self._unsupported("element inspection")

    def click_element(
        self, handle: Any, element: Any, timeout_ms: int
    ) -> None:
        self._unsupported("clicking")

    def fill_element(
        self, handle: Any, element: Any, text: str, timeout_ms: int
    ) -> None:
        self._unsupported("typing")

    def clear_element(
        self, handle: Any, element: Any, timeout_ms: int
    ) -> None:
        self._unsupported("clearing")

    def set_input_files(
        self, handle: Any, element: Any, path: str, timeout_ms: int
    ) -> None:
        self._unsupported("file upload")

    def select_option(
        self, handle: Any, element: Any, value: str, timeout_ms: int
    ) -> None:
        self._unsupported("selecting an option")

    def press_key(
        self,
        handle: Any,
        element: Any | None,
        key: str,
        timeout_ms: int,
    ) -> None:
        self._unsupported("keyboard actions")

    def scroll_page(self, handle: Any, dx: int, dy: int) -> None:
        self._unsupported("scrolling")

    def scroll_to_element(self, handle: Any, element: Any) -> None:
        self._unsupported("scrolling to an element")

    # -- page observation --------------------------------------------------
    def page_text(self, handle: Any) -> str:
        """Return the page's visible text."""
        self._unsupported("reading page text")

    def accessibility_snapshot(self, handle: Any) -> Any:
        """The browser's accessibility tree (driver-native).

        Backends without one raise via ``_unsupported`` and the
        controller derives a tree from the DOM instead.
        """
        self._unsupported("accessibility snapshot")

    def page_content(self, handle: Any) -> dict[str, Any]:
        """Structured page document (headings, paragraphs, lists,
        links, tables) for content extraction."""
        self._unsupported("page content extraction")

    def page_probe(self, handle: Any) -> dict[str, Any]:
        """Dynamic-state probe (SPA awareness): readiness, detected
        frameworks, text/element counts and a content hash."""
        self._unsupported("page probe")

    def downloads(self) -> list[dict[str, Any]]:
        """Downloads the driver has seen (empty by default)."""
        return []

    def network_events(self, handle: Any) -> list[dict[str, Any]]:
        """Requests the page made (failures / HTTP errors).

        Observation only — there is deliberately no backend API
        for firing arbitrary network requests.
        """
        return []

    def dialogs(self, handle: Any) -> list[dict[str, Any]]:
        """Dialogs (alert/confirm/prompt) the page raised.

        Observation only: the driver records and dismisses them
        so a page can never hang the agent; the controller only
        reports what happened.
        """
        return []

    def create_profile_context(
        self, name: str, options: dict[str, Any]
    ) -> str:
        """Create an isolated profile context (own cookies/storage)."""
        self._unsupported("profile contexts")

    def set_active_profile(self, name: str) -> None:
        """Make the named profile context the active one."""
        self._unsupported("profile contexts")

    def interactive_elements(
        self, handle: Any, limit: int
    ) -> list[Any]:
        """Return handles of the page's interactive elements."""
        self._unsupported("listing interactive elements")

    def list_pages(self) -> list[Any]:
        """All page handles the driver knows, including popups the
        controller did not create.  Default: none tracked."""
        return []

    def wait_for(
        self, handle: Any, spec: dict[str, Any], timeout_ms: int
    ) -> None:
        """Block until a condition holds (element/text/URL/title).

        Must be condition-based (driver events/polling), never a
        fixed sleep.  Raises BrowserException(TIMEOUT) when the
        condition does not hold within ``timeout_ms``.
        """
        self._unsupported("waiting for page conditions")

    def screenshot(self, handle: Any) -> bytes:
        """Capture the page as PNG bytes."""
        self._unsupported("screenshots")

    # -- computer-use primitives --------------------------------------
    # Coordinate-level input for the perception layer's visual
    # fallback and hover/drag actions.  Safe defaults keep simple
    # adapters valid; real adapters override them.
    def element_box(
        self, handle: Any, element: Any
    ) -> dict[str, Any] | None:
        """Bounding box {x, y, width, height} of an element."""
        self._unsupported("element bounding boxes")

    def mouse_click(
        self, handle: Any, x: float, y: float, click_count: int = 1
    ) -> None:
        self._unsupported("coordinate clicking")

    def mouse_move(self, handle: Any, x: float, y: float) -> None:
        self._unsupported("mouse movement")

    def mouse_drag(
        self, handle: Any, x1: float, y1: float, x2: float, y2: float
    ) -> None:
        self._unsupported("mouse dragging")

    def focus_element(
        self, handle: Any, element: Any, timeout_ms: int
    ) -> None:
        self._unsupported("focusing elements")

    def set_checked(
        self, handle: Any, element: Any, checked: bool, timeout_ms: int
    ) -> None:
        self._unsupported("checking elements")

    # -- runtime health & capabilities ------------------------------

    def is_alive(self) -> bool:
        """Whether the browser process is still responsive.

        In-process fakes are always alive; real adapters check
        their browser process/connection.  The runtime uses this
        for crash detection — it must never raise.
        """
        return True

    def capabilities(self) -> dict[str, bool]:
        """Engine capabilities this adapter offers.

        Keys are Afnan-level capability names (tabs,
        accessibility, screenshots, ...), never engine-specific
        features.  The runtime merges these with its own.
        """
        return {"tabs": True}

"""Browser backends: the driver boundary of the BrowserController.

A *backend* is the thin, browser-specific layer that actually
talks to a browser process.  The :class:`BrowserController` owns
sessions, tabs and validation and only ever calls this interface,
so the browser technology can be swapped (or faked in tests)
without touching controller, tool or agent code.

The shipped backend is :class:`PlaywrightBackend` (Chromium /
Chrome / Edge / Firefox / WebKit through Microsoft's Playwright,
which works the same on Windows, macOS and Linux).  Playwright is
imported lazily inside :meth:`PlaywrightBackend.start`, so merely
constructing the backend — or importing this module — never
requires it; if it (or its browser binaries) is missing, ``start``
raises a structured ``browser_unavailable`` error instead of
crashing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from afnan_ai.browser.base import BrowserErrorCode, BrowserException

#: Browser names the controller accepts (mapped by the backend)
SUPPORTED_BROWSERS = ("chromium", "chrome", "edge", "firefox", "webkit")


class BrowserBackend(ABC):
    """Interface a concrete browser driver implements.

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


class PlaywrightBackend(BrowserBackend):
    """Real browser control via Playwright (sync API).

    Works on Windows, macOS and Linux.  Requires the ``playwright``
    package and its browser binaries::

        pip install playwright
        playwright install chromium
    """

    name = "playwright"

    #: our browser name -> (playwright browser type, channel)
    _BROWSER_MAP = {
        "chromium": ("chromium", None),
        "chrome": ("chromium", "chrome"),
        "edge": ("chromium", "msedge"),
        "firefox": ("firefox", None),
        "webkit": ("webkit", None),
    }

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._context = None

    # -- lifecycle ------------------------------------------------------
    def start(self, browser: str, headless: bool) -> None:
        sync_playwright = self._import_playwright()
        browser_type_name, channel = self._BROWSER_MAP.get(
            browser, ("chromium", None)
        )
        try:
            self._playwright = sync_playwright().start()
            browser_type = getattr(self._playwright, browser_type_name)
            launch_kwargs: dict[str, Any] = {"headless": headless}
            if channel:
                launch_kwargs["channel"] = channel
            self._browser = browser_type.launch(**launch_kwargs)
            self._context = self._browser.new_context()
        except BrowserException:
            self._cleanup()
            raise
        except Exception as e:
            self._cleanup()
            raise BrowserException(
                f"Could not launch browser {browser!r}: {e}. "
                "Make sure the browser binaries are installed "
                "(`playwright install`).",
                code=BrowserErrorCode.BROWSER_UNAVAILABLE,
                details={"browser": browser},
            ) from e

    def connect(self, endpoint: str) -> None:
        sync_playwright = self._import_playwright()
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.connect_over_cdp(endpoint)
            contexts = self._browser.contexts
            self._context = contexts[0] if contexts else self._browser.new_context()
        except BrowserException:
            self._cleanup()
            raise
        except Exception as e:
            self._cleanup()
            raise BrowserException(
                f"Could not connect to browser at {endpoint!r}: {e}",
                code=BrowserErrorCode.CONNECTION_FAILED,
                details={"endpoint": endpoint},
            ) from e

    def stop(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        try:
            if self._browser is not None:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright is not None:
                self._playwright.stop()
        except Exception:
            pass
        self._browser = None
        self._context = None
        self._playwright = None

    @staticmethod
    def _import_playwright():
        try:
            from playwright.sync_api import sync_playwright
        except Exception as e:
            raise BrowserException(
                "Playwright is not installed. Install it with "
                "`pip install playwright` and "
                "`playwright install chromium` to enable browser "
                "control.",
                code=BrowserErrorCode.BROWSER_UNAVAILABLE,
            ) from e
        return sync_playwright

    def _require_page(self):
        if self._context is None:
            raise BrowserException(
                "Browser is not running",
                code=BrowserErrorCode.BROWSER_NOT_STARTED,
            )
        return self._context

    # -- pages ------------------------------------------------------------
    def new_page(self) -> Any:
        return self._require_page().new_page()

    def close_page(self, handle: Any) -> None:
        handle.close()

    def goto(self, handle: Any, url: str) -> None:
        handle.goto(url)

    def go_back(self, handle: Any) -> None:
        handle.go_back()

    def go_forward(self, handle: Any) -> None:
        handle.go_forward()

    def reload(self, handle: Any) -> None:
        handle.reload()

    def page_url(self, handle: Any) -> str:
        return handle.url or ""

    def page_title(self, handle: Any) -> str:
        return handle.title() or ""

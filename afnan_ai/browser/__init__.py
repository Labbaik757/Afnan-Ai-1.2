"""BrowserController system for Afnan AI.

Programmatic browser control (launch/connect, tabs, navigation,
page state, history, shutdown) that works the same on Windows,
macOS and Linux.  The browser-specific work lives in a swappable
:class:`BrowserBackend` (Playwright by default); the
:class:`BrowserController` owns the session and the browser
Tools expose it through the ToolRegistry, so the core Agent only
ever sees ordinary Tools.

Importing this package never imports a browser driver or launches
anything.
"""

from afnan_ai.browser.backend import (
    SUPPORTED_BROWSERS,
    BrowserBackend,
    PlaywrightBackend,
)
from afnan_ai.browser.base import (
    BrowserError,
    BrowserErrorCode,
    BrowserException,
    ElementInfo,
    PageState,
    TabInfo,
)
from afnan_ai.browser.controller import BrowserController
from afnan_ai.browser.tools import create_browser_tools, register_browser_tools

__all__ = [
    "SUPPORTED_BROWSERS",
    "BrowserBackend",
    "BrowserController",
    "BrowserError",
    "BrowserErrorCode",
    "BrowserException",
    "ElementInfo",
    "PageState",
    "PlaywrightBackend",
    "TabInfo",
    "create_browser_tools",
    "register_browser_tools",
]

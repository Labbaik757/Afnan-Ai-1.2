"""Browser system for Afnan AI.

Programmatic browser control (launch/connect, tabs, navigation,
page state, history, shutdown) that works the same on Windows,
macOS and Linux.  The layering is::

    BrowserController → AfnanBrowserRuntime
        → BrowserEngineAdapter → ChromiumAdapter → Chromium
        → BrowserEngineAdapter → PlaywrightAdapter → Chromium
          (fallback / development)

The controller owns sessions/tabs/validation, the runtime owns
browser lifecycle and recoverable state, and only an adapter
talks to a concrete engine — so the engine can be swapped (a
future customized Afnan Chromium build included) without
touching controller, tools or the core agent.

Importing this package never imports a browser driver or launches
anything.
"""

from afnan_ai.browser.backend import (
    SUPPORTED_BROWSERS,
    BrowserBackend,
    PlaywrightAdapter,
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
from afnan_ai.browser.chromium_adapter import ChromiumAdapter
from afnan_ai.browser.engine import BrowserEngineAdapter
from afnan_ai.browser.models import (
    BrowserAction,
    BrowserElement,
    BrowserObservation,
    BrowserPage,
    BrowserProfile,
    BrowserResult,
    BrowserSession,
    BrowserTab,
    BrowserWindow,
)
from afnan_ai.browser.reliability import BrowserReliability
from afnan_ai.browser.research import WebResearch
from afnan_ai.browser.runtime import AfnanBrowserRuntime
from afnan_ai.browser.security import (
    ActionRisk,
    ApprovalDecision,
    ApprovalGate,
    ApprovalRequest,
    SecurityPolicy,
    Sensitivity,
    classify_action,
)
from afnan_ai.browser.tools import create_browser_tools, register_browser_tools
from afnan_ai.browser.workflow import BrowserTaskResult, BrowserWorkflow

__all__ = [
    "SUPPORTED_BROWSERS",
    "ActionRisk",
    "AfnanBrowserRuntime",
    "ApprovalDecision",
    "ApprovalGate",
    "ApprovalRequest",
    "BrowserAction",
    "BrowserBackend",
    "BrowserController",
    "BrowserElement",
    "BrowserEngineAdapter",
    "BrowserObservation",
    "BrowserPage",
    "BrowserProfile",
    "BrowserReliability",
    "BrowserError",
    "BrowserErrorCode",
    "BrowserException",
    "BrowserResult",
    "BrowserSession",
    "BrowserTab",
    "BrowserTaskResult",
    "BrowserWindow",
    "BrowserWorkflow",
    "ChromiumAdapter",
    "ElementInfo",
    "PageState",
    "PlaywrightAdapter",
    "PlaywrightBackend",
    "SecurityPolicy",
    "Sensitivity",
    "TabInfo",
    "WebResearch",
    "classify_action",
    "create_browser_tools",
    "register_browser_tools",
]

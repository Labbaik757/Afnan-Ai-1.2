"""Browser operations as Tools for the ToolRegistry.

Every BrowserController operation is wrapped in a :class:`Tool`
with a name, an LLM-readable description and an input schema, so
the Planner and the Agent can select and execute browser actions
through the exact same ToolRegistry machinery as every other
capability (``open_url``, ``search_google``, ...).

Failure model: a browser failure (unavailable, not started,
invalid tab, navigation failed, ...) comes back from
``ToolRegistry.execute`` as ``ToolResult(success=False)`` with a
structured error whose ``details`` carry the browser error code —
never a crash, never a silent success.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.browser.base import BrowserException
from afnan_ai.browser.controller import BrowserController
from afnan_ai.tools.base import Tool, ToolExecutionError


class _BrowserTool(Tool):
    """Base class: holds the shared BrowserController and turns
    BrowserExceptions into structured ToolExecutionErrors."""

    def __init__(self, controller: BrowserController):
        self.controller = controller

    def _call(self, operation, *args: Any, **kwargs: Any) -> Any:
        try:
            return operation(*args, **kwargs)
        except BrowserException as e:
            raise ToolExecutionError(
                e.error.message,
                tool=self.name,
                details={"browser_error": e.error.to_dict()},
            ) from e


class BrowserLaunchTool(_BrowserTool):
    name = "browser_launch"
    description = (
        "Launch a web browser (Chromium by default; also chrome, "
        "edge, firefox, webkit) so browser actions can be used. "
        "Safe to call when the browser is already running."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "browser": {"type": "string"},
            "headless": {"type": "boolean"},
        },
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(
            self.controller.launch,
            browser=arguments.get("browser"),
            headless=arguments.get("headless"),
        )


class BrowserConnectTool(_BrowserTool):
    name = "browser_connect"
    description = (
        "Connect to an already-running browser via its CDP "
        "endpoint (e.g. http://localhost:9222) instead of "
        "launching a new one."
    )
    input_schema = {
        "type": "object",
        "properties": {"endpoint": {"type": "string"}},
        "required": ["endpoint"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(self.controller.connect, arguments["endpoint"])


class BrowserNewTabTool(_BrowserTool):
    name = "browser_new_tab"
    description = (
        "Open a new browser tab (optionally at a URL) and make it "
        "the active tab. Requires the browser to be launched."
    )
    input_schema = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(self.controller.new_tab, arguments.get("url"))


class BrowserListTabsTool(_BrowserTool):
    name = "browser_list_tabs"
    description = "List all open browser tabs with their URLs and titles."
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return {"tabs": self._call(self.controller.list_tabs)}


class BrowserSelectTabTool(_BrowserTool):
    name = "browser_select_tab"
    description = "Make an open browser tab (by tab_id) the active tab."
    input_schema = {
        "type": "object",
        "properties": {"tab_id": {"type": "string"}},
        "required": ["tab_id"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(self.controller.select_tab, arguments["tab_id"])


class BrowserCloseTabTool(_BrowserTool):
    name = "browser_close_tab"
    description = (
        "Close a browser tab by tab_id (the active tab when "
        "tab_id is omitted)."
    )
    input_schema = {
        "type": "object",
        "properties": {"tab_id": {"type": "string"}},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(self.controller.close_tab, arguments.get("tab_id"))


class BrowserNavigateTool(_BrowserTool):
    name = "browser_navigate"
    description = (
        "Navigate a browser tab to a URL (the active tab, or "
        "tab_id when given). Creates a tab first if none is open. "
        "Requires the browser to be launched."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "tab_id": {"type": "string"},
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(
            self.controller.navigate,
            arguments["url"],
            tab_id=arguments.get("tab_id"),
        )


class BrowserCurrentPageTool(_BrowserTool):
    name = "browser_current_page"
    description = (
        "Read the current page state of a browser tab: its tab_id, "
        "current URL and page title."
    )
    input_schema = {
        "type": "object",
        "properties": {"tab_id": {"type": "string"}},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(
            self.controller.current_page, tab_id=arguments.get("tab_id")
        )


class BrowserBackTool(_BrowserTool):
    name = "browser_back"
    description = "Go back one page in a browser tab's history."
    input_schema = {
        "type": "object",
        "properties": {"tab_id": {"type": "string"}},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(self.controller.back, tab_id=arguments.get("tab_id"))


class BrowserForwardTool(_BrowserTool):
    name = "browser_forward"
    description = "Go forward one page in a browser tab's history."
    input_schema = {
        "type": "object",
        "properties": {"tab_id": {"type": "string"}},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(
            self.controller.forward, tab_id=arguments.get("tab_id")
        )


class BrowserReloadTool(_BrowserTool):
    name = "browser_reload"
    description = "Reload the current page of a browser tab."
    input_schema = {
        "type": "object",
        "properties": {"tab_id": {"type": "string"}},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(
            self.controller.reload, tab_id=arguments.get("tab_id")
        )


class BrowserShutdownTool(_BrowserTool):
    name = "browser_shutdown"
    description = "Close all browser tabs and shut the browser down."
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(self.controller.shutdown)


def create_browser_tools(
    controller: BrowserController,
) -> list[Tool]:
    """Create the full set of browser Tools bound to *controller*."""
    return [
        BrowserLaunchTool(controller),
        BrowserConnectTool(controller),
        BrowserNewTabTool(controller),
        BrowserListTabsTool(controller),
        BrowserSelectTabTool(controller),
        BrowserCloseTabTool(controller),
        BrowserNavigateTool(controller),
        BrowserCurrentPageTool(controller),
        BrowserBackTool(controller),
        BrowserForwardTool(controller),
        BrowserReloadTool(controller),
        BrowserShutdownTool(controller),
    ]


def register_browser_tools(
    registry, controller: BrowserController | None = None
) -> BrowserController:
    """Register all browser Tools on *registry* and return the
    controller they are bound to (creating one when not given)."""
    controller = controller or BrowserController()
    for tool in create_browser_tools(controller):
        if not registry.has(tool.name):
            registry.register(tool)
    return controller

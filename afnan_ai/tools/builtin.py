"""Built-in tools — the agent's standard capabilities
(open URL/app, web search, screenshot) wrapped as Tools:

* ``open_url`` — ``webbrowser.open(url)``
* ``open_application`` — the platform adapter's ``launch_app``
  (VS Code, Chrome, Edge, WhatsApp, Safari, ...)
* ``search_google`` — build a Google search URL and open it
* ``search_youtube`` — build a YouTube search URL and open it
* ``take_screenshot`` — capture the screen, save it and open it
  through the platform adapter

Dependencies (platform adapter, browser opener, screenshot capture)
are injected, which keeps the tools testable on any OS and free of
platform-specific code — the platform adapter remains the only
place with OS-specific launching.
"""

from __future__ import annotations

import urllib.parse
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from afnan_ai.platform.base import PlatformAdapter
from afnan_ai.tools.base import Tool, ToolExecutionError
from afnan_ai.tools.registry import ToolRegistry

_STRING = {"type": "string"}


def _open_in_browser(url: str):
    """Open *url* in the default browser (looked up at call time so
    tests can patch ``webbrowser.open``)."""
    return webbrowser.open(url)


class OpenUrlTool(Tool):
    name = "open_url"
    description = "Open a URL in the default web browser."
    input_schema = {
        "type": "object",
        "properties": {
            "url": {**_STRING, "description": "The URL to open."},
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    def __init__(self, opener: Callable[[str], Any] | None = None):
        self._opener = opener or _open_in_browser

    def run(self, arguments: dict[str, Any]) -> Any:
        url = str(arguments["url"]).strip()
        try:
            self._opener(url)
        except Exception as e:
            raise ToolExecutionError(
                f"Could not open URL {url!r}: {e}", tool=self.name
            ) from e
        return {"url": url, "opened": True}


class OpenApplicationTool(Tool):
    name = "open_application"
    description = (
        "Open a desktop application by key, e.g. 'chrome', 'vscode', "
        "'edge', 'whatsapp' or 'safari'."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "application": {
                **_STRING,
                "description": "Application key, e.g. 'chrome'.",
            },
        },
        "required": ["application"],
        "additionalProperties": False,
    }

    # Friendly names the voice commands have always accepted
    _ALIASES = {
        "google chrome": "chrome",
        "vs code": "vscode",
        "visual studio code": "vscode",
        "microsoft edge": "edge",
    }

    def __init__(self, adapter: PlatformAdapter | None):
        self._adapter = adapter

    def run(self, arguments: dict[str, Any]) -> Any:
        app = str(arguments["application"]).strip().lower()
        app = self._ALIASES.get(app, app)
        if self._adapter is None:
            raise ToolExecutionError(
                "No platform adapter configured, cannot open applications",
                tool=self.name,
                details={"application": app},
            )
        try:
            launched = self._adapter.launch_app(app)
        except Exception as e:
            raise ToolExecutionError(
                f"Could not open application {app!r}: {e}",
                tool=self.name,
                details={"application": app},
            ) from e
        if not launched:
            raise ToolExecutionError(
                f"Application {app!r} is not available on this system",
                tool=self.name,
                details={"application": app},
            )
        return {"application": app, "launched": True}


class SearchGoogleTool(Tool):
    name = "search_google"
    description = "Search Google for a query and open the results."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {**_STRING, "description": "What to search for."},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, opener: Callable[[str], Any] | None = None):
        self._opener = opener or _open_in_browser

    @staticmethod
    def build_url(query: str) -> str:
        return "https://www.google.com/search?q=" + urllib.parse.quote(query)

    def run(self, arguments: dict[str, Any]) -> Any:
        query = str(arguments["query"]).strip()
        url = self.build_url(query)
        try:
            self._opener(url)
        except Exception as e:
            raise ToolExecutionError(
                f"Could not open Google search for {query!r}: {e}",
                tool=self.name,
            ) from e
        return {"query": query, "url": url, "opened": True}


class SearchYouTubeTool(Tool):
    name = "search_youtube"
    description = "Search YouTube for a query and open the results."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {**_STRING, "description": "What to search for."},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, opener: Callable[[str], Any] | None = None):
        self._opener = opener or _open_in_browser

    @staticmethod
    def build_url(query: str) -> str:
        return (
            "https://www.youtube.com/results?search_query="
            + urllib.parse.quote(query)
        )

    def run(self, arguments: dict[str, Any]) -> Any:
        query = str(arguments["query"]).strip()
        url = self.build_url(query)
        try:
            self._opener(url)
        except Exception as e:
            raise ToolExecutionError(
                f"Could not open YouTube search for {query!r}: {e}",
                tool=self.name,
            ) from e
        return {"query": query, "url": url, "opened": True}


class TakeScreenshotTool(Tool):
    name = "take_screenshot"
    description = "Take a screenshot, save it and open it."
    input_schema = {
        "type": "object",
        "properties": {
            "output_dir": {
                **_STRING,
                "description": "Folder to save into (default 'screenshots').",
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    def __init__(
        self,
        adapter: PlatformAdapter | None,
        capture: Callable[[], Any] | None = None,
    ):
        self._adapter = adapter
        self._capture = capture

    def _capture_screen(self):
        if self._capture is not None:
            return self._capture()
        try:
            import pyautogui
        except Exception:
            return None
        return pyautogui.screenshot()

    def run(self, arguments: dict[str, Any]) -> Any:
        try:
            image = self._capture_screen()
        except Exception as e:
            raise ToolExecutionError(
                f"Screenshot capture failed: {e}", tool=self.name
            ) from e
        if image is None:
            raise ToolExecutionError(
                "Screenshot is not available (screen capture unavailable)",
                tool=self.name,
            )

        screenshots_dir = Path(arguments.get("output_dir") or "screenshots")
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = screenshots_dir / f"screenshot_{timestamp}.png"
        try:
            image.save(str(file_path))
        except Exception as e:
            raise ToolExecutionError(
                f"Could not save screenshot: {e}", tool=self.name
            ) from e

        if self._adapter is not None:
            self._adapter.open_path(str(file_path))
        return str(file_path)


def create_default_registry(
    adapter: PlatformAdapter | None = None,
    *,
    opener: Callable[[str], Any] | None = None,
    screenshot_capture: Callable[[], Any] | None = None,
) -> ToolRegistry:
    """Registry pre-loaded with Afnan's built-in tools."""
    return ToolRegistry(
        [
            OpenUrlTool(opener=opener),
            OpenApplicationTool(adapter),
            SearchGoogleTool(opener=opener),
            SearchYouTubeTool(opener=opener),
            TakeScreenshotTool(adapter, capture=screenshot_capture),
        ]
    )

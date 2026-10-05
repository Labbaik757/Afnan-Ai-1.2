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
from afnan_ai.browser.challenge import detect_challenge
from afnan_ai.browser.controller import BrowserController
from afnan_ai.browser.extraction import clean_document
from afnan_ai.browser.pagination import Paginator
from afnan_ai.browser.research import WebResearch
from afnan_ai.browser.semantics import rank as rank_matches
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
        "the active tab. Requires the browser to be launched. A "
        "'purpose' labels why the task opened this tab, so "
        "multi-tab work stays attributable."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "purpose": {"type": "string"},
        },
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(
            self.controller.new_tab,
            arguments.get("url"),
            purpose=arguments.get("purpose"),
        )


class BrowserListTabsTool(_BrowserTool):
    name = "browser_list_tabs"
    description = (
        "List all open browser tabs with their URLs, titles and "
        "task purposes, and which one is active."
    )
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        tabs = self._call(self.controller.list_tabs)
        return {
            "tabs": tabs,
            "observation": {
                "type": "tab_list",
                "summary": "; ".join(
                    f"{t['tab_id']} [{t.get('purpose') or 'no purpose'}] "
                    f"{t['title'] or t['url']}"
                    for t in tabs
                ) or "No tabs open",
                "count": len(tabs),
            },
        }


class BrowserSetTabPurposeTool(_BrowserTool):
    """Label a tab with its task-level purpose."""

    name = "browser_set_tab_purpose"
    description = (
        "Label a browser tab with the purpose it serves in the "
        "current task (e.g. 'pricing research'), so later steps "
        "can identify the right tab before acting on it."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "tab_id": {"type": "string"},
            "purpose": {"type": "string"},
        },
        "required": ["tab_id", "purpose"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(
            self.controller.set_tab_purpose,
            arguments["tab_id"],
            arguments["purpose"],
        )


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


# ----------------------------------------------------------------------
# Page interaction tools
#
# A *target* is either "ref" (an element reference returned by
# browser_find_elements) or locator fields (selector / test_id /
# label / placeholder / role+name / text) which are resolved fresh
# against the live page.  Stable selectors are preferred; role,
# label, placeholder and text lookups are the accessibility
# fallback.  Every interaction validates its target first.
# ----------------------------------------------------------------------

_TARGET_PROPERTIES = {
    "ref": {"type": "string"},
    "selector": {"type": "string"},
    "test_id": {"type": "string"},
    "label": {"type": "string"},
    "placeholder": {"type": "string"},
    "role": {"type": "string"},
    "name": {"type": "string"},
    "text": {"type": "string"},
    "frame": {"type": "string"},
    "tab_id": {"type": "string"},
    "timeout_ms": {"type": "integer"},
}


def _target_from(arguments: dict[str, Any]) -> dict[str, Any] | None:
    if arguments.get("ref"):
        return {"ref": arguments["ref"]}
    locator = {
        key: arguments[key]
        for key in ("selector", "test_id", "label", "placeholder", "role", "name", "text", "frame")
        if arguments.get(key)
    }
    return {"locator": locator} if locator else None


def _tab_and_timeout(arguments: dict[str, Any]):
    return arguments.get("tab_id"), arguments.get("timeout_ms")


class BrowserFindElementsTool(_BrowserTool):
    name = "browser_find_elements"
    description = (
        "Find elements on the current page by selector, test id, "
        "label, placeholder, accessibility role/name or visible "
        "text, and return each element's identity (ref) and basic "
        "properties (tag, text, attributes, visible, enabled)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "selector": {"type": "string"},
            "test_id": {"type": "string"},
            "label": {"type": "string"},
            "placeholder": {"type": "string"},
            "role": {"type": "string"},
            "name": {"type": "string"},
            "text": {"type": "string"},
            "frame": {"type": "string"},
            "tab_id": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        locator = {
            key: arguments[key]
            for key in ("selector", "test_id", "label", "placeholder", "role", "name", "text", "frame")
            if arguments.get(key)
        }
        elements = self._call(
            self.controller.find_elements,
            locator,
            tab_id=arguments.get("tab_id"),
            limit=arguments.get("limit", 10),
        )
        return {"elements": elements, "count": len(elements)}


class BrowserInspectElementTool(_BrowserTool):
    name = "browser_inspect_element"
    description = (
        "Read one element's identity and basic properties (tag, "
        "text, id/name/type/role attributes, value, visible, "
        "enabled, editable) by ref or locator."
    )
    input_schema = {
        "type": "object",
        "properties": dict(_TARGET_PROPERTIES),
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(
            self.controller.inspect_element,
            _target_from(arguments),
            tab_id=arguments.get("tab_id"),
        )


class BrowserClickTool(_BrowserTool):
    name = "browser_click"
    description = (
        "Click an element identified by ref (from "
        "browser_find_elements) or by selector/role/text locator. "
        "The target is validated (exists, visible, enabled) "
        "before clicking."
    )
    input_schema = {
        "type": "object",
        "properties": dict(_TARGET_PROPERTIES),
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        tab_id, timeout = _tab_and_timeout(arguments)
        return self._call(
            self.controller.click,
            _target_from(arguments),
            tab_id=tab_id,
            timeout_ms=timeout,
        )


class BrowserTypeTool(_BrowserTool):
    name = "browser_type"
    description = (
        "Type text into an input or textarea identified by ref or "
        "locator. Set clear_first to replace the current value."
    )
    input_schema = {
        "type": "object",
        "properties": {
            **_TARGET_PROPERTIES,
            "text": {"type": "string"},
            "clear_first": {"type": "boolean"},
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        tab_id, timeout = _tab_and_timeout(arguments)
        # "text" is the value being typed here, not a locator —
        # keep it out of the target or it would be mistaken for
        # a visible-text locator.
        target_args = {
            key: value
            for key, value in arguments.items()
            if key != "text"
        }
        return self._call(
            self.controller.type_text,
            _target_from(target_args),
            arguments["text"],
            tab_id=tab_id,
            clear_first=bool(arguments.get("clear_first", False)),
            timeout_ms=timeout,
        )


class BrowserClearTool(_BrowserTool):
    name = "browser_clear"
    description = "Clear the current value of an input or textarea."
    input_schema = {
        "type": "object",
        "properties": dict(_TARGET_PROPERTIES),
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        tab_id, timeout = _tab_and_timeout(arguments)
        return self._call(
            self.controller.clear_field,
            _target_from(arguments),
            tab_id=tab_id,
            timeout_ms=timeout,
        )


class BrowserSelectOptionTool(_BrowserTool):
    name = "browser_select_option"
    description = "Select an option (by value) in a <select> dropdown."
    input_schema = {
        "type": "object",
        "properties": {
            **_TARGET_PROPERTIES,
            "value": {"type": "string"},
        },
        "required": ["value"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        tab_id, timeout = _tab_and_timeout(arguments)
        return self._call(
            self.controller.select_option,
            _target_from(arguments),
            arguments["value"],
            tab_id=tab_id,
            timeout_ms=timeout,
        )


class BrowserPressKeyTool(_BrowserTool):
    name = "browser_press_key"
    description = (
        "Press a keyboard key (e.g. Enter, Tab, Escape, ArrowDown, "
        "Control+A) on an element (by ref/locator) or on the page."
    )
    input_schema = {
        "type": "object",
        "properties": {
            **_TARGET_PROPERTIES,
            "key": {"type": "string"},
        },
        "required": ["key"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        tab_id, timeout = _tab_and_timeout(arguments)
        return self._call(
            self.controller.press_key,
            arguments["key"],
            _target_from(arguments),
            tab_id=tab_id,
            timeout_ms=timeout,
        )


class BrowserScrollTool(_BrowserTool):
    name = "browser_scroll"
    description = (
        "Scroll the page by dx/dy pixels, or scroll an element "
        "(by ref/locator) into view when a target is given."
    )
    input_schema = {
        "type": "object",
        "properties": {
            **_TARGET_PROPERTIES,
            "dx": {"type": "integer"},
            "dy": {"type": "integer"},
        },
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        target = _target_from(arguments)
        dx = arguments.get("dx", 0)
        dy = arguments.get("dy", 0)
        if target is None and not dx and not dy:
            dy = 600  # a plain "scroll" means one page down
        return self._call(
            self.controller.scroll_page,
            dx=dx,
            dy=dy,
            target=target,
            tab_id=arguments.get("tab_id"),
        )


class BrowserObservePageTool(_BrowserTool):
    """Structured snapshot of what the page currently shows."""

    name = "browser_observe_page"
    description = (
        "Read the current page's structured state: URL, title, "
        "visible text and the interactive elements (each with a "
        "ref usable by browser_click/browser_type), plus whether "
        "the page changed since the last observation. Recorded "
        "in AgentState as an observation."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "tab_id": {"type": "string"},
            "max_elements": {"type": "integer"},
            "text_limit": {"type": "integer"},
        },
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(
            self.controller.observe,
            tab_id=arguments.get("tab_id"),
            max_elements=arguments.get("max_elements", 25),
            text_limit=arguments.get("text_limit", 2000),
        )


class BrowserWaitForTool(_BrowserTool):
    """Condition-based waiting (no fixed sleeps)."""

    name = "browser_wait_for"
    description = (
        "Wait until the page reaches a state: an element appears "
        "(condition=element_present) or disappears (element_hidden), "
        "text is present (text_present), or the URL/title contains "
        "a value (url_contains/title_contains). Locator fields "
        "(selector/test_id/label/placeholder/role/name) locate the "
        "element; 'text' is used by text_present; 'value' by the "
        "URL/title conditions. Times out with a structured error."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "condition": {
                "type": "string",
                "enum": [
                    "element_present",
                    "element_hidden",
                    "text_present",
                    "url_contains",
                    "title_contains",
                ],
            },
            "selector": {"type": "string"},
            "test_id": {"type": "string"},
            "label": {"type": "string"},
            "placeholder": {"type": "string"},
            "role": {"type": "string"},
            "name": {"type": "string"},
            "text": {"type": "string"},
            "value": {"type": "string"},
            "tab_id": {"type": "string"},
            "timeout_ms": {"type": "integer"},
        },
        "required": ["condition"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        locator = {
            key: arguments[key]
            for key in (
                "selector", "test_id", "label", "placeholder",
                "role", "name", "text",
            )
            if arguments.get(key)
        }
        return self._call(
            self.controller.wait_for,
            arguments["condition"],
            locator=locator or None,
            text=arguments.get("text"),
            value=arguments.get("value"),
            tab_id=arguments.get("tab_id"),
            timeout_ms=arguments.get("timeout_ms"),
        )


class BrowserScreenshotTool(_BrowserTool):
    """Visual snapshot of the page, saved to disk."""

    name = "browser_screenshot"
    description = (
        "Capture a PNG screenshot of the current page. Returns "
        "the saved file path plus page URL/title as a structured "
        "observation recorded in AgentState."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "tab_id": {"type": "string"},
            "output_dir": {"type": "string"},
            "filename": {"type": "string"},
        },
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(
            self.controller.screenshot,
            tab_id=arguments.get("tab_id"),
            output_dir=arguments.get("output_dir"),
            filename=arguments.get("filename"),
        )


class BrowserUploadFileTool(_BrowserTool):
    """Attach a local file to a file input (sensitive action)."""

    name = "browser_upload_file"
    description = (
        "Upload a local file through an <input type=file> element "
        "(by ref or locator). This is a sensitive action: it runs "
        "only with human approval when an approval gate requires "
        "it, and fails with a structured error instead."
    )
    input_schema = {
        "type": "object",
        "properties": {
            **_TARGET_PROPERTIES,
            "path": {"type": "string"},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(
            self.controller.upload_file,
            _target_from(arguments),
            arguments["path"],
            tab_id=arguments.get("tab_id"),
            timeout_ms=arguments.get("timeout_ms"),
        )


class BrowserAccessibilityTreeTool(_BrowserTool):
    """Read the page's normalized accessibility tree."""

    name = "browser_accessibility_tree"
    description = (
        "Read the current page's accessibility tree as structured "
        "nodes (role, name, value, heading level) for buttons, "
        "links, inputs, headings, menus and other elements. The "
        "browser's accessibility tree is preferred; a DOM-derived "
        "tree is returned when it is unavailable. Prefer this "
        "over guessing from pixels."
    )
    input_schema = {
        "type": "object",
        "properties": {"tab_id": {"type": "string"}},
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(
            self.controller.accessibility_tree,
            tab_id=arguments.get("tab_id"),
        )


class BrowserFindSemanticTool(_BrowserTool):
    """Locate an element by natural-language description."""

    name = "browser_find_semantic"
    description = (
        "Find page elements by description (e.g. 'Login button', "
        "'Search field', 'Next page link') using the "
        "accessibility tree and DOM metadata. Returns ranked "
        "matches with confidence scores; matches below 0.5 "
        "confidence carry no element reference and must not be "
        "acted on automatically."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "tab_id": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["description"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        tree = self._call(
            self.controller.accessibility_tree,
            tab_id=arguments.get("tab_id"),
        )
        ranked = rank_matches(arguments["description"], tree["nodes"])
        limit = int(arguments.get("limit", 5))
        matches = []
        for match in ranked[:limit]:
            entry = dict(match)
            if entry["tier"] == "low":
                # never hand out an actionable handle for a guess
                entry["element_ref"] = None
            matches.append(entry)
        best = matches[0] if matches else None
        uncertain = bool(best and best["tier"] != "actionable")
        return {
            "description": arguments["description"],
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
        }


class BrowserExtractContentTool(_BrowserTool):
    """Extract clean, structured content from the current page."""

    name = "browser_extract_content"
    description = (
        "Extract the current page's content in clean structured "
        "form: headings, paragraphs, lists, links and tables "
        "(row/column structure preserved), with boilerplate "
        "filtered and long pages split into bounded chunks. "
        "Pass chunk to read a specific chunk of a long page."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "tab_id": {"type": "string"},
            "max_chars": {"type": "integer"},
            "chunk": {"type": "integer"},
        },
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        doc = self._call(
            self.controller.page_document,
            tab_id=arguments.get("tab_id"),
        )
        content = clean_document(
            doc, max_chars=int(arguments.get("max_chars", 4000))
        )
        chunk_index = arguments.get("chunk")
        if chunk_index is not None:
            chunks = content["chunks"]
            try:
                chosen = chunks[int(chunk_index)]
            except (IndexError, ValueError, TypeError):
                chosen = None
            content = {
                **content,
                "chunks": [chosen] if chosen else [],
                "selected_chunk": (
                    chosen["index"] if chosen else None
                ),
            }
        content["observation"] = {
            "type": "page_content",
            "summary": (
                f"Extracted content from {content['title'][:60]!r} "
                f"({content['url']}): "
                f"{content['counts']['headings']} headings, "
                f"{content['counts']['paragraphs']} paragraphs, "
                f"{content['counts']['lists']} lists, "
                f"{content['counts']['tables']} tables, "
                f"{content['chunk_count']} chunk(s), "
                f"~{content['token_estimate']} tokens"
            ),
            "url": content["url"],
            "counts": content["counts"],
            "content_found": content["content_found"],
        }
        return content


class BrowserSearchTool(_BrowserTool):
    """Run a web search and extract structured results."""

    name = "browser_search"
    description = (
        "Search the web (duckduckgo, google or bing) and return "
        "structured results: rank, title, URL, snippet and "
        "source. Results are remembered so browser_open_result "
        "can open one by index. Navigates the current tab to "
        "the results page."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "engine": {"type": "string",
                       "enum": ["duckduckgo", "google", "bing"]},
            "max_results": {"type": "integer"},
            "tab_id": {"type": "string"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, controller, research):
        super().__init__(controller)
        self.research = research

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(
            self.research.search,
            arguments["query"],
            engine=arguments.get("engine", "duckduckgo"),
            max_results=int(arguments.get("max_results", 10)),
            tab_id=arguments.get("tab_id"),
        )


class BrowserOpenResultTool(_BrowserTool):
    """Open a stored search result and extract its content."""

    name = "browser_open_result"
    description = (
        "Open a result from the most recent browser_search (by "
        "0-based index) in a new tab tagged with its purpose, "
        "and extract the page's structured content."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "index": {"type": "integer"},
            "purpose": {"type": "string"},
            "max_chars": {"type": "integer"},
        },
        "required": ["index"],
        "additionalProperties": False,
    }

    def __init__(self, controller, research):
        super().__init__(controller)
        self.research = research

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._call(
            self.research.open_result,
            int(arguments["index"]),
            purpose=arguments.get("purpose"),
            max_chars=int(arguments.get("max_chars", 4000)),
        )


class BrowserWaitForStableTool(_BrowserTool):
    name = "browser_wait_for_stable"
    description = (
        "Wait until a dynamic (React/Next.js/Vue SPA) page stops "
        "changing: client-side routing, async rendering and DOM "
        "updates have settled. Condition-based — no blind sleeps. "
        "Reports the detected frameworks and page state."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "tab_id": {"type": "string"},
            "timeout_ms": {"type": "integer", "default": 5000},
        },
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._call(
            self.controller.wait_for_stable,
            tab_id=arguments.get("tab_id"),
            timeout_ms=int(arguments.get("timeout_ms") or 5000),
        )
        state = self.controller.current_page(result["tab_id"])
        return {
            **state,
            "stable": result["stable"],
            "ready_state": result["ready_state"],
            "frameworks": result["frameworks"],
            "text_length": result["text_length"],
            "element_count": result["element_count"],
            "observation": {
                "type": "spa_state",
                "summary": (
                    f"Page stable at {state.get('url')}; "
                    f"frameworks: "
                    f"{', '.join(result['frameworks']) or 'none'}."
                ),
                "page": state,
            },
        }


class BrowserCollectItemsTool(_BrowserTool):
    name = "browser_collect_items"
    description = (
        "Collect items from a long list: mode 'scroll' walks an "
        "infinite-scroll feed, mode 'next' clicks through pagination. "
        "Duplicates are skipped; max_items and max_pages guarantee "
        "the collection always terminates."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["scroll", "next"],
                "default": "scroll",
            },
            "item_selector": {"type": "string"},
            "max_items": {"type": "integer", "default": 100},
            "max_pages": {"type": "integer", "default": 10},
            "tab_id": {"type": "string"},
        },
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._call(
            Paginator(self.controller).collect,
            mode=str(arguments.get("mode") or "scroll"),
            item_selector=arguments.get("item_selector") or None,
            max_items=int(arguments.get("max_items") or 100),
            max_pages=int(arguments.get("max_pages") or 10),
            tab_id=arguments.get("tab_id"),
        )
        if not result.get("success"):
            raise ToolExecutionError(
                str(result.get("error") or "Collection failed."),
                tool=self.name,
                details={"pagination": result},
            )
        return result


class BrowserCheckChallengeTool(_BrowserTool):
    name = "browser_check_challenge"
    description = (
        "Check whether the page is showing a CAPTCHA or human "
        "verification challenge. Challenges are never solved or "
        "bypassed: a detected challenge returns human_required so "
        "the task can pause for the user."
    )
    input_schema = {
        "type": "object",
        "properties": {"tab_id": {"type": "string"}},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._call(
            detect_challenge, self.controller, arguments.get("tab_id")
        )
        return {
            **result,
            "status": (
                "human_required" if result["detected"] else "clear"
            ),
            "observation": {
                "type": "challenge_check",
                "summary": (
                    "Human check detected "
                    f"({result.get('type')}): human intervention "
                    "required."
                    if result["detected"]
                    else "No human check detected on this page."
                ),
            },
        }


class BrowserWaitChallengeTool(_BrowserTool):
    name = "browser_wait_challenge"
    description = (
        "Wait for the page's human check (CAPTCHA/verification) to "
        "be solved by the user in the browser. The challenge is "
        "never solved automatically; this only watches until it "
        "clears, then the task can continue."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "tab_id": {"type": "string"},
            "timeout_ms": {"type": "integer", "default": 120000},
        },
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._call(
            self.controller.wait_for_challenge_clear,
            tab_id=arguments.get("tab_id"),
            timeout_ms=int(arguments.get("timeout_ms") or 120000),
        )
        return {
            **result,
            "status": "cleared",
            "observation": {
                "type": "challenge_wait",
                "summary": (
                    "Human check cleared by the user; "
                    "task can continue."
                ),
            },
        }


class BrowserDownloadsTool(_BrowserTool):
    name = "browser_downloads"
    description = (
        "List browser downloads or wait for one to finish. Tracks "
        "filename, type, destination, size and integrity; unsafe "
        "(executable) downloads are flagged and never opened."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "wait"],
                "default": "list",
            },
            "download_id": {"type": "string"},
            "timeout_ms": {"type": "integer", "default": 30000},
        },
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        manager = self.controller.download_manager
        if str(arguments.get("action") or "list") == "wait":
            record = self._call(
                manager.wait_for,
                arguments.get("download_id") or None,
                timeout_ms=int(arguments.get("timeout_ms") or 30000),
            )
            downloads = [record]
        else:
            downloads = manager.list_downloads()
        return {
            "downloads": downloads,
            "count": len(downloads),
            "observation": {
                "type": "downloads",
                "summary": (
                    f"{len(downloads)} download(s) tracked."
                ),
                "downloads": downloads,
            },
        }


class BrowserNetworkStatusTool(_BrowserTool):
    name = "browser_network_status"
    description = (
        "Diagnostic summary of the page's network activity: "
        "failed requests, timeouts, blocked resources and HTTP "
        "errors. Observation only — it cannot send requests."
    )
    input_schema = {
        "type": "object",
        "properties": {"tab_id": {"type": "string"}},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        status = self._call(
            self.controller.network_status,
            tab_id=arguments.get("tab_id"),
        )
        return {
            **status,
            "observation": {
                "type": "network_status",
                "summary": (
                    f"{status['failed_requests']} failed "
                    f"request(s) of {status['total_requests']} "
                    f"on this page."
                ),
            },
        }


class BrowserRateLimitTool(_BrowserTool):
    name = "browser_rate_limit"
    description = (
        "Check whether the site is rate-limiting or blocking "
        "automated access, or take one controlled backoff and "
        "re-check. Never retries the failed action itself."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["check", "backoff"],
                "default": "check",
            },
            "tab_id": {"type": "string"},
            "max_wait_ms": {"type": "integer", "default": 5000},
        },
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if str(arguments.get("action") or "check") == "backoff":
            status = self._call(
                self.controller.rate_limit_backoff,
                tab_id=arguments.get("tab_id"),
                max_wait_ms=int(
                    arguments.get("max_wait_ms") or 5000
                ),
            )
        else:
            status = self._call(
                self.controller.rate_limit_status,
                tab_id=arguments.get("tab_id"),
            )
            status = {**status, "waited_ms": 0}
        return {
            **status,
            "observation": {
                "type": "rate_limit",
                "summary": (
                    "Rate limiting/blocking detected; "
                    "controlled backoff advised."
                    if status["limited"]
                    else "No rate limiting detected."
                ),
            },
        }


class BrowserApprovalsTool(_BrowserTool):
    name = "browser_approvals"
    description = (
        "List the approval gate's recorded decisions for "
        "sensitive browser actions (purchases, sends, account "
        "changes, uploads): what was asked, what the human "
        "decided, and when. Arguments are shown redacted."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 20},
        },
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        limit = max(1, int(arguments.get("limit") or 20))
        decisions = self.controller.approval_gate.decisions[-limit:]
        return {
            "decisions": decisions,
            "count": len(decisions),
            "observation": {
                "type": "approvals",
                "summary": (
                    f"{len(decisions)} approval decision(s) "
                    "recorded."
                ),
            },
        }


class BrowserProfilesTool(_BrowserTool):
    name = "browser_profiles"
    description = (
        "Manage isolated browser profiles: each profile has its "
        "own cookies, storage and tabs. Create a profile, select "
        "the one a task should use, or list them. One profile is "
        "active at a time and profiles never share session data."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "create", "select", "current"],
                "default": "list",
            },
            "name": {"type": "string"},
            "preferences": {"type": "object"},
        },
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        action = str(arguments.get("action") or "list")
        controller = self.controller
        if action == "list":
            result: dict[str, Any] = {
                "profiles": controller.list_profiles()
            }
        elif action == "current":
            result = {
                "profile": controller.profile_info(
                    controller.current_profile
                )
            }
        elif action == "create":
            result = {
                "profile": self._call(
                    controller.create_profile,
                    str(arguments.get("name") or ""),
                    arguments.get("preferences") or {},
                )
            }
        elif action == "select":
            result = {
                "profile": self._call(
                    controller.select_profile,
                    str(arguments.get("name") or ""),
                )
            }
        else:
            raise ToolExecutionError(
                f"Unknown profile action {action!r}.",
                tool=self.name,
            )
        return {
            **result,
            "observation": {
                "type": "browser_profiles",
                "summary": f"Profile action '{action}' completed.",
            },
        }


class BrowserSessionTool(_BrowserTool):
    name = "browser_session"
    description = (
        "Inspect or persist the browsing session: current tab "
        "context, navigation history, or save/load/restore a saved "
        "session so a later task can recover this task's context."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "current", "history", "export", "save", "load",
                    "restore",
                ],
                "default": "current",
            },
            "path": {"type": "string"},
            "tab_id": {"type": "string"},
            "limit": {"type": "integer", "default": 50},
        },
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        session = self.controller.session_manager
        action = str(arguments.get("action") or "current")
        if action == "current":
            result: dict[str, Any] = session.current()
        elif action == "history":
            entries = session.history(
                tab_id=arguments.get("tab_id"),
                limit=int(arguments.get("limit") or 50),
            )
            result = {"history": entries, "count": len(entries)}
        elif action == "export":
            result = {"session": session.export()}
        elif action in ("save", "load", "restore"):
            path = str(arguments.get("path") or "")
            if not path:
                raise ToolExecutionError(
                    f"browser_session {action} needs a 'path'.",
                    tool=self.name,
                )
            if action == "save":
                result = session.save(path)
            elif action == "load":
                data = session.load(path)
                result = {
                    "loaded": True,
                    "tabs": len(data.get("tabs") or []),
                    "history_entries": len(
                        data.get("history") or []
                    ),
                }
            else:
                result = session.restore(path)
        else:
            raise ToolExecutionError(
                f"Unknown session action {action!r}.",
                tool=self.name,
            )
        return {
            **result,
            "observation": {
                "type": "browser_session",
                "summary": f"Session action '{action}' completed.",
            },
        }


def create_browser_tools(
    controller: BrowserController,
) -> list[Tool]:
    """Create the full set of browser Tools bound to *controller*."""
    research = WebResearch(controller)
    # the session is reachable from the controller (and thus the
    # agent) for inspection; the tools close over the same one
    controller.research_session = research
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
        BrowserFindElementsTool(controller),
        BrowserInspectElementTool(controller),
        BrowserClickTool(controller),
        BrowserTypeTool(controller),
        BrowserClearTool(controller),
        BrowserSelectOptionTool(controller),
        BrowserPressKeyTool(controller),
        BrowserScrollTool(controller),
        BrowserObservePageTool(controller),
        BrowserWaitForTool(controller),
        BrowserScreenshotTool(controller),
        BrowserUploadFileTool(controller),
        BrowserAccessibilityTreeTool(controller),
        BrowserFindSemanticTool(controller),
        BrowserSetTabPurposeTool(controller),
        BrowserExtractContentTool(controller),
        BrowserSearchTool(controller, research),
        BrowserOpenResultTool(controller, research),
        BrowserWaitForStableTool(controller),
        BrowserCollectItemsTool(controller),
        BrowserCheckChallengeTool(controller),
        BrowserWaitChallengeTool(controller),
        BrowserDownloadsTool(controller),
        BrowserNetworkStatusTool(controller),
        BrowserRateLimitTool(controller),
        BrowserApprovalsTool(controller),
        BrowserProfilesTool(controller),
        BrowserSessionTool(controller),
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

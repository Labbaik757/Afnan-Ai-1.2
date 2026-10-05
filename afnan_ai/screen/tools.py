"""ScreenObserver capabilities as Tools.

The Planner sees the screen only through these tools:
structured observations (dimensions, element refs, regions,
confidence scores) — never raw image bytes.  Screen failures
become structured tool errors, and every observation output
carries the ``observation`` summary the Executor records into
AgentState automatically.
"""
from __future__ import annotations

from typing import Any

from afnan_ai.screen.base import ScreenException
from afnan_ai.screen.observer import ScreenObserver
from afnan_ai.tools.base import Tool, ToolExecutionError


class _ScreenTool(Tool):
    """Base: shared observer + structured error conversion."""

    def __init__(self, observer: ScreenObserver):
        self.observer = observer

    def _call(self, operation, *args: Any, **kwargs: Any) -> Any:
        try:
            return operation(*args, **kwargs)
        except ScreenException as e:
            raise ToolExecutionError(
                e.error.message,
                tool=self.name,
                details={"screen_error": e.error.to_dict()},
            ) from e


class ScreenObserveTool(_ScreenTool):
    """Observe the screen (desktop or browser page)."""

    name = "screen_observe"
    description = (
        "Observe the current screen and return structured visual "
        "information: screen dimensions, visible UI elements with "
        "regions and confidence scores (DOM elements when a "
        "browser page is observed, pixel-detected elements as "
        "fallback), and whether the screen changed. Returns no "
        "raw image data."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "origin": {"type": "string", "enum": ["desktop", "browser"]},
            "output_dir": {"type": "string"},
            "filename": {"type": "string"},
        },
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        observation = self._call(
            self.observer.observe,
            origin=arguments.get("origin", "desktop"),
            save_dir=arguments.get("output_dir"),
            filename=arguments.get("filename"),
        )
        result = observation.to_dict()
        result["observation"] = observation.summary()
        return result


class ScreenFindElementsTool(_ScreenTool):
    """Find candidate UI elements from the latest observation."""

    name = "screen_find_elements"
    description = (
        "List UI elements from the most recent screen observation "
        "(observing first when needed), filtered by minimum "
        "confidence, kind or source (dom/visual). Each element "
        "has a ref, region and confidence score."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "min_confidence": {"type": "number"},
            "kind": {"type": "string"},
            "source": {"type": "string", "enum": ["dom", "visual"]},
        },
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        elements = self._call(
            self.observer.find_elements,
            min_confidence=float(arguments.get("min_confidence", 0.0)),
            kind=arguments.get("kind"),
            source=arguments.get("source"),
        )
        return {
            "elements": [e.to_dict() for e in elements],
            "count": len(elements),
        }


class ScreenAssessActionTool(_ScreenTool):
    """Judge whether acting on a visual element may proceed."""

    name = "screen_assess_action"
    description = (
        "Assess whether an action on a visual element (by ref "
        "from screen_observe/screen_find_elements) may proceed "
        "automatically: high confidence = ok, medium = must be "
        "verified, low = requires human approval. Low-confidence "
        "elements are never acted on automatically."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "ref": {"type": "string"},
        },
        "required": ["ref"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        assessment = self._call(
            self.observer.assess_action, arguments["ref"]
        )
        return assessment.to_dict()


def create_screen_tools(observer: ScreenObserver) -> list[Tool]:
    """Create the screen Tools bound to *observer*."""
    return [
        ScreenObserveTool(observer),
        ScreenFindElementsTool(observer),
        ScreenAssessActionTool(observer),
    ]


def register_screen_tools(registry, observer: ScreenObserver) -> ScreenObserver:
    """Register all screen Tools on *registry*; return *observer*."""
    for tool in create_screen_tools(observer):
        if not registry.has(tool.name):
            registry.register(tool)
    return observer

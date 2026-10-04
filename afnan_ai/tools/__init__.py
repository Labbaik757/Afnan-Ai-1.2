"""Tools for Afnan AI.

Capabilities are :class:`Tool` objects (name, description, input
schema, execute) registered in a central :class:`ToolRegistry`.
The agent, an LLM or any other component runs them through the
registry and gets structured results/errors back.
"""

from afnan_ai.tools.base import (
    FunctionTool,
    Tool,
    ToolError,
    ToolErrorCode,
    ToolException,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolResult,
    ToolValidationError,
)
from afnan_ai.tools.builtin import (
    OpenApplicationTool,
    OpenUrlTool,
    SearchGoogleTool,
    SearchYouTubeTool,
    TakeScreenshotTool,
    create_default_registry,
)
from afnan_ai.tools.registry import ToolRegistry

__all__ = [
    "FunctionTool",
    "OpenApplicationTool",
    "OpenUrlTool",
    "SearchGoogleTool",
    "SearchYouTubeTool",
    "TakeScreenshotTool",
    "Tool",
    "ToolError",
    "ToolErrorCode",
    "ToolException",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolRegistrationError",
    "ToolRegistry",
    "ToolResult",
    "ToolValidationError",
    "create_default_registry",
]

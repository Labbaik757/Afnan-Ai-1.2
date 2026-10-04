"""Afnan AI — cross-platform voice assistant package."""

from afnan_ai.llm import LLMProvider, OllamaProvider
from afnan_ai.state import AgentState, Observation, StepRecord, TaskStatus, ToolResult

__all__ = [
    "__version__",
    "AgentState",
    "LLMProvider",
    "Observation",
    "OllamaProvider",
    "StepRecord",
    "TaskStatus",
    "ToolResult",
]
__version__ = "1.2"

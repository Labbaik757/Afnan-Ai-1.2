"""Afnan AI — cross-platform voice assistant package."""

from afnan_ai.executor import ExecutionReport, Executor
from afnan_ai.llm import LLMProvider, OllamaProvider
from afnan_ai.planner import Planner, TaskPlan
from afnan_ai.state import AgentState, Observation, StepRecord, TaskStatus, ToolResult
from afnan_ai.tools import Tool, ToolRegistry
from afnan_ai.verifier import VerificationResult, VerificationStatus, Verifier

__all__ = [
    "__version__",
    "AgentState",
    "ExecutionReport",
    "Executor",
    "LLMProvider",
    "Observation",
    "OllamaProvider",
    "Planner",
    "StepRecord",
    "TaskPlan",
    "TaskStatus",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "VerificationResult",
    "VerificationStatus",
    "Verifier",
]
__version__ = "1.2"

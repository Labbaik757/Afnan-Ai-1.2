"""Planner — builds a structured TaskPlan, never executes tools.

The Planner looks at three things:

* the user's **goal**,
* the current :class:`~afnan_ai.state.AgentState` (what already
  happened, what failed, what was observed), and
* the **available tools** (from the central ToolRegistry),

and asks the configured :class:`~afnan_ai.llm.LLMProvider` to
produce a plan.  It only produces the plan — executing its steps
is somebody else's job, later.  No tool is ever called from here;
the tools are consulted only for their names, descriptions and
input schemas, and to validate the arguments the LLM proposed.

Strict output validation
------------------------
The LLM is asked to reply with one JSON object and nothing else::

    {"goal": "...", "steps": [
        {"step_id": "step_1", "description": "...",
         "tool_name": "open_url",
         "arguments": {"url": "https://..."},
         "expected_result": "..."}
    ]}

That output is validated strictly before a TaskPlan is returned:

* it must be valid JSON (one optional ```json fence is stripped)
* the top level must be an object with only ``goal``/``steps``
* ``steps`` must be a non-empty list of objects, each with exactly
  ``step_id``, ``description``, ``tool_name``, ``arguments`` and
  ``expected_result``
* every field must have the right type and be non-empty where
  required, ``step_id`` values must be unique
* ``tool_name`` must be one of the available tools, and
  ``arguments`` must satisfy that tool's input schema (the tool
  object's own validation is reused)

Any violation — and any LLM failure — is raised as a structured
:class:`PlanningError` with a machine-readable code.  The Planner
never returns a half-valid plan and never crashes with a bare
``KeyError``/``TypeError`` on bad LLM output.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable

from afnan_ai.llm.base import (
    LLMConnectionError,
    LLMError,
    LLMInvalidResponseError,
    LLMProvider,
    LLMUnavailableError,
)
from afnan_ai.state import AgentState
from afnan_ai.tools.base import Tool, ToolException
from afnan_ai.tools.registry import ToolRegistry


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PlannerErrorCode(str, Enum):
    EMPTY_GOAL = "empty_goal"
    NO_TOOLS_AVAILABLE = "no_tools_available"
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_CONNECTION_FAILED = "llm_connection_failed"
    LLM_INVALID_RESPONSE = "llm_invalid_response"
    INVALID_LLM_OUTPUT = "invalid_llm_output"
    INVALID_PLAN = "invalid_plan"
    PLANNING_FAILED = "planning_failed"


@dataclass
class PlannerErrorRecord:
    """Serializable form of a planning failure."""

    code: PlannerErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": dict(self.details),
        }


class PlanningError(Exception):
    """Structured planning failure — carries a PlannerErrorRecord."""

    def __init__(
        self,
        message: str,
        *,
        code: PlannerErrorCode = PlannerErrorCode.PLANNING_FAILED,
        details: dict[str, Any] | None = None,
    ):
        self.error = PlannerErrorRecord(
            code=code, message=message, details=dict(details or {})
        )
        super().__init__(message)

    @property
    def code(self) -> PlannerErrorCode:
        return self.error.code

    def to_dict(self) -> dict[str, Any]:
        return self.error.to_dict()


# ----------------------------------------------------------------------
# TaskPlan
# ----------------------------------------------------------------------

_STEP_KEYS = {"step_id", "description", "tool_name", "arguments", "expected_result"}


@dataclass
class PlanStep:
    """One step of a TaskPlan."""

    step_id: str
    description: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    expected_result: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "expected_result": self.expected_result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanStep":
        if not isinstance(data, dict):
            raise PlanningError(
                "Plan step must be an object",
                code=PlannerErrorCode.INVALID_PLAN,
                details={"received": type(data).__name__},
            )
        unknown = set(data) - _STEP_KEYS
        missing = _STEP_KEYS - set(data)
        if unknown or missing:
            raise PlanningError(
                "Plan step has wrong fields "
                f"(missing={sorted(missing)}, unknown={sorted(unknown)})",
                code=PlannerErrorCode.INVALID_PLAN,
                details={"missing": sorted(missing), "unknown": sorted(unknown)},
            )
        return cls(
            step_id=data["step_id"],
            description=data["description"],
            tool_name=data["tool_name"],
            arguments=dict(data["arguments"] or {}),
            expected_result=data["expected_result"],
        )


@dataclass
class TaskPlan:
    """A structured, serializable plan for one goal."""

    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str | None = None
    created_at: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "task_id": self.task_id,
            "steps": [step.to_dict() for step in self.steps],
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskPlan":
        if not isinstance(data, dict):
            raise PlanningError(
                "TaskPlan must be an object",
                code=PlannerErrorCode.INVALID_PLAN,
            )
        return cls(
            goal=data["goal"],
            steps=[PlanStep.from_dict(step) for step in data.get("steps", [])],
            plan_id=data.get("plan_id") or uuid.uuid4().hex,
            task_id=data.get("task_id"),
            created_at=data.get("created_at") or _now_iso(),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "TaskPlan":
        try:
            data = json.loads(text)
        except Exception as e:
            raise PlanningError(
                f"TaskPlan JSON is invalid: {e}",
                code=PlannerErrorCode.INVALID_PLAN,
            ) from e
        return cls.from_dict(data)


# ----------------------------------------------------------------------
# Planner
# ----------------------------------------------------------------------


class Planner:
    """Generate TaskPlans from a goal, AgentState and available tools.

    The Planner only plans.  It holds references to the LLM provider
    and the tool registry, but it never calls ``registry.execute``
    or a tool's ``execute``/``run`` — tools are used read-only for
    their definitions and argument validation.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry | Iterable[Tool] | None = None,
    ):
        self.llm = llm_provider
        self.tools = self._as_registry(tool_registry)

    @staticmethod
    def _as_registry(tools) -> ToolRegistry | None:
        if tools is None or isinstance(tools, ToolRegistry):
            return tools
        return ToolRegistry(list(tools))

    # -- public API ---------------------------------------------------------
    def plan(
        self,
        goal: str,
        *,
        state: AgentState | None = None,
        tools: ToolRegistry | Iterable[Tool] | None = None,
        recovery_context: str | None = None,
    ) -> TaskPlan:
        """Return a validated TaskPlan for *goal*.

        Raises PlanningError (structured) on an empty goal, no
        available tools, an LLM failure, or invalid LLM output.

        ``recovery_context`` is optional extra guidance (used by
        the Recovery mechanism) describing a previous failed or
        uncertain attempt; it is added to the prompt so the new
        plan can avoid repeating the action that already failed.
        """
        if not goal or not str(goal).strip():
            raise PlanningError(
                "Cannot plan for an empty goal",
                code=PlannerErrorCode.EMPTY_GOAL,
            )
        goal = str(goal).strip()

        registry = self._as_registry(tools) or self.tools
        catalog = self._catalog(registry)
        if not catalog:
            raise PlanningError(
                "Cannot plan: no tools are available to plan with",
                code=PlannerErrorCode.NO_TOOLS_AVAILABLE,
            )

        prompt = self.build_prompt(
            goal,
            state=state,
            catalog=catalog,
            recovery_context=recovery_context,
        )
        raw = self._ask_llm(prompt)
        return self.parse_plan(raw, goal=goal, state=state, catalog=catalog)

    # Backwards/alternate naming
    create_plan = plan
    generate_plan = plan

    # -- prompt --------------------------------------------------------------
    def build_prompt(
        self,
        goal: str,
        *,
        state: AgentState | None = None,
        tools: ToolRegistry | Iterable[Tool] | None = None,
        catalog: dict[str, dict[str, Any]] | None = None,
        recovery_context: str | None = None,
    ) -> str:
        catalog = catalog or self._catalog(
            self._as_registry(tools) or self.tools
        )
        tool_lines = []
        for name in sorted(catalog):
            entry = catalog[name]
            schema = json.dumps(entry["input_schema"], ensure_ascii=False)
            tool_lines.append(
                f"- {name}: {entry['description']} "
                f"(arguments schema: {schema})"
            )
        tools_text = "\n".join(tool_lines) if tool_lines else "(no tools)"

        state_text = "No previous task state."
        if state is not None:
            summary = {
                "goal": state.goal,
                "status": state.status.value,
                "current_step": state.current_step,
                "completed_steps": [s.name for s in state.completed_steps],
                "failed_steps": [
                    {"step": s.name, "error": s.error} for s in state.failed_steps
                ],
                "observations": [o.text for o in state.observations[-5:]],
                "recent_tool_results": [
                    {"tool": t.tool, "success": t.success, "error": t.error}
                    for t in state.tool_results[-5:]
                ],
            }
            state_text = json.dumps(summary, ensure_ascii=False)

        recovery_text = ""
        if recovery_context:
            recovery_text = (
                "\nRecovery context (a previous attempt did not "
                "succeed):\n"
                f"{recovery_context}\n"
                "Your new plan must still achieve the user's goal, "
                "but it must NOT repeat the failed action with the "
                "same tool and the same arguments. Choose a "
                "different tool, different arguments, or a "
                "different approach, and build on the steps that "
                "already completed.\n"
            )

        return (
            "You are the planning component of Afnan AI. "
            "Create a plan only — do not execute anything and do not "
            "claim that any action has been taken.\n\n"
            f"User goal: {goal}\n\n"
            f"Current agent state:\n{state_text}\n"
            f"{recovery_text}\n"
            f"Available tools (use only these tool_name values):\n"
            f"{tools_text}\n\n"
            "Reply with exactly one JSON object and no other text, "
            "in this exact shape:\n"
            '{"goal": "<the goal>", "steps": ['
            '{"step_id": "step_1", "description": "<what this step does>", '
            '"tool_name": "<one of the available tools>", '
            '"arguments": {<arguments matching that tool\'s schema>}, '
            '"expected_result": "<what a successful step produces>"}'
            "]}\n"
            "Every step must have step_id, description, tool_name, "
            "arguments and expected_result. step_id values must be "
            "unique. Arguments must satisfy the tool's schema."
        )

    # -- LLM call ---------------------------------------------------------------
    def _ask_llm(self, prompt: str) -> str:
        try:
            reply = self.llm.generate(prompt)
        except LLMUnavailableError as e:
            raise PlanningError(
                str(e) or "LLM provider is not available",
                code=PlannerErrorCode.LLM_UNAVAILABLE,
            ) from e
        except LLMConnectionError as e:
            raise PlanningError(
                str(e) or "LLM provider could not be reached",
                code=PlannerErrorCode.LLM_CONNECTION_FAILED,
            ) from e
        except LLMInvalidResponseError as e:
            raise PlanningError(
                str(e) or "LLM provider returned an invalid response",
                code=PlannerErrorCode.LLM_INVALID_RESPONSE,
            ) from e
        except LLMError as e:
            raise PlanningError(
                str(e) or "LLM provider failed",
                code=PlannerErrorCode.PLANNING_FAILED,
            ) from e
        except Exception as e:
            # A provider that breaks its contract must not crash us
            raise PlanningError(
                f"LLM provider failed while planning: {e}",
                code=PlannerErrorCode.PLANNING_FAILED,
                details={"exception": type(e).__name__},
            ) from e
        if not isinstance(reply, str) or not reply.strip():
            raise PlanningError(
                "LLM provider returned an empty response",
                code=PlannerErrorCode.LLM_INVALID_RESPONSE,
            )
        return reply

    # -- parsing & strict validation ---------------------------------------------
    def parse_plan(
        self,
        raw: str,
        *,
        goal: str,
        state: AgentState | None = None,
        tools: ToolRegistry | Iterable[Tool] | None = None,
        catalog: dict[str, dict[str, Any]] | None = None,
    ) -> TaskPlan:
        """Validate raw LLM output into a TaskPlan (no execution)."""
        catalog = catalog or self._catalog(
            self._as_registry(tools) or self.tools
        )
        data = self._parse_json(raw)
        steps = self._validate_steps(data, catalog)
        return TaskPlan(
            goal=goal,
            steps=steps,
            task_id=state.task_id if state is not None else None,
            metadata={
                "source": "planner",
                "tools_available": sorted(catalog),
            },
        )

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        text = (raw or "").strip()
        # LLMs commonly wrap JSON in a fenced block — strip one fence
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2:
                lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
        try:
            data = json.loads(text)
        except Exception as e:
            raise PlanningError(
                f"LLM output is not valid JSON: {e}",
                code=PlannerErrorCode.INVALID_LLM_OUTPUT,
                details={"output_preview": text[:200]},
            ) from e
        if not isinstance(data, dict):
            raise PlanningError(
                "LLM output must be a JSON object with 'goal' and 'steps'",
                code=PlannerErrorCode.INVALID_LLM_OUTPUT,
                details={"received": type(data).__name__},
            )
        unknown = set(data) - {"goal", "steps"}
        if unknown:
            raise PlanningError(
                f"LLM output has unknown top-level field(s): {sorted(unknown)}",
                code=PlannerErrorCode.INVALID_LLM_OUTPUT,
                details={"unknown": sorted(unknown)},
            )
        if "steps" not in data:
            raise PlanningError(
                "LLM output is missing 'steps'",
                code=PlannerErrorCode.INVALID_LLM_OUTPUT,
            )
        if "goal" in data and not isinstance(data["goal"], str):
            raise PlanningError(
                "LLM output 'goal' must be a string",
                code=PlannerErrorCode.INVALID_LLM_OUTPUT,
            )
        return data

    @staticmethod
    def _validate_steps(
        data: dict[str, Any], catalog: dict[str, dict[str, Any]]
    ) -> list[PlanStep]:
        raw_steps = data["steps"]
        if not isinstance(raw_steps, list) or not raw_steps:
            raise PlanningError(
                "Plan 'steps' must be a non-empty list",
                code=PlannerErrorCode.INVALID_LLM_OUTPUT,
                details={"received": type(raw_steps).__name__},
            )

        steps: list[PlanStep] = []
        seen_ids: set[str] = set()
        for index, raw_step in enumerate(raw_steps):
            where = f"steps[{index}]"
            if not isinstance(raw_step, dict):
                raise PlanningError(
                    f"{where} must be an object",
                    code=PlannerErrorCode.INVALID_LLM_OUTPUT,
                    details={"step_index": index},
                )
            unknown = set(raw_step) - _STEP_KEYS
            missing = _STEP_KEYS - set(raw_step)
            if unknown or missing:
                raise PlanningError(
                    f"{where} has wrong fields "
                    f"(missing={sorted(missing)}, unknown={sorted(unknown)})",
                    code=PlannerErrorCode.INVALID_LLM_OUTPUT,
                    details={
                        "step_index": index,
                        "missing": sorted(missing),
                        "unknown": sorted(unknown),
                    },
                )

            step_id = raw_step["step_id"]
            description = raw_step["description"]
            tool_name = raw_step["tool_name"]
            arguments = raw_step["arguments"]
            expected_result = raw_step["expected_result"]

            for field_name, value in (
                ("step_id", step_id),
                ("description", description),
                ("tool_name", tool_name),
                ("expected_result", expected_result),
            ):
                if not isinstance(value, str) or not value.strip():
                    raise PlanningError(
                        f"{where}.{field_name} must be a non-empty string",
                        code=PlannerErrorCode.INVALID_PLAN,
                        details={"step_index": index, "field": field_name},
                    )
            if not isinstance(arguments, dict):
                raise PlanningError(
                    f"{where}.arguments must be an object",
                    code=PlannerErrorCode.INVALID_PLAN,
                    details={"step_index": index},
                )
            if step_id in seen_ids:
                raise PlanningError(
                    f"Duplicate step_id {step_id!r} in plan",
                    code=PlannerErrorCode.INVALID_PLAN,
                    details={"step_id": step_id},
                )
            seen_ids.add(step_id)

            entry = catalog.get(tool_name)
            if entry is None:
                raise PlanningError(
                    f"{where} selects unknown tool {tool_name!r}",
                    code=PlannerErrorCode.INVALID_PLAN,
                    details={
                        "step_id": step_id,
                        "tool_name": tool_name,
                        "available": sorted(catalog),
                    },
                )

            validated_args = Planner._validate_arguments(
                entry, tool_name, arguments, step_id
            )
            steps.append(
                PlanStep(
                    step_id=step_id.strip(),
                    description=description.strip(),
                    tool_name=tool_name,
                    arguments=validated_args,
                    expected_result=expected_result.strip(),
                )
            )
        return steps

    @staticmethod
    def _validate_arguments(
        entry: dict[str, Any],
        tool_name: str,
        arguments: dict[str, Any],
        step_id: str,
    ) -> dict[str, Any]:
        tool = entry.get("tool")
        if tool is not None:
            try:
                return tool.validate_arguments(arguments)
            except ToolException as e:
                raise PlanningError(
                    f"Invalid arguments for tool {tool_name!r} "
                    f"in step {step_id!r}: {e}",
                    code=PlannerErrorCode.INVALID_PLAN,
                    details={
                        "step_id": step_id,
                        "tool_name": tool_name,
                        "tool_error": e.error.to_dict(),
                    },
                ) from e

        # Definition-only catalog: enforce the schema's required list
        schema = entry.get("input_schema") or {}
        required = list(schema.get("required", []))
        missing = [key for key in required if arguments.get(key) is None]
        if missing:
            raise PlanningError(
                f"Step {step_id!r} is missing argument(s) {missing} "
                f"for tool {tool_name!r}",
                code=PlannerErrorCode.INVALID_PLAN,
                details={"step_id": step_id, "missing": missing},
            )
        return dict(arguments)

    # -- tool catalog (read-only) -------------------------------------------------
    @staticmethod
    def _catalog(
        registry: ToolRegistry | None,
    ) -> dict[str, dict[str, Any]]:
        """name -> {description, input_schema, tool} — never executes."""
        catalog: dict[str, dict[str, Any]] = {}
        if registry is None:
            return catalog
        for tool in registry.list_tools():
            definition = tool.definition()
            catalog[tool.name] = {
                "description": definition["description"],
                "input_schema": definition["input_schema"],
                "tool": tool,
            }
        return catalog

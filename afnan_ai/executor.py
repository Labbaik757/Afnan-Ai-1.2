"""Executor — runs a TaskPlan step by step through the ToolRegistry.

The Planner produces a structured :class:`~afnan_ai.planner.TaskPlan`;
the Executor is the component that carries it out.  For every step,
in order, it:

1. checks the named tool **exists** in the ToolRegistry,
2. **validates the step's arguments** against that tool's input
   schema (the tool's own ``validate_arguments`` is reused),
3. executes the tool **through the registry only**, and
4. records the outcome in :class:`~afnan_ai.state.AgentState`
   (tool result + completed/failed step), whether it worked or not.

Safety rules
------------
* **No arbitrary code execution.**  A step can only name a tool
  that is already registered; its arguments are treated strictly as
  data.  Nothing in a plan — tool name or argument — is ever turned
  into Python code, a shell command or an import.  This module
  contains no dynamic code-evaluation or process-spawning calls,
  and ``tools/check_compatibility.py`` AST-checks that.
* **A failed action is never silently successful.**  An unknown
  tool, invalid arguments, a tool raising, or a tool reporting
  failure all produce ``success=False`` on that step's result, mark
  the whole execution ``failed``, fail the step (and the task) in
  AgentState, and — by default — stop the plan so later steps are
  reported as *skipped* instead of pretending they ran.

Usage::

    executor = Executor(agent.tools, state=agent.state)
    report = executor.execute_plan(plan)
    report.success          # True only if every step really worked
    report.step_results[0].output
    agent.state.status      # completed / failed, steps recorded
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from afnan_ai.planner import PlanStep, TaskPlan
from afnan_ai.state import AgentState, TaskStatus
from afnan_ai.tools.base import (
    ToolError,
    ToolErrorCode,
    ToolException,
    ToolValidationError,
)
from afnan_ai.tools.registry import ToolRegistry


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------
# Structured executor-level errors (plan/state misuse, not step failure)
# ----------------------------------------------------------------------


class ExecutorErrorCode(str, Enum):
    INVALID_PLAN = "invalid_plan"
    EMPTY_PLAN = "empty_plan"
    INVALID_STEP = "invalid_step"


@dataclass
class ExecutorErrorRecord:
    code: ExecutorErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": dict(self.details),
        }


class ExecutionError(Exception):
    """Raised when the Executor is handed something it cannot run
    at all (not a TaskPlan, an empty plan, ...).  Step-level
    failures never raise — they are reported on the ExecutionReport
    and recorded in AgentState instead."""

    def __init__(
        self,
        message: str,
        *,
        code: ExecutorErrorCode = ExecutorErrorCode.INVALID_PLAN,
        details: dict[str, Any] | None = None,
    ):
        self.error = ExecutorErrorRecord(
            code=code, message=message, details=dict(details or {})
        )
        super().__init__(message)

    @property
    def code(self) -> ExecutorErrorCode:
        return self.error.code

    def to_dict(self) -> dict[str, Any]:
        return self.error.to_dict()


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------


@dataclass
class StepExecutionResult:
    """Outcome of one plan step — success is never faked."""

    step_id: str
    description: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    expected_result: str = ""
    success: bool = False
    skipped: bool = False
    output: Any = None
    error: dict[str, Any] | None = None
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def status(self) -> str:
        if self.skipped:
            return "skipped"
        return "completed" if self.success else "failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "expected_result": self.expected_result,
            "status": self.status,
            "success": bool(self.success),
            "skipped": bool(self.skipped),
            "output": self.output,
            "error": dict(self.error) if self.error else None,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass
class ExecutionReport:
    """Outcome of executing a whole TaskPlan."""

    plan_id: str
    goal: str
    task_id: str | None = None
    status: str = "failed"  # "completed" only when every step worked
    step_results: list[StepExecutionResult] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def success(self) -> bool:
        return self.status == "completed" and all(
            r.success for r in self.step_results
        )

    @property
    def succeeded(self) -> list[StepExecutionResult]:
        return [r for r in self.step_results if r.success]

    @property
    def failed(self) -> list[StepExecutionResult]:
        return [r for r in self.step_results if not r.success and not r.skipped]

    @property
    def skipped_steps(self) -> list[StepExecutionResult]:
        return [r for r in self.step_results if r.skipped]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "task_id": self.task_id,
            "status": self.status,
            "success": self.success,
            "step_results": [r.to_dict() for r in self.step_results],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    def summary(self) -> str:
        total = len(self.step_results)
        return (
            f"Plan {self.plan_id} for {self.goal!r}: {self.status} "
            f"({len(self.succeeded)}/{total} step(s) succeeded, "
            f"{len(self.failed)} failed, {len(self.skipped_steps)} skipped)"
        )


# ----------------------------------------------------------------------
# Executor
# ----------------------------------------------------------------------


class Executor:
    """Execute TaskPlans through a ToolRegistry, recording AgentState.

    ``stop_on_failure`` (default True) halts the plan at the first
    failed step — later steps are reported as skipped, never as
    successful.  Set it to False to attempt every step anyway; the
    overall report is still ``failed`` if any step failed.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        state: AgentState | None = None,
        *,
        stop_on_failure: bool = True,
    ):
        self.registry = tool_registry
        # Convenience alias, matching the agent's naming
        self.tools = tool_registry
        self.state: AgentState | None = state
        self.stop_on_failure = bool(stop_on_failure)

    # -- public API ---------------------------------------------------
    def execute_plan(
        self,
        plan: TaskPlan,
        state: AgentState | None = None,
    ) -> ExecutionReport:
        """Execute *plan* step by step and return an ExecutionReport.

        Raises ExecutionError only when there is no runnable plan
        at all (wrong type, no steps).  Individual step failures
        are returned on the report, never hidden and never raised
        as bare tool exceptions.
        """
        steps = self._validated_steps(plan)
        effective_state = self._resolve_state(plan, state)
        self.state = effective_state

        report = ExecutionReport(
            plan_id=plan.plan_id,
            goal=plan.goal,
            task_id=effective_state.task_id,
            started_at=_now_iso(),
        )
        self._begin_task(effective_state, plan)

        halted = False
        for step in steps:
            if halted:
                report.step_results.append(self._skipped_result(step))
                continue
            result = self._execute_step(step, effective_state, plan)
            report.step_results.append(result)
            if not result.success and self.stop_on_failure:
                halted = True

        skipped = report.skipped_steps
        if skipped:
            effective_state.add_observation(
                f"Executor skipped {len(skipped)} step(s) after a failure "
                f"in plan {plan.plan_id}",
                source="executor",
            )

        if report.failed or skipped:
            first_error = (
                report.failed[0].error.get("message")
                if report.failed and report.failed[0].error
                else "plan execution failed"
            )
            report.status = "failed"
            effective_state.fail_task(str(first_error))
        else:
            report.status = "completed"
            effective_state.complete_task(result=report.summary())

        report.finished_at = _now_iso()
        effective_state.add_observation(
            f"Executor finished: {report.summary()}", source="executor"
        )
        return report

    # Alias matching the naming used elsewhere (run/execute a plan)
    run_plan = execute_plan

    # -- plan / state preparation --------------------------------------
    @staticmethod
    def _validated_steps(plan: TaskPlan) -> list[PlanStep]:
        if not isinstance(plan, TaskPlan):
            raise ExecutionError(
                f"Executor can only execute a TaskPlan, "
                f"got {type(plan).__name__}",
                code=ExecutorErrorCode.INVALID_PLAN,
                details={"received": type(plan).__name__},
            )
        if not plan.goal or not str(plan.goal).strip():
            raise ExecutionError(
                "Cannot execute a plan with an empty goal",
                code=ExecutorErrorCode.INVALID_PLAN,
            )
        if not plan.steps:
            raise ExecutionError(
                "Cannot execute a plan with no steps",
                code=ExecutorErrorCode.EMPTY_PLAN,
                details={"plan_id": plan.plan_id},
            )
        for index, step in enumerate(plan.steps):
            if not isinstance(step, PlanStep):
                raise ExecutionError(
                    f"Plan step {index} is not a PlanStep",
                    code=ExecutorErrorCode.INVALID_STEP,
                    details={"step_index": index},
                )
            if not step.step_id or not step.tool_name:
                raise ExecutionError(
                    f"Plan step {index} needs a step_id and a tool_name",
                    code=ExecutorErrorCode.INVALID_STEP,
                    details={"step_index": index},
                )
        return list(plan.steps)

    def _resolve_state(
        self, plan: TaskPlan, state: AgentState | None
    ) -> AgentState:
        if state is not None:
            return state
        if self.state is not None and not self.state.is_terminal:
            return self.state
        if plan.task_id:
            return AgentState.create(plan.goal, task_id=plan.task_id)
        return AgentState.create(plan.goal)

    @staticmethod
    def _begin_task(state: AgentState, plan: TaskPlan) -> None:
        # A state handed over mid-task (e.g. the agent's command
        # state) is kept; a fresh/terminal one starts a new cycle.
        if state.is_terminal or state.status == TaskStatus.PENDING:
            state.start_task()
        state.add_observation(
            f"Executor starting plan {plan.plan_id} for goal: {plan.goal} "
            f"({len(plan.steps)} step(s))",
            source="executor",
        )
        state.metadata["last_execution_plan"] = plan.plan_id

    # -- one step ---------------------------------------------------------
    def _execute_step(
        self,
        step: PlanStep,
        state: AgentState,
        plan: TaskPlan,
    ) -> StepExecutionResult:
        result = StepExecutionResult(
            step_id=step.step_id,
            description=step.description,
            tool_name=step.tool_name,
            arguments=dict(step.arguments) if isinstance(step.arguments, dict) else {},
            expected_result=step.expected_result,
            started_at=_now_iso(),
        )
        state.start_step(
            step.step_id,
            metadata={
                "description": step.description,
                "tool_name": step.tool_name,
                "arguments": result.arguments,
                "expected_result": step.expected_result,
                "plan_id": plan.plan_id,
            },
        )

        error = self._preflight_error(step, result.arguments)
        if error is not None:
            return self._record_failure(result, state, error)

        tool_result = self.registry.execute(step.tool_name, result.arguments)
        if tool_result.success:
            result.success = True
            result.output = tool_result.output
            result.finished_at = _now_iso()
            state.add_tool_result(
                step.tool_name,
                success=True,
                output=tool_result.output,
                metadata={"step_id": step.step_id, "plan_id": plan.plan_id},
            )
            state.complete_step(step.step_id, result=tool_result.output)
            return result

        error = tool_result.error or ToolError(
            code=ToolErrorCode.EXECUTION_FAILED,
            message=f"Tool {step.tool_name!r} failed without an error record",
            tool=step.tool_name,
        )
        return self._record_failure(result, state, error)

    def _preflight_error(
        self, step: PlanStep, arguments: dict[str, Any]
    ) -> ToolError | None:
        """Existence + argument validation, before anything runs.

        Returns a structured ToolError describing the problem, or
        None when the step is safe to execute through the registry.
        """
        if not isinstance(step.arguments, dict):
            return ToolError(
                code=ToolErrorCode.INVALID_ARGUMENTS,
                message=(
                    f"Arguments for step {step.step_id!r} must be an "
                    f"object, got {type(step.arguments).__name__}"
                ),
                tool=step.tool_name or None,
                details={"step_id": step.step_id},
            )

        tool = self.registry.get_or_none(step.tool_name)
        if tool is None:
            return ToolError(
                code=ToolErrorCode.TOOL_NOT_FOUND,
                message=(
                    f"Unknown tool {step.tool_name!r} in step "
                    f"{step.step_id!r}. Available tools: "
                    f"{', '.join(self.registry.names()) or '(none)'}"
                ),
                tool=step.tool_name,
                details={
                    "step_id": step.step_id,
                    "available": self.registry.names(),
                },
            )

        try:
            tool.validate_arguments(arguments)
        except ToolValidationError as e:
            return e.error
        except ToolException as e:  # defensive: keep the structure
            return e.error
        return None

    @staticmethod
    def _record_failure(
        result: StepExecutionResult,
        state: AgentState,
        error: ToolError,
    ) -> StepExecutionResult:
        result.success = False
        result.error = error.to_dict()
        result.finished_at = _now_iso()
        state.add_tool_result(
            result.tool_name,
            success=False,
            error=error.message,
            metadata={"step_id": result.step_id},
        )
        state.fail_step(result.step_id, error.message)
        return result

    @staticmethod
    def _skipped_result(step: PlanStep) -> StepExecutionResult:
        return StepExecutionResult(
            step_id=step.step_id,
            description=step.description,
            tool_name=step.tool_name,
            arguments=dict(step.arguments) if isinstance(step.arguments, dict) else {},
            expected_result=step.expected_result,
            success=False,
            skipped=True,
            error={
                "code": "skipped",
                "message": (
                    f"Step {step.step_id!r} was skipped because an "
                    f"earlier step in the plan failed"
                ),
                "tool": step.tool_name,
                "details": {},
            },
            finished_at=_now_iso(),
        )

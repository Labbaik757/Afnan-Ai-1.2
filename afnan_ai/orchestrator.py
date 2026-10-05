"""Agent — the central orchestrator for Afnan's task lifecycle.

Until now the pieces existed side by side: ``AgentState`` tracked a
task, the ``Planner`` turned a goal into a ``TaskPlan``, the
``Executor`` ran plan steps through the ``ToolRegistry``, and the
``Verifier`` judged each step's actual result against its
``expected_result``.  The :class:`Agent` in this module is the one
component that connects them and manages a complete task from
start to finish:

    user goal received
      → AgentState created / updated
      → Planner generates a TaskPlan
      → Executor executes the *next* step
      → Verifier verifies that step's result
      → AgentState updated
      → next step … or task completion / failure

Safety & lifecycle rules
------------------------
* **Maximum iteration limit is mandatory.**  Every step execution
  counts as one iteration; once ``max_iterations`` is reached the
  agent stops, marks the task failed with a
  ``max_iterations_exceeded`` outcome and never loops forever —
  no matter how long the plan is.  The limit is validated at
  construction (it must be a positive integer) and can be
  overridden per run, but never removed.
* **Failures stop the task honestly.**  A planning failure, an
  execution failure or a failed verification ends the run with a
  matching status; remaining steps are reported as *skipped*,
  never as done.  An *uncertain* verification is recorded but does
  not block an otherwise successful execution (unless
  ``strict_verification=True``).
* **Recovery is bounded, never a blind retry.**  After a failed
  or uncertain step, the Recovery mechanism
  (:class:`~afnan_ai.recovery.RecoveryManager`) hands the failure
  reason, the current AgentState and the previous attempt to the
  Planner and asks for a *different* plan.  A recovery plan that
  would repeat an already-failed action is rejected, recovery is
  capped by ``max_recovery_attempts`` (default 2), every attempt
  is recorded in AgentState, and recovered steps still count
  against the same maximum-iteration limit.
* **No responsibility is duplicated.**  The Agent plans only via
  the Planner, executes only via the Executor (which alone talks
  to the ToolRegistry) and judges only via the Verifier.  It holds
  no tool, model or platform logic of its own.

The voice assistant (``AfnanAgent``) uses one of these as its
central orchestration layer (``agent.orchestrator`` /
``agent.run_task(goal)``) while its existing voice-command behaviour
stays exactly as it was.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _redacted_plan_dict(plan: TaskPlan) -> dict[str, Any]:
    """A plan as stored in AgentState: structure kept, step
    arguments passed through the redactor so a plan that types a
    password never writes the secret into the record."""
    data = plan.to_dict()
    for step in data.get("steps", []):
        if isinstance(step, dict) and "arguments" in step:
            step["arguments"] = redact_arguments(step["arguments"])
    return redact_value(data)


from afnan_ai.executor import (
    ExecutionReport,
    Executor,
    StepExecutionResult,
    sensitive_argument_keys,
)
from afnan_ai.log_config import get_logger
from afnan_ai.planner import PlanStep, Planner, PlanningError, TaskPlan
from afnan_ai.recovery import RecoveryError, RecoveryManager
from afnan_ai.redaction import redact_arguments, redact_value
from afnan_ai.state import AgentState, TaskStatus
from afnan_ai.verifier import (
    VerificationReport,
    VerificationResult,
    VerificationStatus,
    Verifier,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


logger = get_logger(__name__)


class OrchestrationStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    PLANNING_FAILED = "planning_failed"
    MAX_ITERATIONS_EXCEEDED = "max_iterations_exceeded"


@dataclass
class IterationRecord:
    """One loop iteration: a step executed and verified."""

    iteration: int
    step_id: str
    execution: StepExecutionResult
    verification: VerificationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "step_id": self.step_id,
            "execution": self.execution.to_dict(),
            "verification": (
                self.verification.to_dict() if self.verification else None
            ),
        }


@dataclass
class OrchestrationResult:
    """The outcome of one full orchestrated task run."""

    goal: str
    status: OrchestrationStatus
    state: AgentState
    plan: TaskPlan | None = None
    execution: ExecutionReport | None = None
    verification: VerificationReport | None = None
    iterations: int = 0
    max_iterations: int = 10
    error: dict[str, Any] | None = None
    records: list[IterationRecord] = field(default_factory=list)
    recovery_attempts: list[dict[str, Any]] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def success(self) -> bool:
        return self.status == OrchestrationStatus.COMPLETED

    @property
    def task_id(self) -> str:
        return self.state.task_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "status": self.status.value,
            "success": self.success,
            "task_id": self.task_id,
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "plan": self.plan.to_dict() if self.plan else None,
            "execution": self.execution.to_dict() if self.execution else None,
            "verification": (
                self.verification.to_dict() if self.verification else None
            ),
            "records": [r.to_dict() for r in self.records],
            "recovery_attempts": list(self.recovery_attempts),
            "error": self.error,
            "state": self.state.summary(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    def summary(self) -> str:
        return (
            f"Task {self.task_id} ({self.goal!r}): {self.status.value} "
            f"after {self.iterations}/{self.max_iterations} iteration(s)"
        )


class Agent:
    """Central orchestrator connecting AgentState, Planner,
    Executor and Verifier.

    ``max_iterations`` is required by design: it defaults to 10 and
    must be a positive integer, so an orchestrated task can never
    loop forever.
    """

    DEFAULT_MAX_ITERATIONS = 10

    def __init__(
        self,
        *,
        planner: Planner,
        executor: Executor,
        verifier: Verifier,
        state: AgentState | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        strict_verification: bool = False,
        recovery: RecoveryManager | None = None,
        max_recovery_attempts: int = RecoveryManager.DEFAULT_MAX_ATTEMPTS,
        recover_on_uncertain: bool = True,
        checkpointer: Any = None,
        checkpoint_extra_provider: Any = None,
        max_duration_s: float | None = None,
    ):
        if planner is None or executor is None or verifier is None:
            raise ValueError(
                "Agent needs a Planner, an Executor and a Verifier"
            )
        self.planner = planner
        self.executor = executor
        self.verifier = verifier
        self.state: AgentState | None = state
        self.max_iterations = self._validate_limit(max_iterations)
        self.strict_verification = bool(strict_verification)
        # Recovery: after a failed/uncertain step the Planner is
        # asked for a *different* plan (never a blind repeat).  The
        # manager plans only; execution/verification stay with the
        # Executor/Verifier, and every attempt is recorded in
        # AgentState.  max_recovery_attempts=0 disables recovery.
        self.recovery: RecoveryManager = recovery or RecoveryManager(
            planner, max_attempts=max_recovery_attempts
        )
        self.max_recovery_attempts = self.recovery.max_attempts
        self.recover_on_uncertain = bool(recover_on_uncertain)
        # Checkpointing: when a checkpointer is configured, task
        # snapshots (state + plan + iteration + provider extras
        # such as the browser session) are persisted after planning
        # and after every iteration, so an interrupted task can be
        # resumed from its last valid checkpoint instead of
        # starting over.  Checkpoint failures never fail the task.
        self.checkpointer = checkpointer
        self.checkpoint_extra_provider = checkpoint_extra_provider
        self._active_checkpoint: tuple | None = None
        # Optional wall-clock budget for a whole run (seconds);
        # None means steps/iterations alone bound the task.
        self.max_duration_s = self._validate_duration(max_duration_s)

    @staticmethod
    def _validate_limit(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"max_iterations must be a positive integer, got {value!r}"
            )
        return value

    @staticmethod
    def _validate_duration(value: float | None) -> float | None:
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
        ):
            raise ValueError(
                "max_duration_s must be a positive number of "
                f"seconds, got {value!r}"
            )
        return float(value)

    @classmethod
    def from_components(
        cls,
        llm_provider,
        tool_registry,
        *,
        state: AgentState | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        strict_verification: bool = False,
        max_recovery_attempts: int = RecoveryManager.DEFAULT_MAX_ATTEMPTS,
        recover_on_uncertain: bool = True,
        checkpointer: Any = None,
        checkpoint_extra_provider: Any = None,
        max_duration_s: float | None = None,
    ) -> "Agent":
        """Build an Agent (and its Planner/Executor/Verifier) from a
        raw LLM provider + ToolRegistry — handy outside AfnanAgent."""
        return cls(
            planner=Planner(llm_provider, tool_registry),
            executor=Executor(tool_registry),
            verifier=Verifier(),
            state=state,
            max_iterations=max_iterations,
            strict_verification=strict_verification,
            max_recovery_attempts=max_recovery_attempts,
            recover_on_uncertain=recover_on_uncertain,
            checkpointer=checkpointer,
            checkpoint_extra_provider=checkpoint_extra_provider,
            max_duration_s=max_duration_s,
        )

    # -- main entry point -------------------------------------------------
    def run(
        self,
        goal: str,
        *,
        state: AgentState | None = None,
        max_iterations: int | None = None,
        resume_from: dict[str, Any] | None = None,
        max_duration_s: float | None = None,
    ) -> OrchestrationResult:
        """Run one complete task lifecycle for *goal*.

        Never raises for planning or step failures — they come back
        on the OrchestrationResult with a matching status.  Raises
        ValueError only for an unusable goal/limit argument.

        With ``resume_from`` (a validated checkpoint dict), the
        task's AgentState is restored and only the steps that were
        not completed before the interruption run again.
        """
        if not goal or not str(goal).strip():
            raise ValueError("Agent.run needs a non-empty goal")
        goal = str(goal).strip()
        limit = (
            self._validate_limit(max_iterations)
            if max_iterations is not None
            else self.max_iterations
        )
        duration_budget = (
            self._validate_duration(max_duration_s)
            if max_duration_s is not None
            else self.max_duration_s
        )
        started_mono = time.monotonic()

        if resume_from is not None:
            task_state = AgentState.from_dict(resume_from["state"])
        else:
            task_state = self._prepare_state(goal, state)
        self.state = task_state
        result = OrchestrationResult(
            goal=goal,
            status=OrchestrationStatus.FAILED,  # until proven completed
            state=task_state,
            max_iterations=limit,
            started_at=_now_iso(),
        )
        if resume_from is not None:
            result.iterations = int(resume_from.get("iteration") or 0)
        task_state.add_observation(
            f"Agent received goal: {goal}", source="agent"
        )
        logger.info(
            "task %s started: goal=%r max_iterations=%d",
            task_state.task_id, goal, limit,
        )

        # 1) Plan ------------------------------------------------------------
        plan: TaskPlan | None = None
        if resume_from is not None:
            plan, all_done = self._resume_plan(
                resume_from, task_state, result
            )
            if all_done:
                result.plan = (
                    TaskPlan.from_dict(resume_from["plan"])
                    if resume_from.get("plan")
                    else None
                )
                result.status = OrchestrationStatus.COMPLETED
                result.finished_at = _now_iso()
                task_state.add_observation(
                    "Agent resumed from checkpoint: every step was "
                    "already completed before the interruption",
                    source="agent",
                )
                return result
        if plan is None:
            try:
                plan = self.planner.plan(goal, state=task_state)
            except PlanningError as e:
                return self._finish_planning_failure(result, e)
            except Exception as e:  # a Planner breaking its contract
                return self._finish_planning_failure(
                    result,
                    PlanningError(f"Planner failed unexpectedly: {e}"),
                )
            if plan is None or not getattr(plan, "steps", None):
                return self._finish_planning_failure(
                    result,
                    PlanningError("Planner returned an empty plan"),
                )

        result.plan = plan
        task_state.metadata["plan"] = _redacted_plan_dict(plan)
        task_state.add_observation(
            f"Agent generated plan {plan.plan_id} with "
            f"{len(plan.steps)} step(s) for goal: {goal}",
            source="agent",
        )
        self._save_checkpoint(task_state, plan, result)

        # 2) Execute → verify, one step at a time, with recovery ------
        # When a step fails (or cannot be verified), the Recovery
        # mechanism asks the Planner for a *different* plan instead
        # of blindly repeating the failed action.  Recovery is
        # bounded twice: by max_recovery_attempts and by the same
        # overall iteration limit (recovered steps still count).
        executed: list[StepExecutionResult] = []
        verifications: list[VerificationResult] = []
        failure_error: dict[str, Any] | None = None
        outcome = OrchestrationStatus.COMPLETED
        recovery_attempts_used = 0
        attempts_before = len(self.recovery.attempts)

        current_plan = plan
        step_index = 0
        remaining: list = []
        while step_index < len(current_plan.steps):
            step = current_plan.steps[step_index]
            if result.iterations >= limit:
                outcome = OrchestrationStatus.MAX_ITERATIONS_EXCEEDED
                failure_error = {
                    "code": "max_iterations_exceeded",
                    "message": (
                        f"Maximum iteration limit ({limit}) reached; "
                        f"step {step.step_id!r} and the remaining "
                        f"{len(current_plan.steps) - step_index} step(s) "
                        f"were not executed"
                    ),
                    "details": {
                        "max_iterations": limit,
                        "next_step": step.step_id,
                    },
                }
                remaining = list(current_plan.steps[step_index:])
                break
            if duration_budget is not None and (
                time.monotonic() - started_mono
            ) > duration_budget:
                outcome = OrchestrationStatus.MAX_ITERATIONS_EXCEEDED
                failure_error = {
                    "code": "time_limit_exceeded",
                    "message": (
                        f"Time limit ({duration_budget:g}s) "
                        f"reached; step {step.step_id!r} and the "
                        f"remaining "
                        f"{len(current_plan.steps) - step_index} "
                        "step(s) were not executed"
                    ),
                    "details": {
                        "max_duration_s": duration_budget,
                        "next_step": step.step_id,
                    },
                }
                remaining = list(current_plan.steps[step_index:])
                break

            result.iterations += 1
            try:
                execution_result = self.executor.execute_step(
                    step, state=task_state, plan_id=current_plan.plan_id
                )
            except Exception as e:
                # Executor raises only for a malformed step; record
                # it as this step's failure instead of crashing.
                execution_result = StepExecutionResult(
                    step_id=step.step_id,
                    description=step.description,
                    tool_name=step.tool_name,
                    arguments=dict(step.arguments)
                    if isinstance(step.arguments, dict)
                    else {},
                    expected_result=step.expected_result,
                    success=False,
                    error={
                        "code": "execution_failed",
                        "message": f"Step could not be executed: {e}",
                        "tool": step.tool_name,
                        "details": {},
                    },
                    finished_at=_now_iso(),
                )
                task_state.start_step(step.step_id)
                task_state.add_tool_result(
                    step.tool_name,
                    success=False,
                    error=str(e),
                    metadata={"step_id": step.step_id, "plan_id": current_plan.plan_id},
                )
                task_state.fail_step(step.step_id, str(e))

            executed.append(execution_result)
            scrub_keys = self._scrub_stored_plan(
                task_state, step, execution_result.output
            )

            verification_result = self._verify(step, execution_result, task_state)
            if verification_result is not None:
                verifications.append(verification_result)
            result.records.append(
                IterationRecord(
                    iteration=result.iterations,
                    step_id=step.step_id,
                    execution=execution_result,
                    verification=verification_result,
                )
            )
            task_state.add_observation(
                f"Agent iteration {result.iterations}: step "
                f"{step.step_id} executed "
                f"({'success' if execution_result.success else 'failed'})"
                + (
                    f", verification {verification_result.status.value}"
                    if verification_result
                    else ""
                ),
                source="agent",
            )
            self._save_checkpoint(task_state, current_plan, result)

            # Does this step need recovery?
            trigger: str | None = None
            trigger_error: dict[str, Any] | None = None
            fatal = True
            if not execution_result.success:
                trigger = "execution_failed"
                trigger_error = execution_result.error or {
                    "code": "execution_failed",
                    "message": f"Step {step.step_id!r} failed",
                }
            elif (
                verification_result is not None
                and verification_result.status == VerificationStatus.FAILED
            ):
                # A verified-wrong outcome is a task failure unless
                # recovery finds a better way: the action ran, but
                # not the action that was expected.
                trigger = "verification_failed"
                trigger_error = {
                    "code": "verification_failed",
                    "message": verification_result.reason,
                    "details": {"step_id": step.step_id},
                }
            elif (
                verification_result is not None
                and verification_result.status == VerificationStatus.UNCERTAIN
                and (self.recover_on_uncertain or self.strict_verification)
            ):
                trigger = "verification_uncertain"
                trigger_error = {
                    "code": "verification_uncertain",
                    "message": (
                        f"Step {step.step_id!r} could not be verified: "
                        f"{verification_result.reason}"
                    ),
                    "details": {"step_id": step.step_id},
                }
                # Uncertain alone is not fatal (unless strict): if
                # recovery cannot improve it, the original plan
                # continues rather than failing the whole task.
                fatal = self.strict_verification

            if trigger is None:
                step_index += 1
                continue

            # -- recovery: ask the Planner for a different plan ------
            reason = str(trigger_error.get("message") or trigger)
            recovered = False
            recovery_error: dict[str, Any] | None = None
            if (
                self.recovery.enabled
                and recovery_attempts_used < self.recovery.max_attempts
            ):
                recovery_attempts_used += 1
                recovery_step = step
                if scrub_keys:
                    # never hand a typed secret to the replanner
                    masked = dict(step.arguments)
                    for key in scrub_keys:
                        if key in masked:
                            masked[key] = "***"
                    recovery_step = PlanStep(
                        step_id=step.step_id,
                        description=step.description,
                        tool_name=step.tool_name,
                        arguments=masked,
                        expected_result=step.expected_result,
                    )
                try:
                    new_plan = self.recovery.plan_recovery(
                        goal=goal,
                        state=task_state,
                        failed_step=recovery_step,
                        reason=reason,
                        trigger=trigger,
                        attempt=recovery_attempts_used,
                    )
                except RecoveryError as e:
                    recovery_error = e.to_dict()
                else:
                    current_plan = new_plan
                    task_state.metadata["current_plan"] = _redacted_plan_dict(new_plan)
                    task_state.add_observation(
                        f"Agent switching to recovery plan "
                        f"{new_plan.plan_id} ({len(new_plan.steps)} "
                        f"step(s)) after step {step.step_id} "
                        f"({trigger})",
                        source="agent",
                    )
                    step_index = 0
                    recovered = True

            if recovered:
                continue
            if not fatal:
                # Uncertain step, recovery unavailable/failed: keep
                # the original plan going (the uncertainty stays
                # recorded in AgentState and the verification report)
                step_index += 1
                continue

            outcome = OrchestrationStatus.FAILED
            failure_error = dict(trigger_error)
            details = dict(failure_error.get("details") or {})
            details["recovery_attempts"] = recovery_attempts_used
            if recovery_error is not None:
                details["recovery"] = recovery_error
            elif recovery_attempts_used >= self.recovery.max_attempts > 0:
                details["recovery_exhausted"] = True
            failure_error["details"] = details
            remaining = list(current_plan.steps[step_index + 1:])
            break

        result.recovery_attempts = [
            a.to_dict() for a in self.recovery.attempts[attempts_before:]
        ]

        # Remaining steps (after failure / limit) are skipped, not done
        skipped_results = [
            StepExecutionResult(
                step_id=s.step_id,
                description=s.description,
                tool_name=s.tool_name,
                arguments=dict(s.arguments) if isinstance(s.arguments, dict) else {},
                expected_result=s.expected_result,
                success=False,
                skipped=True,
                error={
                    "code": "skipped",
                    "message": (
                        f"Step {s.step_id!r} was not executed because "
                        f"the task stopped ({outcome.value})"
                    ),
                    "tool": s.tool_name,
                    "details": {},
                },
                finished_at=_now_iso(),
            )
            for s in remaining
        ]

        # 3) Aggregate reports + final state --------------------------------
        result.execution = ExecutionReport(
            plan_id=plan.plan_id,
            goal=goal,
            task_id=task_state.task_id,
            status="completed" if outcome == OrchestrationStatus.COMPLETED else "failed",
            step_results=executed + skipped_results,
            started_at=result.started_at,
            finished_at=_now_iso(),
        )
        result.verification = VerificationReport(
            plan_id=plan.plan_id,
            goal=goal,
            task_id=task_state.task_id,
            results=verifications,
        )
        result.status = outcome
        result.error = failure_error
        result.finished_at = _now_iso()

        if outcome == OrchestrationStatus.COMPLETED:
            task_state.complete_task(result=result.summary())
        else:
            task_state.fail_task(
                (failure_error or {}).get("message") or f"Task {outcome.value}"
            )
        task_state.add_observation(
            f"Agent finished: {result.summary()}", source="agent"
        )
        if outcome == OrchestrationStatus.COMPLETED:
            logger.info("task %s", result.summary())
        else:
            logger.warning("task %s", result.summary())
        self._save_checkpoint(task_state, plan, result)
        return result

    # Alias in the vocabulary used by the agent layer
    run_task = run
    execute_goal = run

    # -- observation-driven loop ------------------------------------------

    #: Guidance handed to the Planner in loop mode: decide the
    #: next few actions from the *current* state, or report the
    #: goal complete — the loop re-asks after every batch.
    _LOOP_GUIDANCE = (
        "You are deciding the next actions of an ongoing task, "
        "not the whole task at once. Based strictly on the "
        "current state above (what already completed, what "
        "failed, and the latest observations), propose at most "
        "{batch_limit} next step(s). Do not repeat completed "
        "work and do not repeat a failed action with the same "
        "arguments. If the goal is already achieved, answer "
        "with the completion form instead of steps."
    )

    def run_loop(
        self,
        goal: str,
        *,
        state: AgentState | None = None,
        max_steps: int = 20,
        batch_limit: int = 3,
        max_replans: int = 6,
        max_llm_calls: int = 12,
        max_repeated_actions: int = 2,
        max_duration_s: float | None = None,
    ) -> OrchestrationResult:
        """Run *goal* as a true observe → decide → act → verify
        loop instead of one long pre-generated plan.

        Each cycle the Planner sees the current AgentState (fresh
        observations, completed/failed steps, tool results) and
        proposes a small bounded batch; every action is executed
        and verified before the next decision, a failed or
        uncertain step stops the batch and goes to recovery, and
        the Planner re-decides from the new state — a stale plan
        is never followed blindly.  The loop ends when the
        Planner reports the goal complete, or when any limit
        (steps, time, replans, LLM calls, repeated identical
        actions, recovery attempts) stops it safely.
        """
        if not goal or not str(goal).strip():
            raise ValueError("Agent.run_loop needs a non-empty goal")
        goal = str(goal).strip()
        for name, value in (
            ("max_steps", max_steps),
            ("batch_limit", batch_limit),
            ("max_llm_calls", max_llm_calls),
            ("max_repeated_actions", max_repeated_actions),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer, got {value!r}"
                )
        if (
            isinstance(max_replans, bool)
            or not isinstance(max_replans, int)
            or max_replans < 0
        ):
            raise ValueError(
                f"max_replans must be a non-negative integer, "
                f"got {max_replans!r}"
            )
        duration_budget = (
            self._validate_duration(max_duration_s)
            if max_duration_s is not None
            else self.max_duration_s
        )
        started_mono = time.monotonic()

        task_state = self._prepare_state(goal, state)
        self.state = task_state
        result = OrchestrationResult(
            goal=goal,
            status=OrchestrationStatus.FAILED,
            state=task_state,
            max_iterations=max_steps,
            started_at=_now_iso(),
        )
        task_state.add_observation(
            f"Agent received goal (loop mode): {goal}", source="agent"
        )
        logger.info(
            "loop task %s started: goal=%r max_steps=%d",
            task_state.task_id, goal, max_steps,
        )

        executed: list[StepExecutionResult] = []
        verifications: list[VerificationResult] = []
        attempts_before = len(self.recovery.attempts)
        recovery_attempts_used = 0
        llm_calls = 0
        replans = 0
        signatures: dict[str, int] = {}
        pending: list = []
        pending_plan: TaskPlan | None = None
        last_plan: TaskPlan | None = None
        outcome = OrchestrationStatus.COMPLETED
        failure_error: dict[str, Any] | None = None
        stop = False

        def limits_hit() -> dict[str, Any] | None:
            if result.iterations >= max_steps:
                return {
                    "code": "step_limit_exceeded",
                    "message": (
                        f"Step limit ({max_steps}) reached; the "
                        "task stopped safely"
                    ),
                    "details": {"max_steps": max_steps},
                    "_status": "limit",
                }
            if duration_budget is not None and (
                time.monotonic() - started_mono
            ) > duration_budget:
                return {
                    "code": "time_limit_exceeded",
                    "message": (
                        f"Time limit ({duration_budget:g}s) "
                        "reached; the task stopped safely"
                    ),
                    "details": {"max_duration_s": duration_budget},
                    "_status": "limit",
                }
            return None

        while not stop:
            # -- decide -------------------------------------------
            if pending:
                batch_plan = pending_plan
                batch = pending[:batch_limit]
                pending = pending[batch_limit:]
                if not pending:
                    pending_plan = None
            else:
                hit = limits_hit()
                if hit is not None:
                    outcome = OrchestrationStatus.MAX_ITERATIONS_EXCEEDED
                    failure_error = hit
                    break
                if llm_calls >= max_llm_calls:
                    outcome = OrchestrationStatus.FAILED
                    failure_error = {
                        "code": "llm_call_limit",
                        "message": (
                            f"LLM call limit ({max_llm_calls}) "
                            "reached before the goal completed"
                        ),
                        "details": {"max_llm_calls": max_llm_calls},
                    }
                    break
                if llm_calls >= 1 and replans >= max_replans:
                    outcome = OrchestrationStatus.FAILED
                    failure_error = {
                        "code": "replan_limit",
                        "message": (
                            f"Replan limit ({max_replans}) reached "
                            "before the goal completed"
                        ),
                        "details": {"max_replans": max_replans},
                    }
                    break
                try:
                    plan = self.planner.plan(
                        goal,
                        state=task_state,
                        extra_context=self._LOOP_GUIDANCE.format(
                            batch_limit=batch_limit
                        ),
                    )
                except PlanningError as e:
                    if result.iterations == 0 and not executed:
                        return self._finish_planning_failure(result, e)
                    outcome = OrchestrationStatus.FAILED
                    failure_error = e.to_dict()
                    break
                except Exception as e:
                    if result.iterations == 0 and not executed:
                        return self._finish_planning_failure(
                            result,
                            PlanningError(
                                f"Planner failed unexpectedly: {e}"
                            ),
                        )
                    outcome = OrchestrationStatus.FAILED
                    failure_error = {
                        "code": "planning_failed",
                        "message": f"Planner failed unexpectedly: {e}",
                    }
                    break
                llm_calls += 1
                if llm_calls > 1:
                    replans += 1
                last_plan = plan
                result.plan = plan
                if plan.metadata.get("task_complete") or not plan.steps:
                    task_state.add_observation(
                        "Planner reported the goal complete; "
                        "loop finished",
                        source="agent",
                    )
                    outcome = OrchestrationStatus.COMPLETED
                    break
                task_state.metadata["plan"] = _redacted_plan_dict(plan)
                task_state.add_observation(
                    f"Loop decision: plan {plan.plan_id} proposes "
                    f"{len(plan.steps)} step(s); executing up to "
                    f"{batch_limit} before re-deciding",
                    source="agent",
                )
                batch_plan = plan
                batch = list(plan.steps[:batch_limit])
                if len(plan.steps) > batch_limit:
                    pending = list(plan.steps[batch_limit:])
                    pending_plan = plan

            # -- act + verify (batch stops at the first problem) --
            for step in batch:
                hit = limits_hit()
                if hit is not None:
                    outcome = OrchestrationStatus.MAX_ITERATIONS_EXCEEDED
                    failure_error = hit
                    stop = True
                    break
                signature = json.dumps(
                    {"tool": step.tool_name, "args": step.arguments},
                    sort_keys=True, default=str,
                )
                signatures[signature] = signatures.get(signature, 0) + 1
                if signatures[signature] > max_repeated_actions:
                    outcome = OrchestrationStatus.FAILED
                    failure_error = {
                        "code": "repeated_action",
                        "message": (
                            f"Action {step.tool_name} with the "
                            "same arguments was attempted "
                            f"{signatures[signature]} times; "
                            "stopping instead of looping"
                        ),
                        "details": {
                            "tool": step.tool_name,
                            "max_repeated_actions": max_repeated_actions,
                        },
                    }
                    stop = True
                    break

                result.iterations += 1
                execution_result = self._execute_safely(
                    step, task_state, batch_plan.plan_id
                )
                executed.append(execution_result)
                scrub_keys = self._scrub_stored_plan(
                    task_state, step, execution_result.output
                )
                verification_result = self._verify(
                    step, execution_result, task_state
                )
                if verification_result is not None:
                    verifications.append(verification_result)
                result.records.append(
                    IterationRecord(
                        iteration=result.iterations,
                        step_id=step.step_id,
                        execution=execution_result,
                        verification=verification_result,
                    )
                )
                task_state.add_observation(
                    f"Loop iteration {result.iterations}: step "
                    f"{step.step_id} executed "
                    f"({'success' if execution_result.success else 'failed'})"
                    + (
                        f", verification {verification_result.status.value}"
                        if verification_result
                        else ""
                    ),
                    source="agent",
                )
                self._save_checkpoint(task_state, batch_plan, result)

                trigger, trigger_error, fatal = self._loop_trigger(
                    step, execution_result, verification_result
                )
                if trigger is None:
                    continue

                # -- recover, then re-decide -------------------
                reason = str(trigger_error.get("message") or trigger)
                recovered = False
                recovery_error: dict[str, Any] | None = None
                if (
                    self.recovery.enabled
                    and recovery_attempts_used < self.recovery.max_attempts
                    and replans < max_replans
                ):
                    recovery_attempts_used += 1
                    replans += 1
                    recovery_step = step
                    if scrub_keys:
                        masked = dict(step.arguments)
                        for key in scrub_keys:
                            if key in masked:
                                masked[key] = "***"
                        recovery_step = PlanStep(
                            step_id=step.step_id,
                            description=step.description,
                            tool_name=step.tool_name,
                            arguments=masked,
                            expected_result=step.expected_result,
                        )
                    try:
                        new_plan = self.recovery.plan_recovery(
                            goal=goal,
                            state=task_state,
                            failed_step=recovery_step,
                            reason=reason,
                            trigger=trigger,
                            attempt=recovery_attempts_used,
                        )
                    except RecoveryError as e:
                        recovery_error = e.to_dict()
                    else:
                        recovered = True
                        pending = list(new_plan.steps)
                        pending_plan = new_plan
                        last_plan = new_plan
                        task_state.metadata["current_plan"] = (
                            _redacted_plan_dict(new_plan)
                        )
                if recovered:
                    break  # next cycle executes the recovery batch
                if not fatal:
                    # Uncertain and unrecoverable: stop this
                    # batch and let the Planner re-decide from
                    # the fresh state instead of pressing on.
                    pending = []
                    pending_plan = None
                    task_state.add_observation(
                        f"Step {step.step_id} was uncertain; "
                        "re-deciding from current state",
                        source="agent",
                    )
                    break
                outcome = OrchestrationStatus.FAILED
                failure_error = dict(trigger_error)
                details = dict(failure_error.get("details") or {})
                details["recovery_attempts"] = recovery_attempts_used
                if recovery_error is not None:
                    details["recovery"] = recovery_error
                failure_error["details"] = details
                stop = True
                break
            else:
                # Whole batch verified: discard any unexecuted
                # remainder of that plan and re-decide from the
                # fresh state — never follow a stale plan.
                pending = []
                pending_plan = None

        result.recovery_attempts = [
            a.to_dict() for a in self.recovery.attempts[attempts_before:]
        ]
        result.plan = last_plan
        plan_id = last_plan.plan_id if last_plan else "loop"
        result.execution = ExecutionReport(
            plan_id=plan_id,
            goal=goal,
            task_id=task_state.task_id,
            status=(
                "completed"
                if outcome == OrchestrationStatus.COMPLETED
                else "failed"
            ),
            step_results=executed,
            started_at=result.started_at,
            finished_at=_now_iso(),
        )
        result.verification = VerificationReport(
            plan_id=plan_id,
            goal=goal,
            task_id=task_state.task_id,
            results=verifications,
        )
        result.status = outcome
        result.error = failure_error
        result.finished_at = _now_iso()
        if outcome == OrchestrationStatus.COMPLETED:
            task_state.complete_task(result=result.summary())
        else:
            task_state.fail_task(
                (failure_error or {}).get("message")
                or f"Task {outcome.value}"
            )
        task_state.add_observation(
            f"Agent finished (loop): {result.summary()}",
            source="agent",
        )
        self._save_checkpoint(task_state, last_plan, result)
        return result

    def _execute_safely(
        self,
        step: PlanStep,
        task_state: AgentState,
        plan_id: str,
    ) -> StepExecutionResult:
        """Executor call that converts a malformed-step raise
        into a recorded failure (same contract as run())."""
        try:
            return self.executor.execute_step(
                step, state=task_state, plan_id=plan_id
            )
        except Exception as e:
            task_state.start_step(step.step_id)
            task_state.add_tool_result(
                step.tool_name,
                success=False,
                error=str(e),
                metadata={"step_id": step.step_id, "plan_id": plan_id},
            )
            task_state.fail_step(step.step_id, str(e))
            return StepExecutionResult(
                step_id=step.step_id,
                description=step.description,
                tool_name=step.tool_name,
                arguments=dict(step.arguments)
                if isinstance(step.arguments, dict)
                else {},
                expected_result=step.expected_result,
                success=False,
                error={
                    "code": "execution_failed",
                    "message": f"Step could not be executed: {e}",
                    "tool": step.tool_name,
                    "details": {},
                },
                finished_at=_now_iso(),
            )

    def _loop_trigger(
        self,
        step: PlanStep,
        execution_result: StepExecutionResult,
        verification_result: VerificationResult | None,
    ) -> tuple[str | None, dict[str, Any] | None, bool]:
        """(trigger, error, fatal) for one executed loop step —
        the same failure policy as run()."""
        if not execution_result.success:
            return (
                "execution_failed",
                execution_result.error
                or {
                    "code": "execution_failed",
                    "message": f"Step {step.step_id!r} failed",
                },
                True,
            )
        if (
            verification_result is not None
            and verification_result.status == VerificationStatus.FAILED
        ):
            return (
                "verification_failed",
                {
                    "code": "verification_failed",
                    "message": verification_result.reason,
                    "details": {"step_id": step.step_id},
                },
                True,
            )
        if (
            verification_result is not None
            and verification_result.status == VerificationStatus.UNCERTAIN
            and (self.recover_on_uncertain or self.strict_verification)
        ):
            return (
                "verification_uncertain",
                {
                    "code": "verification_uncertain",
                    "message": (
                        f"Step {step.step_id!r} could not be "
                        f"verified: {verification_result.reason}"
                    ),
                    "details": {"step_id": step.step_id},
                },
                self.strict_verification,
            )
        return None, None, True

    # -- checkpointing -------------------------------------------------

    def _scrub_stored_plan(
        self,
        task_state: AgentState,
        step: PlanStep,
        output: Any,
    ) -> tuple[str, ...]:
        """Mask credential/payment values in stored plan records.

        When a step touched credential or payment material, the
        typed values are secrets: the Executor already scrubbed
        the step record, and here the plan copies kept in
        AgentState metadata get the same treatment.  Returns the
        masked argument keys (for recovery-context scrubbing).
        """
        scrub_keys = sensitive_argument_keys(output)
        if not scrub_keys:
            return ()
        for record_key in ("plan", "current_plan"):
            stored = task_state.metadata.get(record_key)
            if not isinstance(stored, dict):
                continue
            for step_dict in stored.get("steps", []):
                if not isinstance(step_dict, dict):
                    continue
                if step_dict.get("step_id") != step.step_id:
                    continue
                stored_args = step_dict.get("arguments")
                if isinstance(stored_args, dict):
                    for key in scrub_keys:
                        if key in stored_args:
                            stored_args[key] = "***"
        return scrub_keys

    def _resume_plan(
        self,
        checkpoint: dict[str, Any],
        task_state: AgentState,
        result: OrchestrationResult,
    ) -> tuple[TaskPlan | None, bool]:
        """The not-yet-completed remainder of a checkpointed plan.

        Returns ``(plan, False)`` with the steps still to run, or
        ``(None, True)`` when everything already finished.  A
        checkpoint without a plan falls back to ``(None, False)``
        so the caller replans against the restored state.
        """
        if not checkpoint.get("plan"):
            return None, False
        full_plan = TaskPlan.from_dict(checkpoint["plan"])
        completed_ids = {
            record.name for record in task_state.completed_steps
        }
        remaining = [
            step
            for step in full_plan.steps
            if step.step_id not in completed_ids
        ]
        task_state.add_observation(
            f"Agent resumed task from checkpoint at iteration "
            f"{result.iterations}: {len(remaining)} of "
            f"{len(full_plan.steps)} step(s) remaining",
            source="agent",
        )
        if not remaining:
            return None, True
        return (
            TaskPlan(
                goal=full_plan.goal,
                steps=remaining,
                plan_id=full_plan.plan_id,
                task_id=full_plan.task_id,
                created_at=full_plan.created_at,
                metadata=dict(full_plan.metadata),
            ),
            False,
        )

    def _save_checkpoint(
        self,
        task_state: AgentState,
        plan: TaskPlan | None,
        result: OrchestrationResult,
    ):
        """Persist a task checkpoint (never fails the task)."""
        self._active_checkpoint = (task_state, plan, result)
        if self.checkpointer is None:
            return None
        try:
            extra: dict[str, Any] = {}
            if self.checkpoint_extra_provider is not None:
                extra = self.checkpoint_extra_provider() or {}
            return self.checkpointer.save_snapshot(
                state=task_state,
                plan=plan,
                goal=result.goal,
                status=task_state.status.value,
                iteration=result.iterations,
                extra=extra,
            )
        except Exception as exc:  # checkpointing must not kill tasks
            logger.warning(
                "checkpoint save failed for task %s: %s",
                task_state.task_id,
                exc,
            )
            return None

    def save_checkpoint(self):
        """Checkpoint the current (or last) task right now.

        Returns the checkpoint path.  Raises ValueError when no
        checkpointer is configured or no task has run yet.
        """
        if self.checkpointer is None:
            raise ValueError(
                "No checkpointer configured for this Agent"
            )
        if self._active_checkpoint is None:
            raise ValueError("No task has run yet to checkpoint")
        task_state, plan, result = self._active_checkpoint
        return self._save_checkpoint(task_state, plan, result)

    # -- helpers ---------------------------------------------------------
    def _prepare_state(
        self, goal: str, state: AgentState | None
    ) -> AgentState:
        candidate = state if state is not None else self.state
        if candidate is not None and not candidate.is_terminal:
            if candidate.status == TaskStatus.PENDING:
                candidate.start_task()
            return candidate
        # Fresh task (none yet, or the previous one already ended)
        return AgentState.create(goal)

    def _verify(
        self,
        step,
        execution_result: StepExecutionResult,
        state: AgentState,
    ) -> VerificationResult | None:
        try:
            return self.verifier.verify_step(
                step, execution_result, state=state
            )
        except Exception as e:  # a broken verifier must not crash the task
            result = VerificationResult(
                step_id=step.step_id,
                tool_name=step.tool_name,
                status=VerificationStatus.UNCERTAIN,
                expected_result=step.expected_result or "",
                actual_output=execution_result.output,
                reason=f"Verifier could not judge this step: {e}",
                confidence=0.0,
                evidence={"source": "agent", "verifier_error": str(e)},
            )
            self.verifier._record(state, result)  # same recording path
            return result

    def _finish_planning_failure(
        self,
        result: OrchestrationResult,
        error: PlanningError,
    ) -> OrchestrationResult:
        result.status = OrchestrationStatus.PLANNING_FAILED
        result.error = error.to_dict()
        result.finished_at = _now_iso()
        result.state.fail_task(error.error.message)
        result.state.add_observation(
            f"Agent could not plan the task: {error.error.message}",
            source="agent",
        )
        logger.warning(
            "task %s planning failed: %s",
            result.state.task_id, error.error.message,
        )
        return result

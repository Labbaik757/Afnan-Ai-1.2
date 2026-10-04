"""Recovery — replanning after a failed or uncertain step.

When a plan step fails (the tool errored) or its outcome cannot be
verified (the Verifier judged it *uncertain* or *failed*), simply
running the same action again usually fails the same way.  The
:class:`RecoveryManager` in this module gives the **Planner**
everything it needs to produce a *different* plan instead:

* the original goal,
* the failure reason (tool error or verification judgement),
* the current ``AgentState`` (completed steps, observations,
  previous attempts), and
* the exact previous attempt (tool + arguments) that did not work.

Rules
-----
* **No blind repetition.**  A recovery plan that contains any
  action (same tool, same arguments) that already failed — in the
  triggering attempt or in an earlier recovery attempt — is
  rejected with a structured ``repeated_action`` error and never
  executed.
* **Limited attempts.**  ``max_attempts`` (default 2, never
  negative) caps how many recovery plans may be generated for one
  task.  Recovery also stays inside the orchestrator's overall
  maximum-iteration limit, because every recovered step still
  counts as an iteration there.
* **Everything is recorded.**  Every attempt — replanned,
  planner-failed or rejected — is appended to
  ``state.metadata["recovery_attempts"]`` and noted as a
  ``recovery`` observation in AgentState.
* **No duplicated responsibilities.**  Recovery only *plans
  again*: it asks the Planner for a new TaskPlan, exactly like the
  first planning pass.  It never executes a tool, never judges a
  result, and never touches a platform adapter — execution stays
  with the Executor, verification with the Verifier, and the
  decision of when to recover (and when to give up) with the
  orchestrating Agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from afnan_ai.log_config import get_logger
from afnan_ai.planner import Planner, PlanningError, TaskPlan
from afnan_ai.state import AgentState


logger = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _action_signature(tool_name: str, arguments: Any) -> str:
    """Stable identity of one action: tool + exact arguments."""
    try:
        args = json.dumps(arguments or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        args = str(arguments)
    return f"{tool_name}::{args}"


class RecoveryErrorCode(str, Enum):
    PLANNER_FAILED = "planner_failed"
    REPEATED_ACTION = "repeated_action"
    LIMIT_REACHED = "limit_reached"


@dataclass
class RecoveryErrorRecord:
    code: RecoveryErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": dict(self.details),
        }


class RecoveryError(Exception):
    """Raised when a recovery plan cannot be produced (the planner
    failed, the attempt limit is reached, or the new plan would
    blindly repeat an action that already failed)."""

    def __init__(
        self,
        message: str,
        *,
        code: RecoveryErrorCode = RecoveryErrorCode.PLANNER_FAILED,
        details: dict[str, Any] | None = None,
    ):
        self.error = RecoveryErrorRecord(
            code=code, message=message, details=dict(details or {})
        )
        super().__init__(message)

    @property
    def code(self) -> RecoveryErrorCode:
        return self.error.code

    def to_dict(self) -> dict[str, Any]:
        return self.error.to_dict()


@dataclass
class RecoveryAttempt:
    """One recorded recovery attempt for a task."""

    attempt: int
    trigger: str  # execution_failed / verification_failed / verification_uncertain
    failed_step_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    outcome: str = "replanned"  # replanned / planner_failed / repeated_action / limit_reached
    new_plan_id: str | None = None
    error: dict[str, Any] | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "trigger": self.trigger,
            "failed_step_id": self.failed_step_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "reason": self.reason,
            "outcome": self.outcome,
            "new_plan_id": self.new_plan_id,
            "error": dict(self.error) if self.error else None,
            "created_at": self.created_at,
        }


class RecoveryManager:
    """Generate a fresh plan after a failed/uncertain step.

    The manager owns no execution or verification logic: it builds
    the recovery context, asks the Planner for a new plan, checks
    that the plan does not repeat a failed action, and records the
    attempt in AgentState.
    """

    DEFAULT_MAX_ATTEMPTS = 2

    def __init__(
        self,
        planner: Planner,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ):
        if planner is None:
            raise ValueError("RecoveryManager needs a Planner")
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts < 0
        ):
            raise ValueError(
                f"max_attempts must be a non-negative integer, "
                f"got {max_attempts!r}"
            )
        self.planner = planner
        self.max_attempts = max_attempts
        self.attempts: list[RecoveryAttempt] = []

    @property
    def enabled(self) -> bool:
        return self.max_attempts > 0

    # -- main entry point -------------------------------------------------
    def plan_recovery(
        self,
        *,
        goal: str,
        state: AgentState,
        failed_step,
        reason: str,
        trigger: str,
        attempt: int,
    ) -> TaskPlan:
        """Ask the Planner for a new plan that avoids the failed
        action, and record the attempt in *state*.

        Returns the new TaskPlan.  Raises RecoveryError when the
        attempt limit is already reached, the Planner cannot
        produce a plan, or the new plan would repeat an action
        that already failed.
        """
        record = RecoveryAttempt(
            attempt=attempt,
            trigger=trigger,
            failed_step_id=getattr(failed_step, "step_id", "") or "",
            tool_name=getattr(failed_step, "tool_name", "") or "",
            arguments=dict(getattr(failed_step, "arguments", {}) or {}),
            reason=reason or "",
            created_at=_now_iso(),
        )

        if attempt > self.max_attempts:
            record.outcome = "limit_reached"
            record.error = RecoveryError(
                f"Recovery attempt limit ({self.max_attempts}) reached",
                code=RecoveryErrorCode.LIMIT_REACHED,
                details={"attempt": attempt},
            ).to_dict()
            self._record(state, record)
            raise RecoveryError(
                record.error["message"],
                code=RecoveryErrorCode.LIMIT_REACHED,
                details={"attempt": attempt, "max_attempts": self.max_attempts},
            )

        context = self._build_context(
            goal=goal,
            state=state,
            record=record,
        )
        try:
            plan = self.planner.plan(
                goal, state=state, recovery_context=context
            )
        except PlanningError as e:
            record.outcome = "planner_failed"
            record.error = e.to_dict()
            self._record(state, record)
            raise RecoveryError(
                f"Recovery planning failed: {e.error.message}",
                code=RecoveryErrorCode.PLANNER_FAILED,
                details={"planner_error": e.to_dict()},
            ) from e
        except Exception as e:  # a planner breaking its contract
            record.outcome = "planner_failed"
            record.error = RecoveryError(
                f"Recovery planning failed unexpectedly: {e}",
                code=RecoveryErrorCode.PLANNER_FAILED,
            ).to_dict()
            self._record(state, record)
            raise RecoveryError(
                record.error["message"],
                code=RecoveryErrorCode.PLANNER_FAILED,
            ) from e

        repeated = self._find_repeated_action(plan, state, record)
        if repeated is not None:
            record.outcome = "repeated_action"
            record.new_plan_id = getattr(plan, "plan_id", None)
            record.error = RecoveryError(
                f"Recovery plan repeats the already-failed action "
                f"{repeated}; refusing to run it again",
                code=RecoveryErrorCode.REPEATED_ACTION,
                details={"action": repeated},
            ).to_dict()
            self._record(state, record)
            raise RecoveryError(
                record.error["message"],
                code=RecoveryErrorCode.REPEATED_ACTION,
                details={"action": repeated},
            )

        record.outcome = "replanned"
        record.new_plan_id = getattr(plan, "plan_id", None)
        self._record(state, record)
        return plan

    # -- context & repetition checks ---------------------------------------
    def _build_context(
        self,
        *,
        goal: str,
        state: AgentState,
        record: RecoveryAttempt,
    ) -> str:
        completed = [s.name for s in state.completed_steps]
        tried = self._tried_signatures(state, record)
        lines = [
            f"Goal: {goal}",
            f"Recovery attempt: {record.attempt} of {self.max_attempts}",
            f"What went wrong ({record.trigger}): {record.reason}",
            "Previous attempt that did not work: "
            f"step {record.failed_step_id!r} used tool "
            f"{record.tool_name!r} with arguments "
            f"{json.dumps(record.arguments, ensure_ascii=False, default=str)}",
        ]
        if completed:
            lines.append(
                "Steps already completed successfully (do not redo "
                f"them): {', '.join(completed)}"
            )
        if len(tried) > 1:
            lines.append(
                "Actions already tried and failed (do not repeat "
                f"any of these): {'; '.join(sorted(tried))}"
            )
        return "\n".join(lines)

    def _tried_signatures(
        self, state: AgentState, current: RecoveryAttempt
    ) -> set[str]:
        """Signatures of every action known to have failed: the
        triggering attempt plus all earlier recovery attempts."""
        signatures = {
            _action_signature(current.tool_name, current.arguments)
        }
        for entry in state.metadata.get("recovery_attempts", []):
            if isinstance(entry, dict):
                signatures.add(
                    _action_signature(
                        entry.get("tool_name", ""),
                        entry.get("arguments", {}),
                    )
                )
        return signatures

    def _find_repeated_action(
        self, plan: TaskPlan, state: AgentState, record: RecoveryAttempt
    ) -> str | None:
        tried = self._tried_signatures(state, record)
        for step in getattr(plan, "steps", []) or []:
            signature = _action_signature(
                getattr(step, "tool_name", ""),
                getattr(step, "arguments", {}),
            )
            if signature in tried:
                return signature
        return None

    # -- recording -----------------------------------------------------------
    def _record(self, state: AgentState, record: RecoveryAttempt) -> None:
        self.attempts.append(record)
        state.metadata.setdefault("recovery_attempts", []).append(
            record.to_dict()
        )
        state.add_observation(
            f"Recovery attempt {record.attempt} "
            f"({record.trigger}, step {record.failed_step_id}): "
            f"{record.outcome}"
            + (f" -> plan {record.new_plan_id}" if record.new_plan_id else "")
            + (f" — {record.reason}" if record.reason else ""),
            source="recovery",
        )
        logger.info(
            "recovery attempt %d (%s, step %s): %s",
            record.attempt, record.trigger,
            record.failed_step_id, record.outcome,
        )

"""User controls for live agents.

Semantics (never mixed):

- **pause**: temporary suspension at a safe boundary.
  Checkpoint + persist, block new actions, status PAUSED.
- **resume**: validate checkpoint, re-observe, discard stale
  assumptions, continue the loop.
- **stop**: safely terminate the *current execution*.
  The task stays resumable from its checkpoint.
- **cancel**: permanently mark the task cancelled (terminal).
- **retry**: re-run a failed task from scratch or checkpoint.
- **emergency_stop**: immediate safety termination.

Control flow (the control plane never bypasses the loop)::

    UI → TaskControlService → authorization
      → TaskManager / LoopControl → state transition
      → ActivityEvent

Running irreversible actions are never interrupted
mid-state; pause/stop act at the next safe loop boundary
unless EmergencyStop is explicitly required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from afnan_ai.activity import events as E

PAUSE = "pause"
RESUME = "resume"
STOP = "stop"
CANCEL = "cancel"
RETRY = "retry"
APPROVE = "approve"
DENY = "deny"
RESTART_FROM_CHECKPOINT = "restart_from_checkpoint"
EMERGENCY_STOP = "emergency_stop"

ALL_COMMANDS = frozenset(
    {
        PAUSE, RESUME, STOP, CANCEL, RETRY, APPROVE, DENY,
        RESTART_FROM_CHECKPOINT, EMERGENCY_STOP,
    }
)


@dataclass
class ControlCommand:
    command: str
    task_id: str = ""
    approval_id: str = ""
    reason: str = ""
    actor: str = "user"
    issued_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    def __post_init__(self) -> None:
        if self.command not in ALL_COMMANDS:
            raise ValueError(
                f"unknown control command: {self.command!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ControlError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class TaskControlService:
    """Executes user control commands through the authorized
    pipeline.  Holds *references* to the real authorities —
    it never reimplements their state machines."""

    def __init__(
        self,
        *,
        center: Any,
        task_manager: Any,
        loop_controls: dict[str, Any] | None = None,
        authorizer: Callable[[ControlCommand], None]
        | None = None,
        task_worker: Any | None = None,
        approval_center: Any | None = None,
        emergency: Callable[[str], None] | None = None,
    ) -> None:
        self.center = center
        self.tasks = task_manager
        self.loop_controls: dict[str, Any] = (
            loop_controls if loop_controls is not None else {}
        )
        self.authorizer = authorizer
        self.worker = task_worker
        self.approvals = approval_center
        self._emergency = emergency

    # -- helpers -------------------------------------------------------
    def _authorize(self, cmd: ControlCommand) -> None:
        if self.authorizer is not None:
            self.authorizer(cmd)  # raises on denial

    def _get(self, task_id: str) -> Any:
        task = self.tasks.get(task_id)
        if task is None:
            raise ControlError(
                "task_not_found", f"Task not found: {task_id}"
            )
        return task

    @staticmethod
    def _attr(task: Any, name: str, default: str = "") -> str:
        if isinstance(task, dict):
            return str(task.get(name, default))
        return str(getattr(task, name, default))

    def _emit_control(
        self, cmd: ControlCommand, outcome: str
    ) -> None:
        self.center.emit(
            E.CONTROL_ISSUED,
            f"Control '{cmd.command}' {outcome} "
            f"for task {cmd.task_id or '—'}",
            task_id=cmd.task_id,
            actor=cmd.actor,
            source="control",
            details={
                "command": cmd.command,
                "outcome": outcome,
                "reason": cmd.reason[:200],
            },
        )

    def register_loop(
        self, task_id: str, control: Any
    ) -> None:
        """Bind a live AgentLoop's LoopControl to a task."""
        self.loop_controls[task_id] = control

    def unregister_loop(self, task_id: str) -> None:
        self.loop_controls.pop(task_id, None)

    # -- commands --------------------------------------------------------
    def execute(self, cmd: ControlCommand) -> dict[str, Any]:
        self._authorize(cmd)
        handler = {
            PAUSE: self._pause,
            RESUME: self._resume,
            STOP: self._stop,
            CANCEL: self._cancel,
            RETRY: self._retry,
            RESTART_FROM_CHECKPOINT: self._restart,
            EMERGENCY_STOP: self._emergency_stop,
            APPROVE: self._approve,
            DENY: self._deny,
        }[cmd.command]
        try:
            result = handler(cmd)
        except ControlError:
            raise
        except Exception as exc:  # never leak internals
            raise ControlError(
                "control_failed", f"{cmd.command} failed"
            ) from exc
        self._emit_control(cmd, "accepted")
        return result

    def _pause(self, cmd: ControlCommand) -> dict[str, Any]:
        task = self._get(cmd.task_id)
        control = self.loop_controls.get(cmd.task_id)
        if control is not None:
            # Cooperative: the loop pauses at the next safe
            # boundary, checkpoints, and emits task_paused.
            control.request_pause()
        self.tasks.pause(cmd.task_id)
        self.center.emit(
            E.TASK_PAUSED,
            f"Task paused: {self._attr(task, "goal_text")[:160] or cmd.task_id}",
            task_id=cmd.task_id,
            actor=cmd.actor,
            source="control",
            details={"reason": cmd.reason[:200]},
        )
        return {"task_id": cmd.task_id, "status": "paused"}

    def _resume(self, cmd: ControlCommand) -> dict[str, Any]:
        task = self._get(cmd.task_id)
        if self._attr(task, "status") not in (
            "paused", "waiting_for_approval", "failed",
            "pending",
        ):
            raise ControlError(
                "invalid_state",
                f"Cannot resume task in state {self._attr(task, 'status')}",
            )
        control = self.loop_controls.get(cmd.task_id)
        if control is not None:
            control.resume()
        self.tasks.resume(cmd.task_id)
        self.center.emit(
            E.TASK_RESUMED,
            "Task resumed from checkpoint",
            task_id=cmd.task_id,
            actor=cmd.actor,
            source="control",
        )
        # If a worker is attached, kick it; otherwise the
        # task waits in pending for the normal scheduler.
        if self.worker is not None:
            try:
                self.worker.resume_task(cmd.task_id)
            except Exception:
                pass
        return {"task_id": cmd.task_id, "status": "pending"}

    def _stop(self, cmd: ControlCommand) -> dict[str, Any]:
        """Safe termination of the current execution.

        The loop stops at its next safe boundary; the task is
        paused (resumable from checkpoint), and the stop is
        recorded distinctly from pause/cancel.
        """
        self._get(cmd.task_id)
        control = self.loop_controls.get(cmd.task_id)
        if control is not None:
            control.request_stop()
        try:
            self.tasks.pause(cmd.task_id)
        except Exception:
            pass
        self.center.emit(
            E.TASK_STOPPED,
            "Execution stopped safely at a safe boundary",
            task_id=cmd.task_id,
            actor=cmd.actor,
            source="control",
            details={"reason": cmd.reason[:200]},
        )
        return {"task_id": cmd.task_id, "status": "stopped"}

    def _cancel(self, cmd: ControlCommand) -> dict[str, Any]:
        self._get(cmd.task_id)
        control = self.loop_controls.get(cmd.task_id)
        if control is not None:
            control.request_stop()
        self.unregister_loop(cmd.task_id)
        self.tasks.cancel(cmd.task_id)
        self.center.emit(
            E.TASK_CANCELLED,
            "Task cancelled permanently",
            task_id=cmd.task_id,
            actor=cmd.actor,
            source="control",
            details={"reason": cmd.reason[:200]},
        )
        return {"task_id": cmd.task_id, "status": "cancelled"}

    def _retry(self, cmd: ControlCommand) -> dict[str, Any]:
        task = self._get(cmd.task_id)
        if self._attr(task, "status") not in ("failed", "cancelled", "completed"):
            raise ControlError(
                "invalid_state",
                f"Cannot retry task in state {self._attr(task, 'status')}",
            )
        self.tasks.resume(cmd.task_id)  # failed → pending
        self.center.emit(
            E.TASK_STARTED,
            "Task retried",
            task_id=cmd.task_id,
            actor=cmd.actor,
            source="control",
        )
        return {"task_id": cmd.task_id, "status": "pending"}

    def _restart(self, cmd: ControlCommand) -> dict[str, Any]:
        # Restart from checkpoint == resume with re-observation.
        return self._resume(cmd)

    def _emergency_stop(self, cmd: ControlCommand) -> dict[str, Any]:
        if self._emergency is not None:
            self._emergency(cmd.reason or "user emergency stop")
        for control in list(self.loop_controls.values()):
            try:
                control.request_stop()
            except Exception:
                pass
        self.center.emit(
            E.EMERGENCY_STOP,
            "Emergency stop engaged",
            task_id=cmd.task_id,
            actor=cmd.actor,
            source="control",
            details={"reason": cmd.reason[:200]},
        )
        return {"stopped": True}

    def _approve(self, cmd: ControlCommand) -> dict[str, Any]:
        if self.approvals is None:
            raise ControlError(
                "no_approval_center", "Approval center unavailable"
            )
        return self.approvals.decide(
            cmd.approval_id, granted=True, actor=cmd.actor
        )

    def _deny(self, cmd: ControlCommand) -> dict[str, Any]:
        if self.approvals is None:
            raise ControlError(
                "no_approval_center", "Approval center unavailable"
            )
        return self.approvals.decide(
            cmd.approval_id, granted=False, actor=cmd.actor
        )

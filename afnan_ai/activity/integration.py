"""Wiring: connect real Afnan runtime systems to the ActivityCenter.

Every adapter subscribes to *existing* event hooks — no
subsystem is reimplemented and no orchestration is
duplicated::

    attach_activity_center(
        center,
        agent_loop_factory=...,
        task_manager=...,
        audit_logger=...,
        artifact_manager=...,
        background_runner=...,
        security_center=...,
        workspace_manager=...,
    )

After attaching:

- AgentLoop ``on_event`` → ActivityEvent (via from_loop_event)
- TaskManager transitions → task_* events (via a small hook;
  TaskManager gains an optional ``on_event`` callback)
- AuditLogger records → security-relevant activity events
- ArtifactManager create/update → artifact_created events
- BackgroundRunner entries → task lifecycle events
- SecurityCenter emergency → emergency_stop events

The loop keeps running if the UI disconnects: publishing is
best-effort and never raises into the agent.
"""

from __future__ import annotations

from typing import Any, Callable

from afnan_ai.activity import events as E
from afnan_ai.activity.approval import ApprovalCenter
from afnan_ai.activity.center import ActivityCenter
from afnan_ai.activity.control import TaskControlService
from afnan_ai.activity.notifications import NotificationService
from afnan_ai.activity.query import (
    ActivityQueryService,
    ArtifactQueryService,
    TaskQueryService,
)
from afnan_ai.activity.timeline import ProgressTracker


class ActivityIntegration:
    """Holds every service; the app's single composition root."""

    def __init__(
        self,
        center: ActivityCenter,
        *,
        task_manager: Any = None,
        control: TaskControlService | None = None,
        approvals: ApprovalCenter | None = None,
        task_query: TaskQueryService | None = None,
        activity_query: ActivityQueryService | None = None,
        artifact_query: ArtifactQueryService | None = None,
    ) -> None:
        self.center = center
        self.task_manager = task_manager
        self.control = control
        self.approvals = approvals
        self.task_query = task_query
        self.activity_query = activity_query
        self.artifact_query = artifact_query


def _task_goal(task_manager: Any, task_id: str) -> str:
    try:
        task = task_manager.get(task_id)
        return str(getattr(task, "goal_text", ""))[:300]
    except Exception:
        return ""


def attach_agent_loop(
    center: ActivityCenter,
    *,
    task_id: str,
    workspace_id: str = "",
    task_manager: Any = None,
) -> Callable[[Any], None]:
    """Return an ``on_event`` sink for AgentLoop.run(...).

    Pass the returned callable as ``on_event``; every real
    loop event becomes an ActivityEvent for this task.
    """

    def sink(loop_event: Any) -> None:
        try:
            activity = E.from_loop_event(
                loop_event,
                task_id=task_id,
                workspace_id=workspace_id,
            )
            center.publish(activity)
        except Exception:
            pass

    # Task lifecycle bookends around the run.
    center.emit(
        E.TASK_STARTED,
        f"Task started: {_task_goal(task_manager, task_id)[:160] or task_id}",
        task_id=task_id,
        workspace_id=workspace_id,
        source="loop",
        details=(
            {"goal": _task_goal(task_manager, task_id)}
            if task_manager
            else {}
        ),
    )
    return sink


def attach_task_manager(
    center: ActivityCenter,
    task_manager: Any,
) -> None:
    """Hook TaskManager transitions into activity events.

    Adds an optional ``on_event`` callback attribute; the
    manager calls it (best-effort) on every transition.
    """
    original_transition = task_manager._transition

    def hooked(task: Any, status: str) -> Any:
        before = task.status
        result = original_transition(task, status)
        try:
            type_map = {
                "pending": E.TASK_CREATED,
                "running": E.TASK_STARTED,
                "paused": E.TASK_PAUSED,
                "waiting_for_approval": E.APPROVAL_REQUIRED,
                "completed": E.TASK_COMPLETED,
                "failed": E.TASK_FAILED,
                "cancelled": E.TASK_CANCELLED,
            }
            event_type = type_map.get(status)
            if event_type and before != status:
                center.emit(
                    event_type,
                    f"Task {status}: "
                    f"{str(getattr(task, 'goal_text', ''))[:160]}",
                    task_id=str(
                        getattr(task, "task_id", "")
                    ),
                    source="task",
                    details={
                        "from": before,
                        "goal": str(
                            getattr(task, "goal_text", "")
                        )[:300],
                    },
                )
        except Exception:
            pass
        return result

    task_manager._transition = hooked  # type: ignore[method-assign]


def attach_audit_logger(
    center: ActivityCenter, audit_logger: Any
) -> None:
    """Surface security-relevant audit records as activity."""
    previous = getattr(audit_logger, "_on_event", None)

    def hook(record: dict[str, Any]) -> None:
        if previous is not None:
            try:
                previous(record)
            except Exception:
                pass
        try:
            activity = E.from_audit_record(record)
            if activity is not None:
                center.publish(activity)
        except Exception:
            pass

    audit_logger._on_event = hook


def attach_artifact_manager(
    center: ActivityCenter, artifact_manager: Any
) -> None:
    """Emit artifact_created when artifacts are created."""
    original_create = artifact_manager.create

    def hooked(*args: Any, **kwargs: Any) -> Any:
        artifact = original_create(*args, **kwargs)
        try:
            name = str(
                getattr(artifact, "name", "")
                or kwargs.get("name", "")
            )
            task_id = str(kwargs.get("task_id", ""))
            center.emit(
                E.ARTIFACT_CREATED,
                f"Artifact created: {name[:160]}",
                task_id=task_id,
                source="artifact",
                details={
                    "artifact_name": name,
                    "artifact_type": str(
                        getattr(artifact, "type", "")
                    ),
                },
                evidence_refs=[f"artifact:{name}"] if name else [],
            )
        except Exception:
            pass
        return artifact

    artifact_manager.create = hooked  # type: ignore[method-assign]


def attach_background_runner(
    center: ActivityCenter, runner: Any
) -> None:
    """Forward background task entries as activity events."""
    previous = getattr(runner, "on_event", None)

    def hook(entry: dict[str, Any]) -> None:
        if previous is not None:
            try:
                previous(entry)
            except Exception:
                pass
        try:
            kind = str(entry.get("kind", ""))
            task_id = str(entry.get("task_id", ""))
            type_map = {
                "task_started": E.TASK_STARTED,
                "task_completed": E.TASK_COMPLETED,
                "task_failed": E.TASK_FAILED,
                "checkpoint": E.CHECKPOINT_CREATED,
            }
            event_type = type_map.get(kind, E.WAITING)
            center.emit(
                event_type,
                str(entry.get("message", kind))[:400],
                task_id=task_id,
                source="system",
                details={
                    k: v
                    for k, v in entry.items()
                    if k not in ("message",)
                },
            )
        except Exception:
            pass

    runner.on_event = hook


def attach_emergency(
    center: ActivityCenter, security_center: Any
) -> Callable[[str], None]:
    """Return the emergency callable for TaskControlService."""
    original_trip = security_center.trip_emergency

    def trip(reason: str = "") -> None:
        try:
            original_trip()
        except Exception:
            pass
        try:
            center.emit(
                E.EMERGENCY_STOP,
                "Emergency stop engaged",
                actor="user",
                source="control",
                details={"reason": str(reason)[:200]},
            )
        except Exception:
            pass

    return trip


def build_integration(
    *,
    store_path: str | None = None,
    task_manager: Any = None,
    security_center: Any = None,
    audit_logger: Any = None,
    artifact_manager: Any = None,
    background_runner: Any = None,
    task_worker: Any | None = None,
    authorizer: Callable[[Any], None] | None = None,
) -> ActivityIntegration:
    """Compose the full ActivityCenter stack (app entry point)."""
    from afnan_ai.activity.center import EventStore

    center = ActivityCenter(store=EventStore(store_path))
    progress = ProgressTracker()
    notifier = NotificationService()
    center.progress = progress
    approvals = ApprovalCenter(
        center=center,
        security_center=security_center,
        task_manager=task_manager,
    )
    center.approvals = approvals
    center.notifier = notifier

    emergency = None
    if security_center is not None:
        emergency = attach_emergency(center, security_center)

    control = TaskControlService(
        center=center,
        task_manager=task_manager,
        authorizer=authorizer,
        task_worker=task_worker,
        approval_center=approvals,
        emergency=emergency,
    )

    if task_manager is not None:
        attach_task_manager(center, task_manager)
    if audit_logger is not None:
        attach_audit_logger(center, audit_logger)
    if artifact_manager is not None:
        attach_artifact_manager(center, artifact_manager)
    if background_runner is not None:
        attach_background_runner(center, background_runner)

    return ActivityIntegration(
        center,
        task_manager=task_manager,
        control=control,
        approvals=approvals,
        task_query=TaskQueryService(
            center=center,
            task_manager=task_manager,
            progress=progress,
        )
        if task_manager
        else None,
        activity_query=ActivityQueryService(center=center),
        artifact_query=ArtifactQueryService(
            center=center, artifact_manager=artifact_manager
        )
        if artifact_manager
        else None,
    )

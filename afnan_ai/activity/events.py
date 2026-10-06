"""Activity event model for the Afnan Activity Center.

One structured, serializable event type.  Sources (AgentLoop,
TaskManager, AuditLogger, ArtifactManager, BackgroundRunner,
control plane) all adapt into this shape so the timeline,
queries and UI speak one language.

An ActivityEvent never carries secrets or raw chain-of-thought:
``summary`` is user-safe prose, ``details`` is redacted.
The ``audit_ref`` links to the tamper-evident AuditCenter
record for the same occurrence (correlation, not duplication).
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from afnan_ai.redaction import redact_value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- event types ------------------------------------------------------
# Canonical event types; loop/audit/artifact sources map into these.

TASK_CREATED = "task_created"
TASK_STARTED = "task_started"
PLANNING = "planning"
OBSERVING = "observing"
DECIDING = "deciding"
ACTION_STARTED = "action_started"
ACTION_COMPLETED = "action_completed"
ACTION_FAILED = "action_failed"
VERIFICATION_STARTED = "verification_started"
VERIFICATION_COMPLETED = "verification_completed"
RECOVERY_STARTED = "recovery_started"
REPLANNING = "replanning"
APPROVAL_REQUIRED = "approval_required"
APPROVAL_GRANTED = "approval_granted"
APPROVAL_DENIED = "approval_denied"
WAITING = "waiting"
CHECKPOINT_CREATED = "checkpoint_created"
ARTIFACT_CREATED = "artifact_created"
TASK_PAUSED = "task_paused"
TASK_RESUMED = "task_resumed"
TASK_CANCELLED = "task_cancelled"
TASK_COMPLETED = "task_completed"
TASK_FAILED = "task_failed"
WORKSPACE_STARTED = "workspace_started"
WORKSPACE_STOPPED = "workspace_stopped"
RESOURCE_LIMIT = "resource_limit"
EMERGENCY_STOP = "emergency_stop"
TASK_STOPPED = "task_stopped"
CONTROL_ISSUED = "control_issued"
SUMMARY_GENERATED = "summary_generated"

ALL_TYPES = frozenset(
    {
        TASK_CREATED, TASK_STARTED, PLANNING, OBSERVING,
        DECIDING, ACTION_STARTED, ACTION_COMPLETED,
        ACTION_FAILED, VERIFICATION_STARTED,
        VERIFICATION_COMPLETED, RECOVERY_STARTED,
        REPLANNING, APPROVAL_REQUIRED, APPROVAL_GRANTED,
        APPROVAL_DENIED, WAITING, CHECKPOINT_CREATED,
        ARTIFACT_CREATED, TASK_PAUSED, TASK_RESUMED,
        TASK_CANCELLED, TASK_COMPLETED, TASK_FAILED,
        WORKSPACE_STARTED, WORKSPACE_STOPPED,
        RESOURCE_LIMIT, EMERGENCY_STOP, TASK_STOPPED,
        CONTROL_ISSUED, SUMMARY_GENERATED,
    }
)

# LoopEvent.type → ActivityEvent.type
LOOP_TYPE_MAP = {
    "task_started": TASK_STARTED,
    "observation_received": OBSERVING,
    "decision_created": DECIDING,
    "action_started": ACTION_STARTED,
    "action_completed": ACTION_COMPLETED,
    "verification_completed": VERIFICATION_COMPLETED,
    "recovery_started": RECOVERY_STARTED,
    "replan_started": REPLANNING,
    "approval_required": APPROVAL_REQUIRED,
    "checkpoint_created": CHECKPOINT_CREATED,
    "task_completed": TASK_COMPLETED,
    "task_failed": TASK_FAILED,
    "task_paused": TASK_PAUSED,
    "security_warning": RESOURCE_LIMIT,
    "progress_stalled": WAITING,
}


@dataclass
class ActivityEvent:
    """One user-safe activity record."""

    type: str
    summary: str
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    seq: int = 0  # assigned by ActivityCenter (per-center order)
    at: str = field(default_factory=_now_iso)
    task_id: str = ""
    workspace_id: str = ""
    goal_id: str = ""
    actor: str = "agent"
    source: str = "loop"  # loop|task|audit|artifact|control|system
    details: dict[str, Any] = field(default_factory=dict)
    audit_ref: str = ""  # seq/hash of the AuditCenter record
    evidence_refs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.type not in ALL_TYPES:
            raise ValueError(f"unknown activity type: {self.type!r}")
        # Never let secrets or raw reasoning through.
        self.summary = str(self.summary)[:500]
        self.details = redact_value(dict(self.details or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActivityEvent":
        known = {f for f in cls.__dataclass_fields__}
        return cls(
            **{k: v for k, v in data.items() if k in known}
        )


def from_loop_event(
    loop_event: Any, *, task_id: str = "", workspace_id: str = ""
) -> ActivityEvent:
    """Adapt an AgentLoop LoopEvent (real runtime event)."""
    raw_type = getattr(loop_event, "type", "observation_received")
    details = dict(getattr(loop_event, "details", {}) or {})
    step_id = getattr(loop_event, "step_id", None)
    if step_id:
        details.setdefault("step_id", step_id)
    return ActivityEvent(
        type=LOOP_TYPE_MAP.get(str(raw_type), OBSERVING),
        summary=str(getattr(loop_event, "message", ""))[:500],
        at=str(
            getattr(loop_event, "at", None) or _now_iso()
        ),
        task_id=task_id or str(details.pop("task_id", "")),
        workspace_id=workspace_id
        or str(details.pop("workspace_id", "")),
        source="loop",
        details=details,
    )


def from_audit_record(
    record: dict[str, Any],
) -> ActivityEvent | None:
    """Adapt an AuditLogger record; None if not user-visible."""
    event = str(record.get("event", ""))
    # Only surface security-relevant occurrences; routine
    # allow-logs would spam the timeline.
    mapping = {
        "approval_required": APPROVAL_REQUIRED,
        "approval_granted": APPROVAL_GRANTED,
        "approval_denied": APPROVAL_DENIED,
        "emergency_stop": EMERGENCY_STOP,
        "resource_limit": RESOURCE_LIMIT,
        "policy_denied": ACTION_FAILED,
    }
    activity_type = mapping.get(event)
    if activity_type is None:
        return None
    return ActivityEvent(
        type=activity_type,
        summary=str(record.get("action", event))[:500],
        at=str(record.get("at") or _now_iso()),
        task_id=str(record.get("task_id", "")),
        actor=str(record.get("actor", "agent")),
        source="audit",
        details={
            "risk_level": record.get("risk_level", ""),
            "reason": str(
                record.get("details", {})
            )[:300]
            if isinstance(record.get("details"), dict)
            else "",
        },
        audit_ref=str(
            record.get("hash", "")
        ) or str(record.get("seq", "")),
    )

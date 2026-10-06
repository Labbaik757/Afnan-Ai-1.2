"""Agent status model — the user-facing live state.

Standard live-agent states.  The tracker derives status
from TaskManager + AgentLoop events; the UI never touches
the loop directly.
"""

from __future__ import annotations

IDLE = "IDLE"
PLANNING = "PLANNING"
OBSERVING = "OBSERVING"
EXECUTING = "EXECUTING"
VERIFYING = "VERIFYING"
RECOVERING = "RECOVERING"
WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
PAUSED = "PAUSED"
WAITING = "WAITING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"
STOPPED = "STOPPED"

ALL = frozenset(
    {
        IDLE, PLANNING, OBSERVING, EXECUTING, VERIFYING,
        RECOVERING, WAITING_FOR_APPROVAL, PAUSED, WAITING,
        COMPLETED, FAILED, CANCELLED, STOPPED,
    }
)

# Terminal states: no further transitions expected.
TERMINAL = frozenset({COMPLETED, FAILED, CANCELLED, STOPPED})

# ActivityEvent.type → AgentStatus
_EVENT_STATUS_MAP = {
    "task_created": IDLE,
    "task_started": OBSERVING,
    "planning": PLANNING,
    "observing": OBSERVING,
    "deciding": PLANNING,
    "action_started": EXECUTING,
    "action_completed": EXECUTING,
    "action_failed": RECOVERING,
    "verification_started": VERIFYING,
    "verification_completed": EXECUTING,
    "recovery_started": RECOVERING,
    "replanning": PLANNING,
    "approval_required": WAITING_FOR_APPROVAL,
    "approval_granted": EXECUTING,
    "approval_denied": FAILED,
    "waiting": WAITING,
    "checkpoint_created": PAUSED,
    "task_paused": PAUSED,
    "task_resumed": OBSERVING,
    "task_cancelled": CANCELLED,
    "task_completed": COMPLETED,
    "task_failed": FAILED,
    "task_stopped": STOPPED,
    "emergency_stop": STOPPED,
}


def status_for_event(event_type: str) -> str | None:
    """Map an activity event to a status, if it changes it."""
    return _EVENT_STATUS_MAP.get(event_type)


class AgentStatusTracker:
    """Per-task live status, updated from activity events."""

    def __init__(self) -> None:
        self._status: dict[str, str] = {}
        self._updated_at: dict[str, str] = {}

    def update(
        self, task_id: str, event_type: str, at: str
    ) -> str | None:
        """Apply an event; return new status if it changed."""
        new_status = status_for_event(event_type)
        if new_status is None:
            return None
        old = self._status.get(task_id, IDLE)
        if old in TERMINAL and new_status not in TERMINAL:
            # Never resurrect a finished task via a late event.
            return None
        if new_status != old:
            self._status[task_id] = new_status
            self._updated_at[task_id] = at
            return new_status
        return None

    def get(self, task_id: str) -> str:
        return self._status.get(task_id, IDLE)

    def snapshot(self) -> dict[str, dict[str, str]]:
        return {
            task_id: {
                "status": status,
                "updated_at": self._updated_at.get(task_id, ""),
            }
            for task_id, status in self._status.items()
        }

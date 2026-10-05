"""NotificationService — channel-independent notifications.

Important activity (approvals, completions, failures, stops)
is fanned out to registered channels.  Channels are plain
callables; the service knows nothing about web/mobile/voice
transports, keeping the architecture UI-independent.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from afnan_ai.activity import events as E

Channel = Callable[[dict[str, Any]], None]


@dataclass
class Notification:
    kind: str
    title: str
    body: str = ""
    task_id: str = ""
    severity: str = "info"  # info|warning|critical
    at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ActivityEvent.type → (kind, severity, title template)
_NOTIFY_MAP: dict[str, tuple[str, str, str]] = {
    E.APPROVAL_REQUIRED: (
        "approval_required", "warning", "Approval required"
    ),
    E.TASK_COMPLETED: (
        "task_completed", "info", "Task completed"
    ),
    E.TASK_FAILED: (
        "task_failed", "critical", "Task failed"
    ),
    E.TASK_PAUSED: ("task_paused", "info", "Task paused"),
    E.WAITING: ("task_waiting", "info", "Task waiting"),
    E.RESOURCE_LIMIT: (
        "resource_limit", "warning", "Resource limit"
    ),
    E.EMERGENCY_STOP: (
        "emergency_stop", "critical", "Emergency stop"
    ),
    E.TASK_CANCELLED: (
        "task_cancelled", "info", "Task cancelled"
    ),
}


class NotificationService:
    def __init__(self) -> None:
        self._channels: dict[str, Channel] = {}
        self._lock = threading.RLock()
        self._history: list[Notification] = []

    def add_channel(
        self, name: str, fn: Channel
    ) -> None:
        with self._lock:
            self._channels[name] = fn

    def remove_channel(self, name: str) -> None:
        with self._lock:
            self._channels.pop(name, None)

    def apply(self, event: E.ActivityEvent) -> None:
        """Fold activity events into notifications (bus hook)."""
        mapped = _NOTIFY_MAP.get(event.type)
        if mapped is None:
            return
        kind, severity, title = mapped
        self.notify(
            Notification(
                kind=kind,
                title=title,
                body=event.summary[:300],
                task_id=event.task_id,
                severity=severity,
            )
        )

    def notify(self, note: Notification) -> None:
        with self._lock:
            self._history.append(note)
            self._history = self._history[-500:]
            channels = list(self._channels.values())
        payload = note.to_dict()
        for ch in channels:
            try:
                ch(payload)
            except Exception:
                pass  # channels never break the service

    def history(
        self, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._history[-limit:])
        return [n.to_dict() for n in reversed(items)]

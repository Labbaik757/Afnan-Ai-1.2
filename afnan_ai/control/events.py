"""Real-time event delivery for remote clients.

The stream is a thin projection over the existing ActivityCenter:
``subscribe()`` fans live events out, ``sync(after_seq=...)``
replays what a reconnecting client missed.  No duplicate event
store is created.  Backpressure: slow clients get a bounded buffer;
overflow drops oldest with a counter (never blocks the agent).
"""

from __future__ import annotations

import collections
import threading
import time
import uuid
from typing import Any, Callable

from afnan_ai.control.models import (
    EVENT_CATEGORIES,
    ControlEvent,
)

# ActivityCenter event types -> control plane categories.
_ACTIVITY_TO_CATEGORY = {
    "task_created": "task.created",
    "task_started": "task.started",
    "task_paused": "task.paused",
    "task_resumed": "task.resumed",
    "task_completed": "task.completed",
    "task_failed": "task.failed",
    "task_cancelled": "task.cancelled",
    "approval_required": "approval.requested",
    "approval_granted": "approval.resolved",
    "approval_denied": "approval.resolved",
    "checkpoint_created": "checkpoint.created",
    "artifact_created": "artifact.created",
    "emergency_stop": "security.alert",
}


class EventStream:
    """Per-session event delivery with replay cursors."""

    def __init__(
        self,
        activity_center,
        *,
        buffer_size: int = 1000,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.activity = activity_center
        self._buffer_size = buffer_size
        self._on_event = on_event
        self._lock = threading.RLock()
        self._seq = 0
        # session_id -> deque[ControlEvent]
        self._buffers: dict[
            str, collections.deque[ControlEvent]
        ] = {}
        self._cursors: dict[str, int] = {}
        self._dropped: dict[str, int] = {}
        self._filters: dict[str, set[str] | None] = {}
        self._sub_id: int | None = None
        if activity_center is not None:
            try:
                self._sub_id = activity_center.subscribe(
                    self._on_activity
                )
            except Exception:
                self._sub_id = None

    # -- client attachment ----------------------------------------------------

    def attach(
        self,
        session_id: str,
        *,
        categories: set[str] | None = None,
        from_seq: int = 0,
    ) -> dict[str, Any]:
        """Attach a session; returns missed events for replay."""
        if categories is not None:
            unknown = set(categories) - EVENT_CATEGORIES
            if unknown:
                raise ValueError(
                    f"unknown event categories: {sorted(unknown)}"
                )
        with self._lock:
            self._buffers.setdefault(
                session_id,
                collections.deque(maxlen=self._buffer_size),
            )
            self._cursors.setdefault(session_id, from_seq)
            self._dropped.setdefault(session_id, 0)
            self._filters[session_id] = (
                set(categories) if categories else None
            )
        missed: list[dict[str, Any]] = []
        if self.activity is not None and from_seq:
            try:
                sync_state = self.activity.sync(
                    after_seq=from_seq, limit=500
                )
                for raw in sync_state.get("missed_events", []):
                    event = self._from_activity_dict(raw)
                    if event is not None and self._passes(
                        session_id, event
                    ):
                        missed.append(event.to_dict())
            except Exception:
                pass
        with self._lock:
            cursor = self._cursors.get(session_id, 0)
        return {
            "session_id": session_id,
            "cursor": cursor,
            "missed_events": missed,
            "dropped": self._dropped.get(session_id, 0),
        }

    def detach(self, session_id: str) -> None:
        with self._lock:
            self._buffers.pop(session_id, None)
            self._cursors.pop(session_id, None)
            self._dropped.pop(session_id, None)
            self._filters.pop(session_id, None)

    def poll(
        self, session_id: str, *, acknowledge_up_to: int = 0
    ) -> list[dict[str, Any]]:
        """Drain buffered events for a session (long-poll fallback)."""
        with self._lock:
            buf = self._buffers.get(session_id)
            if buf is None:
                return []
            events = [e.to_dict() for e in buf]
            buf.clear()
            if acknowledge_up_to:
                self._cursors[session_id] = max(
                    self._cursors.get(session_id, 0),
                    acknowledge_up_to,
                )
            dropped = self._dropped.get(session_id, 0)
            self._dropped[session_id] = 0
        if dropped:
            events.append(
                {
                    "event_id": f"evt_{uuid.uuid4().hex[:12]}",
                    "category": "stream.gap",
                    "seq": 0,
                    "at": time.time(),
                    "data": {"dropped": dropped},
                    "protocol_version": "v1",
                }
            )
        return events

    def cursor(self, session_id: str) -> int:
        with self._lock:
            return self._cursors.get(session_id, 0)

    # -- publishing -------------------------------------------------------------

    def publish(
        self,
        category: str,
        data: dict[str, Any] | None = None,
        *,
        device_id: str = "",
        session_id: str = "",
        correlation_id: str = "",
    ) -> ControlEvent | None:
        if category not in EVENT_CATEGORIES:
            return None
        with self._lock:
            self._seq += 1
            event = ControlEvent(
                category=category,
                seq=self._seq,
                device_id=device_id,
                session_id=session_id,
                correlation_id=correlation_id,
                data=dict(data or {}),
            )
            for sid, buf in self._buffers.items():
                if not self._passes(sid, event):
                    continue
                if len(buf) >= self._buffer_size:
                    self._dropped[sid] = (
                        self._dropped.get(sid, 0) + 1
                    )
                buf.append(event)
                self._cursors[sid] = event.seq
        if self._on_event is not None:
            try:
                self._on_event(event.to_dict())
            except Exception:
                pass
        return event

    # -- internals --------------------------------------------------------------

    def _passes(
        self, session_id: str, event: ControlEvent
    ) -> bool:
        wanted = self._filters.get(session_id)
        if not wanted:
            return True
        return event.category in wanted

    def _on_activity(self, stored) -> None:
        """ActivityCenter subscriber: project into control events."""
        try:
            category = _ACTIVITY_TO_CATEGORY.get(
                getattr(stored, "type", "")
            )
            if category is None:
                return
            self.publish(
                category,
                {
                    "summary": str(
                        getattr(stored, "summary", "")
                    )[:300],
                    "task_id": getattr(stored, "task_id", ""),
                    "actor": getattr(stored, "actor", ""),
                },
                correlation_id=getattr(
                    stored, "event_id", ""
                ),
            )
        except Exception:
            pass

    def _from_activity_dict(
        self, raw: dict[str, Any]
    ) -> ControlEvent | None:
        category = _ACTIVITY_TO_CATEGORY.get(
            str(raw.get("type", ""))
        )
        if category is None:
            return None
        with self._lock:
            self._seq += 1
            seq = self._seq
        return ControlEvent(
            event_id=str(raw.get("event_id", "")),
            category=category,
            seq=seq,
            at=time.time(),
            data={
                "summary": str(raw.get("summary", ""))[:300],
                "task_id": str(raw.get("task_id", "")),
            },
        )

    def close(self) -> None:
        if (
            self.activity is not None
            and self._sub_id is not None
        ):
            try:
                self.activity.unsubscribe(self._sub_id)
            except Exception:
                pass
            self._sub_id = None

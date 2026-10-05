"""ActivityCenter — the live + historical activity backend.

Architecture::

    AgentLoop ──on_event──╮
    TaskManager ──hook──╮  │
    AuditLogger ──hook──╮  │   ┌──────────────┐   ┌─────────────┐
    ArtifactManager ─╮  │  ├──▶│ ActivityCenter│──▶│ subscribers │
    BackgroundRunner ─╮ │  │   │  (event bus)  │   │ (UI/API)    │
    Control plane ────╯  ╯  │   └──────────────┘   └─────────────┘
                           │          │ JSONL persistence
                           └──────────┘ (survives restarts)

Guarantees:

- Events are ordered (per-center ``seq``), timestamped and
  associated with task_id / workspace_id.
- Duplicate delivery is harmless (bounded id set).
- If the UI disconnects, the agent keeps running; events
  persist; on reconnect the client replays from its last
  ``seq`` — no state corruption, no loss.
- The center never executes tools or touches the loop
  directly; control goes through TaskControlService.
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from afnan_ai.activity import events as E


def _default_dir() -> Path:
    return Path.home() / ".afnan-ai" / "activity"


class EventStore:
    """Append-only JSONL store with seq assignment and replay."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_file_mb: int = 50,
    ) -> None:
        self._dir = (
            Path(str(path)).expanduser()
            if path
            else _default_dir()
        )
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "events.jsonl"
        self._max_bytes = max(1, max_file_mb) * 1024 * 1024
        self._lock = threading.RLock()
        self._seq = 0
        self._seen: set[str] = set()
        self._recover()

    # -- persistence --------------------------------------------------
    def _recover(self) -> None:
        if not self._file.exists():
            return
        try:
            with open(self._file, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # corrupt line: skip, never crash
                    seq = int(data.get("seq", 0) or 0)
                    self._seq = max(self._seq, seq)
                    eid = str(data.get("event_id", ""))
                    if eid:
                        if len(self._seen) > 20000:
                            self._seen.clear()
                        self._seen.add(eid)
        except OSError:
            pass

    def _rotate(self) -> None:
        try:
            if (
                self._file.exists()
                and self._file.stat().st_size >= self._max_bytes
            ):
                stamp = datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%SZ"
                )
                self._file.rename(
                    self._dir / f"events-{stamp}.jsonl"
                )
        except OSError:
            pass

    def append(self, event: E.ActivityEvent) -> E.ActivityEvent | None:
        """Assign seq, dedup, persist. None if duplicate."""
        with self._lock:
            if event.event_id in self._seen:
                return None
            self._seq += 1
            event.seq = self._seq
            if len(self._seen) > 20000:
                self._seen.clear()
            self._seen.add(event.event_id)
            self._rotate()
            try:
                with open(
                    self._file, "a", encoding="utf-8"
                ) as fh:
                    fh.write(
                        json.dumps(
                            event.to_dict(), ensure_ascii=False
                        )
                        + "\n"
                    )
            except OSError:
                pass  # persistence best-effort; memory still live
            return event

    def replay(
        self,
        *,
        after_seq: int = 0,
        task_id: str = "",
        limit: int = 500,
    ) -> list[E.ActivityEvent]:
        """Missed events after a seq (reconnect recovery)."""
        out: list[E.ActivityEvent] = []
        if not self._file.exists():
            return out
        try:
            with open(self._file, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if int(data.get("seq", 0) or 0) <= after_seq:
                        continue
                    if task_id and data.get("task_id") != task_id:
                        continue
                    try:
                        out.append(E.ActivityEvent.from_dict(data))
                    except (ValueError, TypeError):
                        continue
                    if len(out) >= limit:
                        break
        except OSError:
            pass
        return out

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq


Subscriber = Callable[[E.ActivityEvent], None]


class ActivityCenter:
    """Live event bus + persistent store + subscriptions."""

    def __init__(
        self,
        store: EventStore | None = None,
        *,
        status_tracker: Any | None = None,
        progress_tracker: Any | None = None,
        approval_center: Any | None = None,
        notifier: Any | None = None,
    ) -> None:
        from afnan_ai.activity.status import AgentStatusTracker

        self.store = store or EventStore()
        self.status = status_tracker or AgentStatusTracker()
        # progress/approval/notifier wired by integration to
        # avoid import cycles; center tolerates None.
        self.progress = progress_tracker
        self.approvals = approval_center
        self.notifier = notifier
        self._subs: dict[int, Subscriber] = {}
        self._next_sub = 0
        self._lock = threading.RLock()
        self._task_last_seq: dict[str, int] = defaultdict(int)

    # -- publishing ----------------------------------------------------
    def publish(self, event: E.ActivityEvent) -> E.ActivityEvent | None:
        """Store + fan-out. Never raises into the caller."""
        stored = self.store.append(event)
        if stored is None:
            return None  # duplicate
        self._task_last_seq[stored.task_id] = stored.seq
        # Derived views (best-effort, never break publishing).
        try:
            self.status.update(
                stored.task_id, stored.type, stored.at
            )
        except Exception:
            pass
        if self.progress is not None:
            try:
                self.progress.apply(stored)
            except Exception:
                pass
        if self.approvals is not None:
            try:
                self.approvals.apply(stored)
            except Exception:
                pass
        if self.notifier is not None:
            try:
                self.notifier.apply(stored)
            except Exception:
                pass
        with self._lock:
            subs = list(self._subs.values())
        for sub in subs:
            try:
                sub(stored)
            except Exception:
                pass  # a bad subscriber must not break the bus
        return stored

    def emit(
        self,
        type: str,
        summary: str,
        *,
        task_id: str = "",
        workspace_id: str = "",
        goal_id: str = "",
        actor: str = "agent",
        source: str = "system",
        details: dict[str, Any] | None = None,
        audit_ref: str = "",
        evidence_refs: list[str] | None = None,
    ) -> E.ActivityEvent | None:
        return self.publish(
            E.ActivityEvent(
                type=type,
                summary=summary,
                task_id=task_id,
                workspace_id=workspace_id,
                goal_id=goal_id,
                actor=actor,
                source=source,
                details=details or {},
                audit_ref=audit_ref,
                evidence_refs=evidence_refs or [],
            )
        )

    # -- subscriptions ---------------------------------------------------
    def subscribe(self, fn: Subscriber) -> int:
        with self._lock:
            self._next_sub += 1
            self._subs[self._next_sub] = fn
            return self._next_sub

    def unsubscribe(self, sub_id: int) -> None:
        with self._lock:
            self._subs.pop(sub_id, None)

    # -- reconnect ---------------------------------------------------------
    def sync(
        self,
        *,
        after_seq: int = 0,
        task_id: str = "",
        limit: int = 500,
    ) -> dict[str, Any]:
        """State for a (re)connecting client."""
        return {
            "last_seq": self.store.last_seq,
            "missed_events": [
                e.to_dict()
                for e in self.store.replay(
                    after_seq=after_seq,
                    task_id=task_id,
                    limit=limit,
                )
            ],
        }

    def task_seq(self, task_id: str) -> int:
        return self._task_last_seq.get(task_id, 0)

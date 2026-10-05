"""BackgroundTaskRunner — persistent background execution.

Executes TaskManager tasks (including Scheduler-produced
ones) through the existing AgentLoop: Observe → Decide →
Validate → Execute → Verify.  No orchestration is duplicated
here — the runner owns only claim/execute/record, the loop
owns thinking.

Safety is inherited, never weakened:

* sensitive actions still stop at the human-approval gate;
  an approval-needed outcome parks the task as
  ``waiting_for_approval`` instead of executing;
* per-task timeout, step and retry limits flow into the
  loop's LoopLimits; a task's retry budget is enforced by
  the TaskManager, and the loop's own stall/repeat guards
  prevent blind repetition;
* the runner starts nothing by itself: :meth:`start` is an
  explicit caller action, :meth:`process_available` runs a
  synchronous drain for tests/cron-style use, and
  :meth:`stop` shuts down cleanly.

Persistence: on start, tasks left ``running`` by a dead
process are recovered as paused, then re-queued *with their
checkpoint reference* so the loop resumes from the last
valid checkpoint and completed steps are never re-executed.
Corrupted checkpoints are detected by the CheckpointManager
(checksum) and treated as "start fresh, note it in the
audit trail" — never trusted.

Every meaningful transition is recorded in a redacted
structured audit trail (JSONL): secrets and sensitive page
data are scrubbed by the shared redactor before anything is
written.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from afnan_ai.redaction import redact_text, redact_value

APPROVAL_CODES = frozenset({
    "approval_required", "approval_denied",
})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLog:
    """Append-only JSONL audit trail (redacted)."""

    def __init__(self, path: str | None = None):
        self.path = Path(path).expanduser() if path else None
        self._lock = threading.Lock()

    def record(
        self, event_type: str, *, task_id: str = "",
        message: str = "", **details: Any,
    ) -> dict[str, Any]:
        entry = {
            "at": _now_iso(),
            "type": str(event_type),
            "task_id": str(task_id),
            "message": redact_text(str(message))[:500],
            "details": redact_value(dict(details)),
        }
        if self.path is not None:
            try:
                line = json.dumps(entry, ensure_ascii=False)
                with self._lock:
                    self.path.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    with self.path.open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(line + "\n")
            except OSError:
                pass  # auditing must never break execution
        return entry


class BackgroundTaskRunner:
    """Claim → execute → record over the persistent queue."""

    def __init__(
        self,
        task_manager: Any,
        run_task: Callable[
            [Any, dict[str, Any] | None], Any
        ],
        *,
        scheduler: Any = None,
        checkpointer: Any = None,
        audit_log: AuditLog | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        max_concurrent: int = 1,
        poll_interval_s: float = 1.0,
        security_center: Any = None,
    ):
        self.task_manager = task_manager
        self.run_task = run_task
        self.scheduler = scheduler
        self.checkpointer = checkpointer
        self.audit = audit_log or AuditLog()
        self.on_event = on_event
        self.max_concurrent = max(1, int(max_concurrent))
        self.poll_interval_s = max(0.05, float(poll_interval_s))
        self.events: deque[dict[str, Any]] = deque(maxlen=500)
        self._lock = threading.RLock()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._started = False
        # Central security: permission snapshots + policy
        # revalidation after restarts.  Snapshots live in
        # task metadata (capability ids + version only —
        # never secrets).
        self._security = security_center
        if security_center is not None:
            try:
                security_center.emergency.register(
                    self.stop
                )
            except Exception:
                pass

    # -- events ---------------------------------------------------------
    def _emit(
        self, event_type: str, *, task_id: str = "",
        message: str = "", **details: Any,
    ) -> None:
        entry = self.audit.record(
            event_type, task_id=task_id,
            message=message, **details,
        )
        with self._lock:
            self.events.append(entry)
        if self.on_event is not None:
            try:
                self.on_event(entry)
            except Exception:
                pass  # listeners never break the runner

    # -- permission snapshots & policy revalidation ----------
    def _snapshot_permissions(self, task: Any) -> dict[str, Any]:
        center = self._security
        if center is None:
            return {}
        actor = getattr(center, "agent_actor", None)
        label = getattr(actor, "label", "agent:main")
        capabilities = sorted(
            center.permissions.capabilities_for(label)
        )
        current = center.versions.current()
        return {
            "capabilities": capabilities,
            "policy_version": (
                current.version_id if current else ""
            ),
        }

    def _revalidate_policy(self, task: Any) -> Any | None:
        """Return a failure outcome when the task must not
        run under the current policy; otherwise snapshot the
        current permissions and return None."""
        center = self._security
        if center is None:
            return None
        meta = getattr(task, "metadata", None) or {}
        old = (
            meta.get("permission_snapshot")
            if isinstance(meta, dict) else None
        )
        snapshot = self._snapshot_permissions(task)
        try:
            task.metadata["permission_snapshot"] = snapshot
            self.task_manager._save()
        except Exception:
            pass
        current_version = snapshot.get("policy_version", "")
        if (
            isinstance(old, dict)
            and old.get("policy_version")
            and old.get("policy_version") != current_version
        ):
            # Policy changed since this task was queued:
            # re-check that the capabilities it was given are
            # still held.
            revoked = [
                c for c in (old.get("capabilities") or [])
                if not center.permissions.check(
                    getattr(
                        center.agent_actor, "label",
                        "agent:main",
                    ),
                    c,
                )
            ]
            self._emit(
                "policy_revalidated",
                task_id=task.task_id,
                message=(
                    "Policy changed since queueing; "
                    "capabilities re-checked"
                ),
                policy_version=current_version,
                revoked=revoked,
            )
            if revoked:
                self._emit(
                    "task_policy_revoked",
                    task_id=task.task_id,
                    message=(
                        "Capabilities revoked by policy "
                        "change; task will not run"
                    ),
                    revoked=revoked,
                )
                return self._record_failure(
                    task,
                    "Capabilities revoked by policy change: "
                    + ", ".join(revoked),
                    code="policy_revoked",
                )
        return None

    # -- recovery ---------------------------------------------------------
    def recover(self) -> list[Any]:
        """Prepare the queue after a (re)start.

        Tasks a dead process left ``running`` become paused
        (via the TaskManager); crashed ones are then safely
        re-queued with their checkpoint reference intact so
        the loop can resume.  Human-paused tasks stay paused.
        """
        recovered = self.task_manager.recover_on_startup()
        requeued = []
        for task in recovered:
            task.metadata.pop("crashed", None)
            self.task_manager._save()
            try:
                self.task_manager.resume(task.task_id)
                requeued.append(task)
            except Exception:
                continue
            self._emit(
                "task_recovered",
                task_id=task.task_id,
                message=(
                    "Recovered after restart; resuming from "
                    "checkpoint if available"
                ),
                checkpoint_ref=bool(task.checkpoint_ref),
            )
        return requeued

    # -- execution -----------------------------------------------------
    def _load_resume(self, task: Any) -> dict[str, Any] | None:
        if not task.checkpoint_ref or self.checkpointer is None:
            return None
        try:
            return self.checkpointer.load(task.checkpoint_ref)
        except Exception as exc:
            self._emit(
                "checkpoint_invalid",
                task_id=task.task_id,
                message=(
                    "Checkpoint could not be validated "
                    f"({exc}); starting fresh"
                ),
            )
            return None

    def _execute_claimed(self, task: Any) -> Any:
        # Permission snapshot + policy revalidation: a task
        # claimed after a restart (or a policy change) must
        # re-check that its capabilities are still granted
        # under the current policy version.
        revalidation = self._revalidate_policy(task)
        if revalidation is not None:
            return revalidation
        resume_from = self._load_resume(task)
        self._emit(
            "task_started",
            task_id=task.task_id,
            message=task.goal_text[:160],
            resumed=resume_from is not None,
            attempt=task.attempts,
        )
        try:
            result = self.run_task(task, resume_from)
        except Exception as exc:
            return self._record_failure(
                task, f"Runner error: {exc}"
            )
        for event in getattr(result, "events", None) or []:
            try:
                payload = (
                    event.to_dict()
                    if hasattr(event, "to_dict")
                    else {"type": str(event)}
                )
            except Exception:
                continue
            self._emit(
                f"loop_{payload.get('type', 'event')}",
                task_id=task.task_id,
                message=str(payload.get("message", ""))[:300],
            )
        status = getattr(result, "status", None)
        status_value = getattr(status, "value", status)
        error = getattr(result, "error", None) or {}
        if status_value == "completed":
            state = getattr(result, "state", None)
            summary = (
                f"Task {getattr(state, 'task_id', task.task_id)} "
                "completed"
            )
            with self._lock:
                done = self.task_manager.complete(
                    task.task_id, summary
                )
            self._emit(
                "task_completed", task_id=task.task_id,
                message=summary,
            )
            return done
        code = str(error.get("code") or "")
        if code in APPROVAL_CODES:
            with self._lock:
                parked = self.task_manager.wait_for_approval(
                    task.task_id
                )
            self._emit(
                "task_waiting_approval",
                task_id=task.task_id,
                message=(
                    "Sensitive action needs human approval; "
                    "task parked safely"
                ),
                code=code,
            )
            return parked
        message = str(
            error.get("message") or f"Task ended {status_value}"
        )
        return self._record_failure(task, message, code=code)

    def _record_failure(
        self, task: Any, message: str, *, code: str = ""
    ) -> Any:
        with self._lock:
            updated = self.task_manager.fail(task.task_id, message)
        if updated.status == "pending":
            self._emit(
                "task_retry_scheduled",
                task_id=task.task_id,
                message=message,
                attempt=updated.attempts,
                max_retries=updated.max_retries,
                code=code,
            )
        else:
            self._emit(
                "task_failed",
                task_id=task.task_id,
                message=message,
                attempts=updated.attempts,
                code=code,
            )
        return updated

    def process_available(
        self, max_tasks: int | None = None
    ) -> list[Any]:
        """Synchronous drain: fire due schedules, then run
        pending tasks (bounded by max_concurrent at a time)
        until the queue is empty.  Used by tests and
        cron-style invocation."""
        if self.scheduler is not None:
            created = self.scheduler.tick()
            for task in created:
                self._emit(
                    "task_scheduled",
                    task_id=task.task_id,
                    message=task.goal_text[:160],
                )
        finished: list[Any] = []
        while max_tasks is None or len(finished) < max_tasks:
            claimed = []
            for _ in range(self.max_concurrent):
                if max_tasks is not None and (
                    len(finished) + len(claimed) >= max_tasks
                ):
                    break
                task = self.task_manager.claim_next()
                if task is None:
                    break
                claimed.append(task)
            if not claimed:
                break
            if len(claimed) == 1:
                finished.append(self._execute_claimed(claimed[0]))
            else:
                threads = []
                for task in claimed:
                    thread = threading.Thread(
                        target=self._execute_claimed,
                        args=(task,), daemon=True,
                    )
                    threads.append((task, thread))
                    thread.start()
                for task, thread in threads:
                    thread.join()
                    finished.append(
                        self.task_manager.get(task.task_id)
                    )
        return finished

    def resume_task(self, task_id: str) -> Any | None:
        """Resume a paused / waiting-for-approval task now
        (used right after a human approval)."""
        task = self.task_manager.get(task_id)
        if task is None or task.status not in (
            "paused", "waiting_for_approval",
        ):
            return None
        # Claim exactly this task (never a different one).
        task.attempts += 1
        task.started_at = _now_iso()
        task.status = "running"
        task.updated_at = _now_iso()
        self.task_manager._save()
        self._emit(
            "task_resumed", task_id=task.task_id,
            message="Task resumed by request",
        )
        return self._execute_claimed(task)

    # -- threaded lifecycle ---------------------------------------------
    def start(self) -> None:
        """Start the background loop (explicit caller action)."""
        with self._lock:
            if self._started:
                return
            self._started = True
        self._stop.clear()
        self.recover()
        worker = threading.Thread(
            target=self._main_loop,
            name="afnan-background-runner",
            daemon=True,
        )
        self._threads = [worker]
        worker.start()
        self._emit("runner_started", message="Background runner started")

    def _main_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.process_available(max_tasks=self.max_concurrent)
            except Exception as exc:
                self._emit(
                    "runner_error", message=f"Cycle error: {exc}"
                )
            self._stop.wait(self.poll_interval_s)

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=timeout_s)
        self._threads = []
        with self._lock:
            self._started = False
        self._emit("runner_stopped", message="Background runner stopped")

    @property
    def running(self) -> bool:
        return self._started and not self._stop.is_set()

    def status(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for task in self.task_manager.list():
            counts[task.status] = counts.get(task.status, 0) + 1
        schedules = (
            len(self.scheduler.list(status="active"))
            if self.scheduler is not None
            else 0
        )
        return {
            "running": self.running,
            "tasks": counts,
            "active_schedules": schedules,
            "recent_events": list(self.events)[-10:],
        }

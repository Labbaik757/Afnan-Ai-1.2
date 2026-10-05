"""TaskManager + TaskWorker — persistent long-running tasks.

Tasks are the units of work that serve goals.  They live in a
persistent queue with an explicit lifecycle::

    pending → running → completed | failed
                  ↘ paused → pending (resume)
                  ↘ waiting_for_approval → pending (resume)
    any → cancelled

Every task carries its retry budget, timeout, checkpoint
reference and verified result summary, so a process restart
never silently loses or blindly re-runs work: tasks found in
``running`` after a restart are recovered as ``paused``
(resumable from their checkpoint), never auto-executed.

The :class:`TaskWorker` is deliberately a foundation, not an
autonomous daemon: it only runs when explicitly invoked
(``run_once`` / ``run_pending`` / ``resume_task``), executes
through an injected runner (the AgentLoop), preserves the
human-approval restrictions (an approval-needed outcome parks
the task as ``waiting_for_approval``), and starts no threads
or background execution on its own.  A future scheduler calls
the worker; the worker never calls itself.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from afnan_ai.persistence import JsonFileStore

TASK_STATUSES = frozenset({
    "pending", "running", "paused", "waiting_for_approval",
    "completed", "failed", "cancelled",
})

_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"running", "cancelled", "paused"}),
    "running": frozenset({
        "completed", "failed", "paused", "waiting_for_approval",
        "cancelled", "pending",
    }),
    "paused": frozenset({"pending", "cancelled"}),
    "waiting_for_approval": frozenset({"pending", "cancelled"}),
    "completed": frozenset(),
    "failed": frozenset({"pending"}),
    "cancelled": frozenset(),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ManagedTask:
    task_id: str
    goal_text: str
    status: str = "pending"
    goal_id: str = ""  # ManagedGoal this task serves (optional)
    priority: int = 3
    attempts: int = 0
    max_retries: int = 2
    timeout_s: float | None = None
    checkpoint_ref: str = ""
    result_summary: str = ""
    error: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal_text": self.goal_text,
            "status": self.status,
            "goal_id": self.goal_id,
            "priority": self.priority,
            "attempts": self.attempts,
            "max_retries": self.max_retries,
            "timeout_s": self.timeout_s,
            "checkpoint_ref": self.checkpoint_ref,
            "result_summary": self.result_summary,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManagedTask":
        return cls(
            task_id=str(data.get("task_id", "")),
            goal_text=str(data.get("goal_text", "")),
            status=str(data.get("status", "pending")),
            goal_id=str(data.get("goal_id", "")),
            priority=int(data.get("priority", 3)),
            attempts=int(data.get("attempts", 0)),
            max_retries=int(data.get("max_retries", 2)),
            timeout_s=data.get("timeout_s"),
            checkpoint_ref=str(data.get("checkpoint_ref", "")),
            result_summary=str(data.get("result_summary", "")),
            error=str(data.get("error", "")),
            created_at=str(data.get("created_at", _now())),
            updated_at=str(data.get("updated_at", _now())),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            metadata=dict(data.get("metadata") or {}),
        )


class TaskManager:
    """Persistent task queue over a local JSON store."""

    def __init__(self, path: str):
        self._store = JsonFileStore(path)
        self._tasks: dict[str, ManagedTask] | None = None

    # -- storage ---------------------------------------------------
    def _load(self) -> dict[str, ManagedTask]:
        if self._tasks is None:
            data = self._store.read({"version": 1, "tasks": []})
            tasks: dict[str, ManagedTask] = {}
            for item in data.get("tasks") or []:
                try:
                    task = ManagedTask.from_dict(item)
                except Exception:
                    continue
                if task.task_id:
                    tasks[task.task_id] = task
            self._tasks = tasks
        return self._tasks

    def _save(self) -> None:
        self._store.write({
            "version": 1,
            "tasks": [t.to_dict() for t in self._load().values()],
        })

    def _require(self, task_id: str) -> ManagedTask:
        task = self._load().get(task_id)
        if task is None:
            raise TaskError(
                "not_found", f"No task with id {task_id!r}"
            )
        return task

    def _transition(self, task: ManagedTask, status: str) -> ManagedTask:
        allowed = _TRANSITIONS.get(task.status, frozenset())
        if status not in allowed:
            raise TaskError(
                "invalid_transition",
                f"Task {task.task_id} cannot move from "
                f"{task.status} to {status}",
            )
        task.status = status
        task.updated_at = _now()
        self._save()
        return task

    # -- queue -------------------------------------------------------
    def enqueue(
        self,
        goal_text: str,
        *,
        goal_id: str = "",
        priority: int = 3,
        max_retries: int = 2,
        timeout_s: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ManagedTask:
        goal_text = str(goal_text or "").strip()
        if not goal_text:
            raise TaskError(
                "empty_goal", "A task needs a goal description"
            )
        task = ManagedTask(
            task_id=f"task_{uuid.uuid4().hex[:12]}",
            goal_text=goal_text,
            goal_id=str(goal_id or ""),
            priority=max(1, min(5, int(priority))),
            max_retries=max(0, int(max_retries)),
            timeout_s=timeout_s,
            metadata=dict(metadata or {}),
        )
        self._load()[task.task_id] = task
        self._save()
        return task

    def get(self, task_id: str) -> ManagedTask | None:
        return self._load().get(task_id)

    def list(
        self, *, status: str | None = None
    ) -> list[ManagedTask]:
        tasks = list(self._load().values())
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        return sorted(
            tasks, key=lambda t: (t.priority, t.created_at)
        )

    def claim_next(self) -> ManagedTask | None:
        """Move the highest-priority pending task to running."""
        pending = self.list(status="pending")
        if not pending:
            return None
        task = pending[0]
        task.attempts += 1
        task.started_at = _now()
        self._transition(task, "running")
        return task

    # -- lifecycle -----------------------------------------------------
    def complete(self, task_id: str, summary: str = "") -> ManagedTask:
        task = self._require(task_id)
        task.result_summary = str(summary)[:1000]
        task.finished_at = _now()
        return self._transition(task, "completed")

    def fail(
        self, task_id: str, error: str, *, retry: bool = True
    ) -> ManagedTask:
        """Fail a task, requeueing it while retries remain."""
        task = self._require(task_id)
        task.error = str(error)[:500]
        if retry and task.attempts <= task.max_retries:
            task.finished_at = None
            return self._transition(task, "pending")
        task.finished_at = _now()
        return self._transition(task, "failed")

    def pause(self, task_id: str) -> ManagedTask:
        return self._transition(self._require(task_id), "paused")

    def wait_for_approval(self, task_id: str) -> ManagedTask:
        return self._transition(
            self._require(task_id), "waiting_for_approval"
        )

    def resume(self, task_id: str) -> ManagedTask:
        """Paused/waiting/failed tasks go back to pending."""
        return self._transition(self._require(task_id), "pending")

    def cancel(self, task_id: str) -> ManagedTask:
        task = self._require(task_id)
        task.finished_at = _now()
        return self._transition(task, "cancelled")

    def record_checkpoint(
        self, task_id: str, checkpoint_ref: str
    ) -> ManagedTask:
        task = self._require(task_id)
        task.checkpoint_ref = str(checkpoint_ref)
        task.updated_at = _now()
        self._save()
        return task

    def recover_on_startup(self) -> list[ManagedTask]:
        """Recover tasks left ``running`` by a dead process.

        They become ``paused`` — resumable from their
        checkpoint — and are never auto-executed.
        """
        recovered = []
        for task in self._load().values():
            if task.status == "running":
                task.error = (
                    "Process restarted while this task was "
                    "running; paused for safe resume"
                )
                task.status = "paused"
                task.updated_at = _now()
                recovered.append(task)
        if recovered:
            self._save()
        return recovered


#: runner(task, resume_from) -> result with .status and .error
TaskRunner = Callable[[ManagedTask, dict[str, Any] | None], Any]


class TaskWorker:
    """Explicitly-invoked executor over the task queue.

    Runs tasks one at a time through the injected runner (the
    AgentLoop).  Approval-needed outcomes park the task as
    ``waiting_for_approval``; failures follow the task's retry
    budget.  Starts nothing by itself — a future scheduler is
    expected to call :meth:`run_pending`.
    """

    def __init__(
        self,
        manager: TaskManager,
        runner: TaskRunner,
        *,
        checkpointer: Any = None,
    ):
        self.manager = manager
        self.runner = runner
        self.checkpointer = checkpointer

    def run_once(self) -> ManagedTask | None:
        task = self.manager.claim_next()
        if task is None:
            return None
        return self._execute(task, resume_from=None)

    def run_pending(
        self, max_tasks: int | None = None
    ) -> list[ManagedTask]:
        done: list[ManagedTask] = []
        while max_tasks is None or len(done) < max_tasks:
            task = self.run_once()
            if task is None:
                break
            done.append(task)
        return done

    def resume_task(self, task_id: str) -> ManagedTask | None:
        """Resume a paused/waiting task from its checkpoint."""
        task = self.manager.get(task_id)
        if task is None or task.status not in (
            "paused", "waiting_for_approval",
        ):
            return None
        resume_from = None
        if task.checkpoint_ref and self.checkpointer is not None:
            try:
                resume_from = self.checkpointer.load(
                    task.checkpoint_ref
                )
            except Exception:
                resume_from = None  # best-effort; run fresh
        task.attempts += 1
        task.started_at = _now()
        # Direct status set: resume may come from waiting states
        # that only allow pending; claim manually.
        task.status = "running"
        task.updated_at = _now()
        self.manager._save()
        return self._execute(task, resume_from=resume_from)

    def _execute(
        self, task: ManagedTask, resume_from: dict[str, Any] | None
    ) -> ManagedTask:
        try:
            result = self.runner(task, resume_from)
        except Exception as exc:  # a runner crash is a task failure
            return self.manager.fail(task.task_id, f"Runner error: {exc}")
        status = getattr(result, "status", None)
        status_value = getattr(status, "value", status)
        error = getattr(result, "error", None) or {}
        if status_value == "completed":
            summary = ""
            state = getattr(result, "state", None)
            if state is not None:
                try:
                    summary = (
                        f"Task {state.task_id} completed for "
                        f"goal: {task.goal_text}"
                    )
                except Exception:
                    summary = ""
            return self.manager.complete(task.task_id, summary)
        if error.get("code") in (
            "approval_required", "approval_denied", "loop_paused",
        ):
            return self.manager.wait_for_approval(task.task_id)
        return self.manager.fail(
            task.task_id,
            str(error.get("message") or f"Task ended {status_value}"),
        )

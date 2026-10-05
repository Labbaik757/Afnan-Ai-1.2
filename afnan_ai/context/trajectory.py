"""TrajectoryStore — persistent per-task execution trajectories.

Every task's trajectory (observations, decisions, actions,
results, verifications, recoveries, approvals, checkpoints)
is recorded here, associated with the task id, and persisted
to a JSON file.  After a restart the trajectory is recovered
so the agent continues from evidence instead of from zero.

Retention policy: per-task entries are capped
(``max_trajectory_entries``), and only the most recent
``retention_tasks`` finished tasks are kept — long-running
agents cannot grow this file without bound.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from afnan_ai.context.models import ContextBudget, TrajectoryEntry
from afnan_ai.log_config import get_logger
from afnan_ai.persistence import JsonFileStore
from afnan_ai.redaction import redact_text

logger = get_logger(__name__)


class TrajectoryStore:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        budget: ContextBudget | None = None,
    ) -> None:
        self.path = (
            Path(str(path)).expanduser() if path else None
        )
        self.budget = budget or ContextBudget()
        # task_id -> {"goal": str, "status": str, "entries": [...]}
        self._tasks: dict[str, dict[str, Any]] = {}
        if self.path is not None:
            self._load()

    # -- recording ------------------------------------------------------
    def record(
        self,
        task_id: str,
        entry: TrajectoryEntry | dict[str, Any],
        *,
        goal: str = "",
    ) -> TrajectoryEntry:
        if isinstance(entry, dict):
            record = TrajectoryEntry.from_dict(entry)
        else:
            record = entry
        # Redaction at the boundary: summaries never carry secrets.
        record.summary = redact_text(record.summary)[:500]
        task = self._tasks.get(task_id)
        if task is None:
            task = {
                "goal": goal,
                "status": "running",
                "entries": [],
            }
            self._tasks[task_id] = task
        entries: list[dict[str, Any]] = task["entries"]
        record.seq = len(entries) + 1
        entries.append(record.to_dict())
        # Per-task cap: drop oldest non-checkpoint entries first.
        cap = self.budget.max_trajectory_entries
        while len(entries) > cap:
            for index, existing in enumerate(entries):
                if existing.get("kind") != "checkpoint":
                    entries.pop(index)
                    break
            else:  # only checkpoints left; drop the oldest
                entries.pop(0)
        # Renumber for stable seqs.
        for index, existing in enumerate(entries):
            existing["seq"] = index + 1
        self._save()
        return record

    def set_status(self, task_id: str, status: str) -> None:
        task = self._tasks.get(task_id)
        if task is not None:
            task["status"] = status
            self._save()

    def entries(
        self, task_id: str, *, limit: int | None = None
    ) -> list[TrajectoryEntry]:
        task = self._tasks.get(task_id)
        if task is None:
            return []
        stored = task["entries"]
        if limit is not None:
            stored = stored[-limit:]
        return [TrajectoryEntry.from_dict(e) for e in stored]

    def task_ids(self) -> list[str]:
        return sorted(self._tasks)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        return {
            "goal": task["goal"],
            "status": task["status"],
            "entries": list(task["entries"]),
        }

    def recover(self, task_id: str) -> list[TrajectoryEntry]:
        """Recover a task's trajectory after a restart."""
        self._load()  # re-read: another process may have written
        return self.entries(task_id)

    # -- persistence ------------------------------------------------------
    def _load(self) -> None:
        if self.path is None:
            return
        try:
            store = JsonFileStore(str(self.path))
            data = store.read({})
        except Exception as e:  # corrupt file -> start empty
            logger.warning(
                "trajectory store unreadable (%s): %s", self.path, e
            )
            return
        tasks = data.get("tasks") if isinstance(data, dict) else None
        if isinstance(tasks, dict):
            # Keep only well-formed task records.
            self._tasks = {
                str(task_id): {
                    "goal": str(task.get("goal", "")),
                    "status": str(task.get("status", "running")),
                    "entries": [
                        e for e in (task.get("entries") or [])
                        if isinstance(e, dict)
                    ][: self.budget.max_trajectory_entries],
                }
                for task_id, task in tasks.items()
                if isinstance(task, dict)
            }

    def _save(self) -> None:
        if self.path is None:
            return
        # Retention: keep the most recent finished tasks plus
        # every task still marked running.
        running = {
            tid: task
            for tid, task in self._tasks.items()
            if task.get("status") == "running"
        }
        finished = [
            (tid, task)
            for tid, task in self._tasks.items()
            if task.get("status") != "running"
        ]
        keep_finished = dict(
            finished[-self.budget.retention_tasks :]
        )
        pruned = {**running, **keep_finished}
        if len(pruned) != len(self._tasks):
            logger.info(
                "trajectory retention: %d -> %d tasks",
                len(self._tasks), len(pruned),
            )
            self._tasks = pruned
        try:
            store = JsonFileStore(str(self.path))
            store.write({"tasks": self._tasks})
        except Exception as e:  # persistence never breaks runs
            logger.warning(
                "trajectory store write failed (%s): %s",
                self.path, e,
            )

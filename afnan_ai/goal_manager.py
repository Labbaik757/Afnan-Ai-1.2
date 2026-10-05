"""GoalManager — persistent, long-lived goals.

Goals outlive tasks: a goal has a description, status,
priority, milestones, dependencies, completion criteria and
the tasks that served it.  Progress is computed from verified
milestones (never from claims), and the AgentLoop loads the
active goals on every run so work stays pointed at what
actually matters.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from afnan_ai.persistence import JsonFileStore

GOAL_STATUSES = frozenset({
    "active", "paused", "completed", "cancelled",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GoalError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Milestone:
    title: str
    done: bool = False
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "done": self.done,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Milestone":
        return cls(
            title=str(data.get("title", "")),
            done=bool(data.get("done", False)),
            completed_at=data.get("completed_at"),
        )


@dataclass
class ManagedGoal:
    goal_id: str
    description: str
    status: str = "active"
    priority: int = 3  # 1 (highest) .. 5
    milestones: list[Milestone] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    completion_criteria: str = ""
    task_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        """Verified progress: completed milestones / total."""
        if self.status == "completed":
            return 1.0
        if not self.milestones:
            return 0.0
        done = sum(1 for m in self.milestones if m.done)
        return round(done / len(self.milestones), 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "milestones": [m.to_dict() for m in self.milestones],
            "dependencies": list(self.dependencies),
            "completion_criteria": self.completion_criteria,
            "task_ids": list(self.task_ids),
            "progress": self.progress,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManagedGoal":
        return cls(
            goal_id=str(data.get("goal_id", "")),
            description=str(data.get("description", "")),
            status=str(data.get("status", "active")),
            priority=int(data.get("priority", 3)),
            milestones=[
                Milestone.from_dict(m)
                for m in data.get("milestones") or []
            ],
            dependencies=[
                str(d) for d in data.get("dependencies") or []
            ],
            completion_criteria=str(
                data.get("completion_criteria", "")
            ),
            task_ids=[str(t) for t in data.get("task_ids") or []],
            created_at=str(data.get("created_at", _now())),
            updated_at=str(data.get("updated_at", _now())),
            metadata=dict(data.get("metadata") or {}),
        )


class GoalManager:
    """Persistent goal lifecycle over a local JSON store."""

    def __init__(self, path: str):
        self._store = JsonFileStore(path)
        self._goals: dict[str, ManagedGoal] | None = None

    # -- storage -----------------------------------------------------
    def _load(self) -> dict[str, ManagedGoal]:
        if self._goals is None:
            data = self._store.read({"version": 1, "goals": []})
            goals: dict[str, ManagedGoal] = {}
            for item in data.get("goals") or []:
                try:
                    goal = ManagedGoal.from_dict(item)
                except Exception:
                    continue
                if goal.goal_id:
                    goals[goal.goal_id] = goal
            self._goals = goals
        return self._goals

    def _save(self) -> None:
        self._store.write({
            "version": 1,
            "goals": [g.to_dict() for g in self._load().values()],
        })

    def _require(self, goal_id: str) -> ManagedGoal:
        goal = self._load().get(goal_id)
        if goal is None:
            raise GoalError(
                "not_found", f"No goal with id {goal_id!r}"
            )
        return goal

    # -- lifecycle -----------------------------------------------------
    def create_goal(
        self,
        description: str,
        *,
        priority: int = 3,
        completion_criteria: str = "",
        milestones: list[str] | None = None,
        dependencies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ManagedGoal:
        description = str(description or "").strip()
        if not description:
            raise GoalError(
                "empty_description", "A goal needs a description"
            )
        goal = ManagedGoal(
            goal_id=f"goal_{uuid.uuid4().hex[:12]}",
            description=description,
            priority=max(1, min(5, int(priority))),
            completion_criteria=str(completion_criteria or ""),
            milestones=[
                Milestone(title=str(m)) for m in milestones or []
            ],
            dependencies=[
                str(d) for d in dependencies or []
            ],
            metadata=dict(metadata or {}),
        )
        self._load()[goal.goal_id] = goal
        self._save()
        return goal

    def get(self, goal_id: str) -> ManagedGoal | None:
        return self._load().get(goal_id)

    def list(
        self, *, status: str | None = None
    ) -> list[ManagedGoal]:
        goals = list(self._load().values())
        if status is not None:
            goals = [g for g in goals if g.status == status]
        return sorted(
            goals, key=lambda g: (g.priority, g.created_at)
        )

    def active_goals(self) -> list[ManagedGoal]:
        """Goals the AgentLoop should keep in view."""
        return self.list(status="active")

    def _set_status(self, goal_id: str, status: str) -> ManagedGoal:
        goal = self._require(goal_id)
        goal.status = status
        goal.updated_at = _now()
        self._save()
        return goal

    def pause(self, goal_id: str) -> ManagedGoal:
        return self._set_status(goal_id, "paused")

    def resume(self, goal_id: str) -> ManagedGoal:
        return self._set_status(goal_id, "active")

    def complete(self, goal_id: str) -> ManagedGoal:
        goal = self._set_status(goal_id, "completed")
        for milestone in goal.milestones:
            if not milestone.done:
                milestone.done = True
                milestone.completed_at = milestone.completed_at or _now()
        goal.updated_at = _now()
        self._save()
        return goal

    def cancel(self, goal_id: str) -> ManagedGoal:
        return self._set_status(goal_id, "cancelled")

    def update(
        self,
        goal_id: str,
        *,
        description: str | None = None,
        priority: int | None = None,
        completion_criteria: str | None = None,
    ) -> ManagedGoal:
        goal = self._require(goal_id)
        if description is not None:
            goal.description = str(description)
        if priority is not None:
            goal.priority = max(1, min(5, int(priority)))
        if completion_criteria is not None:
            goal.completion_criteria = str(completion_criteria)
        goal.updated_at = _now()
        self._save()
        return goal

    # -- milestones & tasks --------------------------------------------
    def add_milestone(
        self, goal_id: str, title: str
    ) -> ManagedGoal:
        goal = self._require(goal_id)
        goal.milestones.append(Milestone(title=str(title)))
        goal.updated_at = _now()
        self._save()
        return goal

    def complete_milestone(
        self, goal_id: str, title: str
    ) -> ManagedGoal:
        goal = self._require(goal_id)
        for milestone in goal.milestones:
            if milestone.title == title and not milestone.done:
                milestone.done = True
                milestone.completed_at = _now()
                break
        else:
            raise GoalError(
                "milestone_not_found",
                f"Goal {goal_id!r} has no open milestone "
                f"{title!r}",
            )
        goal.updated_at = _now()
        self._save()
        return goal

    def link_task(self, goal_id: str, task_id: str) -> ManagedGoal:
        goal = self._require(goal_id)
        if task_id not in goal.task_ids:
            goal.task_ids.append(str(task_id))
        goal.updated_at = _now()
        self._save()
        return goal

    def record_task_result(
        self, goal_id: str, success: bool, summary: str
    ) -> ManagedGoal:
        """Record a verified task outcome against the goal."""
        goal = self._require(goal_id)
        goal.metadata["last_task_result"] = {
            "success": bool(success),
            "summary": str(summary)[:300],
            "at": _now(),
        }
        goal.updated_at = _now()
        self._save()
        return goal

    def dependencies_satisfied(self, goal_id: str) -> bool:
        goal = self._require(goal_id)
        goals = self._load()
        return all(
            dep in goals and goals[dep].status == "completed"
            for dep in goal.dependencies
        )

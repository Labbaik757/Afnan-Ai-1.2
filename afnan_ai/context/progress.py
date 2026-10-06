"""Progress intelligence: honest progress from verified state.

Progress is computed from *verified* completed milestones,
remaining dependencies, blockers, failed attempts and
pending approvals — never a fake percentage.  When the
denominator is unknown, progress is reported as counts,
not a ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afnan_ai.context.models import SubGoal, SubGoalStatus


@dataclass
class ProgressReport:
    completed_milestones: int = 0
    total_milestones: int = 0
    remaining: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    failed_attempts: int = 0
    pending_approvals: int = 0
    percent: float | None = None  # None = unknown denominator

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_milestones": self.completed_milestones,
            "total_milestones": self.total_milestones,
            "remaining": list(self.remaining),
            "blockers": list(self.blockers),
            "failed_attempts": self.failed_attempts,
            "pending_approvals": self.pending_approvals,
            "percent": (
                round(self.percent, 1)
                if self.percent is not None
                else None
            ),
        }

    def summary_text(self) -> str:
        if self.percent is not None:
            base = (
                f"{self.completed_milestones}/"
                f"{self.total_milestones} milestones "
                f"({self.percent:.0f}%)"
            )
        else:
            base = (
                f"{self.completed_milestones} milestones "
                "verified complete"
            )
        if self.blockers:
            base += f"; blocked: {'; '.join(self.blockers[:2])}"
        if self.pending_approvals:
            base += (
                f"; {self.pending_approvals} approval(s) pending"
            )
        return base


class ProgressTracker:
    """Derive progress from subgoal + execution state."""

    def __init__(self) -> None:
        self._blockers: list[str] = []
        self._failed_attempts = 0
        self._pending_approvals = 0

    def add_blocker(self, blocker: str) -> None:
        text = blocker.strip()[:200]
        if text and text not in self._blockers:
            self._blockers.append(text)

    def clear_blocker(self, blocker: str) -> None:
        text = blocker.strip()[:200]
        if text in self._blockers:
            self._blockers.remove(text)

    def record_failed_attempt(self) -> None:
        self._failed_attempts += 1

    def set_pending_approvals(self, count: int) -> None:
        self._pending_approvals = max(0, int(count))

    def dependencies_satisfied(
        self, subgoal: SubGoal, by_id: dict[str, SubGoal]
    ) -> bool:
        return all(
            by_id.get(dep_id) is not None
            and by_id[dep_id].status == SubGoalStatus.COMPLETED
            for dep_id in subgoal.depends_on
        )

    def ready_subgoals(
        self, subgoals: list[SubGoal]
    ) -> list[SubGoal]:
        """Pending subgoals whose dependencies are verified."""
        by_id = {s.subgoal_id: s for s in subgoals}
        return [
            s
            for s in subgoals
            if s.status == SubGoalStatus.PENDING
            and self.dependencies_satisfied(s, by_id)
        ]

    def report(
        self, subgoals: list[SubGoal]
    ) -> ProgressReport:
        completed = [
            s
            for s in subgoals
            if s.status == SubGoalStatus.COMPLETED
        ]
        remaining = [
            s.description[:120]
            for s in subgoals
            if s.status
            in (SubGoalStatus.PENDING, SubGoalStatus.ACTIVE)
        ]
        total = len(subgoals)
        percent: float | None = None
        if total > 0:
            percent = 100.0 * len(completed) / total
        return ProgressReport(
            completed_milestones=len(completed),
            total_milestones=total,
            remaining=remaining,
            blockers=list(self._blockers),
            failed_attempts=self._failed_attempts,
            pending_approvals=self._pending_approvals,
            percent=percent,
        )

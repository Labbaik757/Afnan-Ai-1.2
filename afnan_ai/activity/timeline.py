"""Timeline, progress, filtering and summaries.

- ``TaskProgress``: per-task live progress derived from events.
  No fake percentages — progress is step-based and only
  reported where the plan gives a reliable denominator.
- ``ActivityFilter``: structured query over the event store.
- ``ActivityTimeline``: user-safe chronological view of a task.
  Internal chain-of-thought is never exposed; only
  execution summaries, actions, observations and outcomes.
- ``ActivitySummary``: auto-generated safe summaries from
  verified activity/state (never from raw reasoning).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from afnan_ai.activity import events as E


def _parse(at: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(at))
    except (ValueError, TypeError):
        return None


@dataclass
class TaskProgress:
    task_id: str
    goal: str = ""
    status: str = "IDLE"
    current_step: str = ""
    completed_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    failed_steps: list[str] = field(default_factory=list)
    retry_count: int = 0
    recovery_count: int = 0
    started_at: str = ""
    updated_at: str = ""
    elapsed_s: float = 0.0
    progress_fraction: float | None = None  # None = unknown
    workspace_id: str = ""
    tool_category: str = ""
    browser_context: str = ""
    approvals_pending: int = 0
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProgressTracker:
    """Builds TaskProgress purely from activity events."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskProgress] = {}
        self._plan_total: dict[str, int] = {}

    def _get(self, task_id: str) -> TaskProgress:
        if task_id not in self._tasks:
            self._tasks[task_id] = TaskProgress(task_id=task_id)
        return self._tasks[task_id]

    def apply(self, event: E.ActivityEvent) -> None:
        if not event.task_id:
            return
        p = self._get(event.task_id)
        d = event.details or {}
        p.updated_at = event.at
        if event.workspace_id:
            p.workspace_id = event.workspace_id
        step = str(d.get("step_id") or d.get("step") or "")
        tool = str(d.get("tool") or d.get("tool_category") or "")
        if tool:
            p.tool_category = tool
        ctx = str(d.get("browser_context") or d.get("url") or "")
        if ctx:
            p.browser_context = ctx[:200]

        if event.type == E.TASK_CREATED:
            p.goal = str(d.get("goal", ""))[:300]
        elif event.type == E.TASK_STARTED:
            p.started_at = p.started_at or event.at
            total = d.get("planned_steps")
            if isinstance(total, int) and total > 0:
                self._plan_total[event.task_id] = total
        elif event.type == E.PLANNING:
            p.current_step = "Planning next actions"
        elif event.type == E.OBSERVING:
            p.current_step = "Observing current state"
        elif event.type == E.ACTION_STARTED:
            p.current_step = (
                f"Executing: {step or tool or 'action'}"
            )
        elif event.type == E.ACTION_COMPLETED:
            if step and step not in p.completed_steps:
                p.completed_steps.append(step)
            p.current_step = ""
        elif event.type == E.ACTION_FAILED:
            if step and step not in p.failed_steps:
                p.failed_steps.append(step)
            p.retry_count += 1
        elif event.type == E.RECOVERY_STARTED:
            p.recovery_count += 1
            p.current_step = "Recovering from failure"
        elif event.type == E.APPROVAL_REQUIRED:
            p.approvals_pending += 1
            p.current_step = "Waiting for your approval"
        elif event.type in (E.APPROVAL_GRANTED, E.APPROVAL_DENIED):
            p.approvals_pending = max(0, p.approvals_pending - 1)
        elif event.type == E.ARTIFACT_CREATED:
            name = str(d.get("artifact_name") or d.get("name") or "")
            if name and name not in p.artifacts:
                p.artifacts.append(name)
        elif event.type == E.TASK_PAUSED:
            p.current_step = "Paused"
        elif event.type == E.TASK_RESUMED:
            p.current_step = ""

        # Elapsed + honest progress fraction.
        start = _parse(p.started_at)
        end = _parse(p.updated_at)
        if start and end:
            p.elapsed_s = max(
                0.0, (end - start).total_seconds()
            )
        total = self._plan_total.get(event.task_id)
        done = len(p.completed_steps)
        if total:
            p.progress_fraction = min(1.0, done / total)
        else:
            p.progress_fraction = None  # unknown: never faked

    def get(self, task_id: str) -> TaskProgress | None:
        return self._tasks.get(task_id)

    def all(self) -> list[TaskProgress]:
        return list(self._tasks.values())


@dataclass
class ActivityFilter:
    """Structured search over activity (secrets never indexed)."""

    types: list[str] = field(default_factory=list)
    task_id: str = ""
    workspace_id: str = ""
    goal_contains: str = ""
    tool_category: str = ""
    error_code: str = ""
    artifact_name: str = ""
    since: str = ""
    until: str = ""
    text: str = ""
    limit: int = 100
    offset: int = 0

    def matches(self, event: E.ActivityEvent) -> bool:
        if self.types and event.type not in self.types:
            return False
        if self.task_id and event.task_id != self.task_id:
            return False
        if (
            self.workspace_id
            and event.workspace_id != self.workspace_id
        ):
            return False
        d = event.details or {}
        if (
            self.tool_category
            and str(d.get("tool_category") or d.get("tool"))
            != self.tool_category
        ):
            return False
        if (
            self.error_code
            and str(d.get("error_code")) != self.error_code
        ):
            return False
        if self.artifact_name and self.artifact_name not in str(
            d.get("artifact_name") or d.get("name") or ""
        ):
            return False
        if self.since and event.at < self.since:
            return False
        if self.until and event.at > self.until:
            return False
        if self.text:
            needle = self.text.lower()
            hay = (
                event.summary + " " + event.task_id + " "
                + str(d.get("tool", ""))
            ).lower()
            if needle not in hay:
                return False
        if self.goal_contains and self.goal_contains.lower() not in (
            event.summary + " " + str(d.get("goal", ""))
        ).lower():
            return False
        return True


class ActivityTimeline:
    """User-safe chronological view of one task."""

    # Types that carry internal reasoning — summarized, never raw.
    _INTERNAL = frozenset({E.PLANNING, E.DECIDING, E.OBSERVING})

    def __init__(
        self, events: list[E.ActivityEvent]
    ) -> None:
        self.events = sorted(events, key=lambda e: e.seq)

    def items(self) -> list[dict[str, Any]]:
        out = []
        for e in self.events:
            item: dict[str, Any] = {
                "seq": e.seq,
                "at": e.at,
                "type": e.type,
                "summary": e.summary,
            }
            d = dict(e.details or {})
            # Safe detail subset for expansion.
            safe_keys = (
                "tool", "tool_category", "target", "url",
                "step_id", "error_code", "user_message",
                "artifact_name", "artifact_type",
                "verification", "recovery_action",
                "evidence_refs",
            )
            safe = {
                k: d[k]
                for k in safe_keys
                if k in d and d[k] not in ("", None)
            }
            if e.evidence_refs:
                safe["evidence_refs"] = list(e.evidence_refs)
            if safe:
                item["details"] = safe
            out.append(item)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"items": self.items(), "count": len(self.events)}


@dataclass
class ActivitySummary:
    """Auto-generated safe summary of a (long) task."""

    task_id: str
    objective: str = ""
    completed_work: list[str] = field(default_factory=list)
    current_state: str = ""
    blockers: list[str] = field(default_factory=list)
    approvals: list[str] = field(default_factory=list)
    failures_recoveries: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    next_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(
    task_id: str,
    events: list[E.ActivityEvent],
    progress: TaskProgress | None = None,
) -> ActivitySummary:
    """Generate a summary from verified activity — never from
    raw chain-of-thought."""
    s = ActivitySummary(task_id=task_id)
    ordered = sorted(events, key=lambda e: e.seq)
    for e in ordered:
        d = e.details or {}
        if e.type == E.TASK_CREATED and not s.objective:
            s.objective = str(d.get("goal", ""))[:300]
        elif e.type == E.ACTION_COMPLETED:
            label = str(
                d.get("step_id") or d.get("tool") or e.summary
            )[:160]
            if label not in s.completed_work:
                s.completed_work.append(label)
        elif e.type == E.APPROVAL_REQUIRED:
            s.approvals.append(
                f"Pending: {e.summary[:120]}"
            )
        elif e.type in (E.APPROVAL_GRANTED, E.APPROVAL_DENIED):
            s.approvals.append(
                f"{'Granted' if e.type == E.APPROVAL_GRANTED else 'Denied'}: "
                + e.summary[:120]
            )
        elif e.type == E.ACTION_FAILED:
            s.failures_recoveries.append(
                f"Failed: {e.summary[:120]}"
            )
        elif e.type == E.RECOVERY_STARTED:
            s.failures_recoveries.append(
                f"Recovering: {e.summary[:120]}"
            )
        elif e.type == E.ARTIFACT_CREATED:
            name = str(d.get("artifact_name", ""))[:160]
            if name and name not in s.artifacts:
                s.artifacts.append(name)
        elif e.type == E.WAITING:
            s.blockers.append(e.summary[:160])
    if progress:
        s.current_state = progress.current_step or progress.status
        if not s.objective:
            s.objective = progress.goal
    # Cap lengths for UI sanity.
    for attr in (
        "completed_work", "approvals", "failures_recoveries",
        "artifacts", "blockers",
    ):
        setattr(s, attr, getattr(s, attr)[-20:])
    return s

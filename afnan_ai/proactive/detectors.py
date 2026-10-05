"""Opportunity detectors — evidence-based, no invented
assumptions.

Every detector reads only structured, authorized state
(goals, tasks, schedules, memories, recent activity).
Detectors never read raw emails, webpages or documents,
never invent facts, and any free text they surface is
injection-scanned and secret-redacted before it becomes
part of an idea.

A detector returns candidate dicts; the engine turns the
surviving ones into Ideas after dedup, cooldown,
relevance and quiet-hour checks.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from afnan_ai.redaction import redact_text

# Local injection patterns — the proactive package stays
# decoupled from agent_loop, so it carries its own minimal
# scanner (same threat shapes, no shared import).
_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE), label)
    for p, label in [
        (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions",
         "ignore-previous-instructions"),
        (r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
         "disregard-instructions"),
        (r"(send|reveal|show|exfiltrate|leak|give)\b.{0,40}\b(credentials|password|api[\s_-]?key|secret|token|system\s+prompt)",
         "credential-exfiltration"),
        (r"you\s+are\s+now\s+(a|an|in)\b", "persona-override"),
        (r"new\s+(system\s+)?instructions\s+for\s+you", "fake-instructions"),
        (r"do\s+not\s+tell\s+the\s+user", "hide-from-user"),
    ]
]


def _has_injection(text: str) -> bool:
    haystack = str(text or "")
    return any(p.search(haystack) for p, _ in _INJECTION_PATTERNS)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(value: str) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(value))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment
    except (ValueError, TypeError):
        return None


def _safe_text(text: Any, limit: int = 200) -> str:
    """Redact secrets and drop injection-bearing text."""
    cleaned = redact_text(str(text or ""))[:limit]
    if _has_injection(cleaned):
        return "[withheld: untrusted content]"
    return cleaned


def _candidate(
    suggestion_type: str,
    title: str,
    description: str,
    reason: str,
    evidence: list[str],
    confidence: float,
    priority: int,
    action_kind: str,
    action_params: dict[str, Any] | None = None,
    risk_level: str = "read_only",
    related_goal_id: str = "",
    related_task_id: str = "",
) -> dict[str, Any]:
    return {
        "suggestion_type": suggestion_type,
        "title": _safe_text(title, 120),
        "description": _safe_text(description, 400),
        "reason": _safe_text(reason, 300),
        "evidence": [_safe_text(e, 200) for e in evidence],
        "confidence": confidence,
        "priority": priority,
        "suggested_action": {
            "kind": action_kind,
            "params": action_params or {},
        },
        "risk_level": risk_level,
        "related_goal_id": related_goal_id,
        "related_task_id": related_task_id,
    }


class DetectionContext:
    """Authorized state snapshot for one sweep."""

    def __init__(
        self,
        *,
        goals: list[Any] | None = None,
        tasks: list[Any] | None = None,
        schedules: list[Any] | None = None,
        memories: list[Any] | None = None,
        allowed_sources: tuple[str, ...] = (),
    ) -> None:
        self.goals = goals or []
        self.tasks = tasks or []
        self.schedules = schedules or []
        self.memories = memories or []
        self.allowed = set(allowed_sources)

    def allows(self, source: str) -> bool:
        return source in self.allowed


def detect_unfinished_tasks(
    ctx: DetectionContext,
) -> list[dict[str, Any]]:
    """Pending/paused/waiting tasks that have been idle."""
    if not ctx.allows("tasks"):
        return []
    out = []
    cutoff = _now() - timedelta(hours=24)
    for task in ctx.tasks:
        status = getattr(task, "status", "")
        if status not in (
            "pending", "paused", "waiting_for_approval"
        ):
            continue
        updated = _parse(getattr(task, "updated_at", "") or
                         getattr(task, "created_at", ""))
        if updated is None or updated > cutoff:
            continue
        goal_text = _safe_text(
            getattr(task, "goal_text", ""), 120
        )
        out.append(_candidate(
            "unfinished_task",
            f"Unfinished task: {goal_text[:60]}",
            f"This task has been {status} for over a day.",
            "Idle tasks are often forgotten, not done.",
            [f"task {task.task_id} status={status}",
             f"idle since {updated.isoformat()}"],
            confidence=0.75,
            priority=4,
            action_kind="resume_task",
            action_params={"task_id": task.task_id},
            risk_level="reversible",
            related_goal_id=getattr(task, "goal_id", ""),
            related_task_id=task.task_id,
        ))
    return out


def detect_stalled_goals(
    ctx: DetectionContext,
) -> list[dict[str, Any]]:
    """Active goals with low progress and no recent work."""
    if not ctx.allows("goals"):
        return []
    out = []
    cutoff = _now() - timedelta(days=3)
    for goal in ctx.goals:
        if getattr(goal, "status", "") != "active":
            continue
        progress = 0.0
        try:
            progress = float(goal.progress)
        except (TypeError, ValueError, AttributeError):
            pass
        if progress >= 0.8:
            continue
        recent = False
        if ctx.allows("tasks"):
            for task in ctx.tasks:
                if getattr(task, "goal_id", "") != (
                    goal.goal_id
                ):
                    continue
                updated = _parse(
                    getattr(task, "updated_at", "") or ""
                )
                if updated and updated > cutoff:
                    recent = True
                    break
        if recent:
            continue
        desc = _safe_text(
            getattr(goal, "description", ""), 100
        )
        out.append(_candidate(
            "goal_progress",
            f"Goal stalled: {desc[:60]}",
            "No task activity in 3 days; progress is "
            f"{int(progress * 100)}%.",
            "Stalled goals usually need a concrete next "
            "step, not more waiting.",
            [f"goal {goal.goal_id} progress={progress:.2f}",
             "no task activity in 3 days"],
            confidence=0.7,
            priority=4,
            action_kind="plan_next_step",
            action_params={"goal_id": goal.goal_id},
            risk_level="read_only",
            related_goal_id=goal.goal_id,
        ))
    return out


def detect_upcoming_deadlines(
    ctx: DetectionContext,
) -> list[dict[str, Any]]:
    """Scheduled runs due within the next 24 hours."""
    if not ctx.allows("schedules"):
        return []
    out = []
    horizon = _now() + timedelta(hours=24)
    for schedule in ctx.schedules:
        if getattr(schedule, "status", "") != "active":
            continue
        nxt = _parse(getattr(
            schedule, "next_run_at", "") or "")
        if nxt is None or nxt > horizon or nxt < _now():
            continue
        goal_text = _safe_text(
            getattr(schedule, "goal_text", ""), 100
        )
        out.append(_candidate(
            "upcoming_deadline",
            f"Upcoming: {goal_text[:60]}",
            f"Scheduled to run at {nxt.isoformat()}.",
            "A heads-up before scheduled work runs.",
            [f"schedule {schedule.schedule_id}",
             f"next_run_at={nxt.isoformat()}"],
            confidence=0.9,
            priority=3,
            action_kind="notify",
            action_params={
                "schedule_id": schedule.schedule_id
            },
            risk_level="read_only",
        ))
    return out


def detect_missed_dependencies(
    ctx: DetectionContext,
) -> list[dict[str, Any]]:
    """Goals whose dependencies are not satisfied."""
    if not ctx.allows("goals"):
        return []
    out = []
    by_id = {g.goal_id: g for g in ctx.goals}
    for goal in ctx.goals:
        if getattr(goal, "status", "") != "active":
            continue
        deps = getattr(goal, "dependencies", None) or []
        missing = [
            d for d in deps
            if d not in by_id
            or getattr(by_id[d], "status", "") != (
                "completed"
            )
        ]
        if not missing:
            continue
        desc = _safe_text(
            getattr(goal, "description", ""), 100
        )
        out.append(_candidate(
            "missed_dependency",
            f"Blocked goal: {desc[:60]}",
            f"Waiting on {len(missing)} unfinished "
            "dependenc(ies).",
            "Goals with unsatisfied dependencies cannot "
            "finish; the blocker is the real next step.",
            [f"goal {goal.goal_id}",
             f"missing={missing[:3]}"],
            confidence=0.85,
            priority=5,
            action_kind="review_dependencies",
            action_params={
                "goal_id": goal.goal_id,
                "missing": missing[:5],
            },
            risk_level="read_only",
            related_goal_id=goal.goal_id,
        ))
    return out


def detect_follow_ups(
    ctx: DetectionContext,
) -> list[dict[str, Any]]:
    """Recently completed tasks that may need a follow-up."""
    if not ctx.allows("tasks") or not ctx.allows(
        "activity"
    ):
        return []
    out = []
    cutoff = _now() - timedelta(days=2)
    for task in ctx.tasks:
        if getattr(task, "status", "") != "completed":
            continue
        updated = _parse(getattr(task, "updated_at", ""))
        if updated is None or updated < cutoff:
            continue
        meta = getattr(task, "metadata", None) or {}
        if meta.get("followed_up"):
            continue
        goal_text = _safe_text(
            getattr(task, "goal_text", ""), 100
        )
        out.append(_candidate(
            "follow_up",
            f"Follow up: {goal_text[:60]}",
            "Completed recently; a quick review keeps "
            "momentum.",
            "Completed work often has a natural next "
            "step while context is fresh.",
            [f"task {task.task_id} completed",
             f"at {updated.isoformat()}"],
            confidence=0.55,
            priority=2,
            action_kind="review_result",
            action_params={"task_id": task.task_id},
            risk_level="read_only",
            related_goal_id=getattr(task, "goal_id", ""),
            related_task_id=task.task_id,
        ))
    return out


def detect_recurring_workflows(
    ctx: DetectionContext,
) -> list[dict[str, Any]]:
    """Repeated completed task patterns → automation."""
    if not ctx.allows("tasks"):
        return []
    from collections import Counter

    done = [
        t for t in ctx.tasks
        if getattr(t, "status", "") == "completed"
    ]
    if len(done) < 3:
        return []
    keys = Counter(
        _safe_text(getattr(t, "goal_text", ""), 60)
        .lower()[:40]
        for t in done
    )
    out = []
    for key, count in keys.most_common(3):
        if count < 3 or not key.strip():
            continue
        out.append(_candidate(
            "recurring_workflow",
            f"Automate: {key[:60]}",
            f"Done {count} times — a scheduled workflow "
            "could handle it.",
            "Repetition is the signal for automation; "
            "the pattern is observed, not assumed.",
            [f"{count} completions of similar tasks"],
            confidence=0.65,
            priority=2,
            action_kind="propose_schedule",
            action_params={"pattern": key[:60]},
            risk_level="reversible",
        ))
    return out


ALL_DETECTORS = (
    detect_unfinished_tasks,
    detect_stalled_goals,
    detect_upcoming_deadlines,
    detect_missed_dependencies,
    detect_follow_ups,
    detect_recurring_workflows,
)


def run_detectors(
    ctx: DetectionContext,
) -> list[dict[str, Any]]:
    """Run every detector; never let one crash the sweep."""
    candidates: list[dict[str, Any]] = []
    for detector in ALL_DETECTORS:
        try:
            candidates.extend(detector(ctx))
        except Exception:
            continue  # one bad detector never kills a sweep
    return candidates

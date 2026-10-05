"""Evidence-based context summarization.

Long tasks get automatic summaries — completed work,
remaining work, discoveries, constraints, failures,
decisions, next objective — built *only* from recorded
trajectory entries, facts and sub-goal states.  This module
never calls a model, so it cannot invent facts: every line
of a summary traces back to something the agent actually
observed, decided or verified.
"""

from __future__ import annotations

import re
from typing import Any

from afnan_ai.context.models import (
    ContextItem,
    ItemKind,
    SubGoal,
    SubGoalStatus,
    TrajectoryEntry,
)


def _texts(
    items: list[ContextItem], kinds: set[ItemKind], limit: int = 8
) -> list[str]:
    out = []
    for item in items:
        if item.kind in kinds and item.text.strip():
            out.append(item.text.strip()[:220])
            if len(out) >= limit:
                break
    return out


def summarize_items(
    items: list[ContextItem],
    subgoals: list[SubGoal],
    *,
    goal: str = "",
) -> dict[str, Any]:
    """Build an evidence-based summary from recorded context.

    Facts may only enter via ``add_fact`` (source user or
    verified_result) — this function copies them verbatim and
    never synthesizes new claims.
    """
    completed = _texts(
        items, {ItemKind.RESULT, ItemKind.VERIFICATION}, limit=10
    )
    # Verified sub-goals are completed work with evidence.
    for sub in subgoals:
        if sub.status is SubGoalStatus.COMPLETED:
            line = f"Sub-goal done: {sub.description}"
            if sub.evidence:
                line += f" ({sub.evidence[:120]})"
            completed.append(line)
    remaining = [
        f"Sub-goal pending: {sub.description}"
        for sub in subgoals
        if sub.status
        in (SubGoalStatus.PENDING, SubGoalStatus.ACTIVE)
    ]
    failures = _texts(items, {ItemKind.FAILURE}, limit=8)
    recoveries = _texts(items, {ItemKind.RECOVERY}, limit=8)
    decisions = _texts(items, {ItemKind.DECISION}, limit=8)
    discoveries = _texts(
        items, {ItemKind.OBSERVATION}, limit=6
    )
    constraints = _texts(items, {ItemKind.CONSTRAINT}, limit=8)
    facts: dict[str, str] = {}
    for item in items:
        if item.kind is ItemKind.FACT and item.text.strip():
            # Facts are stored verbatim as "Fact [key]: text".
            match = re.match(r"^Fact \[(.+?)\]:\s*(.*)$",
                             item.text.strip(), re.DOTALL)
            if match:
                facts[match.group(1)] = match.group(2)[:300]
            else:
                facts[item.text[:40]] = item.text.strip()[:300]
    next_objective = ""
    for sub in subgoals:
        if sub.status in (
            SubGoalStatus.PENDING, SubGoalStatus.ACTIVE
        ):
            next_objective = sub.description
            break
    return {
        "goal": goal,
        "completed_work": completed,
        "remaining_work": remaining,
        "important_discoveries": discoveries,
        "constraints": constraints,
        "failures": failures,
        "recovery_attempts": recoveries,
        "decisions": decisions,
        "facts": facts,
        "next_objective": next_objective,
        "evidence_based": True,
    }


def summarize_trajectory(
    entries: list[TrajectoryEntry],
    *,
    goal: str = "",
) -> dict[str, Any]:
    """Summarize raw trajectory entries (evidence only)."""
    completed: list[str] = []
    failures: list[str] = []
    recoveries: list[str] = []
    decisions: list[str] = []
    approvals: list[str] = []
    for entry in entries:
        text = entry.summary[:220]
        if entry.kind == "verification" and "verified" in text.lower():
            completed.append(text)
        elif entry.kind in ("result", "action_completed"):
            completed.append(text)
        elif entry.kind in ("failure", "action_failed"):
            failures.append(text)
        elif entry.kind == "recovery":
            recoveries.append(text)
        elif entry.kind == "decision":
            decisions.append(text)
        elif entry.kind == "approval":
            approvals.append(text)
    # De-duplicate while preserving order.
    def dedup(lines: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                out.append(line)
        return out[:10]

    return {
        "goal": goal,
        "completed_work": dedup(completed),
        "failures": dedup(failures),
        "recovery_attempts": dedup(recoveries),
        "decisions": dedup(decisions),
        "approvals": dedup(approvals),
        "total_entries": len(entries),
        "evidence_based": True,
    }


def compact_text(lines: list[str], *, per_line: int = 160) -> str:
    """Join summary lines into a short block for prompts."""
    return "\n".join(
        f"- {line[:per_line]}" for line in lines if line
    )

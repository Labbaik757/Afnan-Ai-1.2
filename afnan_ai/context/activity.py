"""ActivityCenter integration: user-facing context summaries.

The user sees: current objective, completed milestones,
current phase, blockers, recent verified work and the next
action.  Raw internal reasoning is never exposed — only
the redacted, sensitivity-filtered summary.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.context.models import ContextItemV2
from afnan_ai.context.progress import ProgressReport
from afnan_ai.context.sensitivity import filter_for_activity
from afnan_ai.redaction import redact_text


def build_activity_summary(
    *,
    goal: str,
    progress: ProgressReport | None = None,
    current_phase: str = "",
    next_action: str = "",
    recent_items: list[ContextItemV2] | None = None,
) -> dict[str, Any]:
    """Assemble the user-facing summary payload."""
    items = filter_for_activity(recent_items or [])
    summary: dict[str, Any] = {
        "objective": redact_text(goal)[:300],
        "current_phase": redact_text(current_phase)[:160],
        "next_action": redact_text(next_action)[:200],
        "recent_verified_work": [
            i["text"][:160] for i in items[:5]
        ],
    }
    if progress is not None:
        summary["progress"] = progress.to_dict()
        summary["progress_text"] = progress.summary_text()
    return summary


def publish_activity_summary(
    activity_center: Any,
    summary: dict[str, Any],
    *,
    task_id: str = "",
) -> bool:
    """Emit the summary; never raises (best-effort)."""
    try:
        activity_center.emit(
            "context_summary",
            "[context] " + summary.get(
                "progress_text", "task update"
            )[:200],
            task_id=task_id,
            source="context",
            details=summary,
        )
        return True
    except Exception:
        return False

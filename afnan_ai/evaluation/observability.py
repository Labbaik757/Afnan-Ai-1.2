"""Evaluation observability: ActivityCenter + audit.

evaluation.started / task.started / task.completed /
task.failed / verification.completed / score.created /
failure.classified / regression.detected /
baseline.created / completed.

Sensitive benchmark data is never logged — only
counts, ids and redacted summaries.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.redaction import redact_text

_EVAL_EVENTS = frozenset({
    "evaluation.started",
    "evaluation.task.started",
    "evaluation.task.completed",
    "evaluation.task.failed",
    "evaluation.verification.completed",
    "evaluation.score.created",
    "evaluation.failure.classified",
    "evaluation.regression.detected",
    "evaluation.baseline.created",
    "evaluation.completed",
})


def emit_evaluation_event(
    activity_center: Any,
    event: str,
    summary: str,
    *,
    run_id: str = "",
) -> bool:
    if event not in _EVAL_EVENTS:
        return False
    try:
        activity_center.emit(
            event,
            "[evaluation] " + redact_text(summary)[:200],
            task_id=run_id,
            source="evaluation",
        )
        return True
    except Exception:
        return False

"""Research observability: ActivityCenter + audit events.

research.started / planned / search.started /
source.discovered / source.acquired / evidence.extracted /
claim.created / claim.verified / contradiction.detected /
citation.invalid / checkpoint.created / replanned /
completed / partial / failed.

Sensitive raw content is never logged — only counts,
ids and redacted summaries.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.redaction import redact_text

_RESEARCH_EVENTS = frozenset({
    "research.started",
    "research.planned",
    "research.search.started",
    "research.source.discovered",
    "research.source.acquired",
    "research.evidence.extracted",
    "research.claim.created",
    "research.claim.verified",
    "research.contradiction.detected",
    "research.citation.invalid",
    "research.checkpoint.created",
    "research.replanned",
    "research.completed",
    "research.partial",
    "research.failed",
})


def emit_research_event(
    activity_center: Any,
    event: str,
    summary: str,
    *,
    session_id: str = "",
    task_id: str = "",
    details: dict[str, Any] | None = None,
) -> bool:
    if event not in _RESEARCH_EVENTS:
        return False
    try:
        activity_center.emit(
            event,
            "[research] " + redact_text(summary)[:200],
            task_id=task_id or session_id,
            source="research",
            details={
                k: (
                    redact_text(str(v))[:200]
                    if isinstance(v, str)
                    else v
                )
                for k, v in (details or {}).items()
            },
        )
        return True
    except Exception:
        return False


def emit_research_audit(
    audit: Any,
    event: str,
    summary: str,
    *,
    session_id: str = "",
    actor: str = "agent",
) -> bool:
    record = {
        "event": event,
        "session_id": session_id,
        "actor": actor,
        "summary": redact_text(summary)[:300],
    }
    try:
        if hasattr(audit, "record"):
            audit.record(record)
        elif hasattr(audit, "log"):
            audit.log(record)
        else:
            return False
        return True
    except Exception:
        return False

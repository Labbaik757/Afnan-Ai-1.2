"""Skill observability: ActivityCenter + AuditCenter events.

ActivityCenter (user-facing): skill discovered/selected/
validated/started, step started/completed/failed, approval
required, recovery, verification, skill completed — with
redacted summaries, never raw reasoning.

AuditCenter (security): permission checks, capability use,
security findings, credential access attempts, publication
and version changes.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.redaction import redact_text

_ACTIVITY_EVENTS = frozenset({
    "skill_discovered",
    "skill_selected",
    "skill_validated",
    "skill_started",
    "step_started",
    "step_completed",
    "step_failed",
    "approval_required",
    "recovery",
    "verification",
    "skill_completed",
})

_AUDIT_EVENTS = frozenset({
    "permission_check",
    "capability_used",
    "security_finding",
    "credential_access",
    "publication",
    "version_change",
    "rollback",
})


def emit_activity(
    activity_center: Any,
    event: str,
    skill_id: str,
    summary: str,
    *,
    task_id: str = "",
    details: dict[str, Any] | None = None,
) -> bool:
    """Best-effort user-facing skill event."""
    if event not in _ACTIVITY_EVENTS:
        event = "skill_started"
    try:
        activity_center.emit(
            event,
            f"[skill:{skill_id}] "
            + redact_text(summary)[:200],
            task_id=task_id,
            source="skills",
            details={
                k: redact_text(str(v))[:200]
                for k, v in (details or {}).items()
            },
        )
        return True
    except Exception:
        return False


def emit_audit(
    audit: Any,
    event: str,
    skill_id: str,
    summary: str,
    *,
    actor: str = "agent",
    details: dict[str, Any] | None = None,
) -> bool:
    """Best-effort security audit record."""
    if event not in _AUDIT_EVENTS:
        return False
    record = {
        "event": event,
        "skill_id": skill_id,
        "actor": actor,
        "summary": redact_text(summary)[:300],
        "details": {
            k: redact_text(str(v))[:200]
            for k, v in (details or {}).items()
        },
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

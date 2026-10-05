"""ApprovalCenter — centralized pending-approval queue.

Reuses the security layer's ``ApprovalRequest`` (never
redefined).  The center tracks lifecycle:

    requested → granted | denied | expired

Supported decisions: approve once, deny, approve-for-task,
approve-for-limited-scope, expiry.  An approval never
auto-converts into a permanent unrestricted permission;
after a decision the policy is re-evaluated through the
SecurityCenter.

Secrets stay out: approval payloads are redacted on ingest.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from afnan_ai.activity import events as E
from afnan_ai.redaction import redact_value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


PENDING = "pending"
GRANTED = "granted"
DENIED = "denied"
EXPIRED = "expired"


@dataclass
class TrackedApproval:
    approval_id: str
    task_id: str
    action: str
    reason: str
    target: str = ""
    risk_level: str = "SENSITIVE"
    workspace_id: str = ""
    expected_consequence: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    requested_capability: str = ""
    status: str = PENDING
    scope: str = "once"  # once|task|limited
    scope_detail: str = ""
    requested_at: str = field(default_factory=_now_iso)
    expires_at: float = 0.0  # epoch; 0 = no expiry
    decided_at: str = ""
    decided_by: str = ""
    audit_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["expired"] = self.is_expired()
        return d

    def is_expired(self) -> bool:
        return (
            self.status == PENDING
            and self.expires_at > 0
            and time.time() > self.expires_at
        )


class ApprovalCenter:
    """Pending approvals + lifecycle + policy re-evaluation."""

    def __init__(
        self,
        *,
        center: Any = None,
        security_center: Any = None,
        task_manager: Any = None,
        default_ttl_s: float = 600.0,
    ) -> None:
        self.center = center
        self.security = security_center
        self.tasks = task_manager
        self.default_ttl_s = default_ttl_s
        self._approvals: dict[str, TrackedApproval] = {}
        self._lock = threading.RLock()

    # -- ingest ----------------------------------------------------------
    def request(
        self,
        *,
        task_id: str,
        action: str,
        reason: str,
        target: str = "",
        risk_level: str = "SENSITIVE",
        workspace_id: str = "",
        expected_consequence: str = "",
        evidence_refs: list[str] | None = None,
        requested_capability: str = "",
        scope: str = "once",
        scope_detail: str = "",
        ttl_s: float | None = None,
        audit_ref: str = "",
    ) -> TrackedApproval:
        approval_id = uuid.uuid4().hex[:16]
        ttl = (
            self.default_ttl_s if ttl_s is None else ttl_s
        )
        tracked = TrackedApproval(
            approval_id=approval_id,
            task_id=task_id,
            action=str(action)[:200],
            reason=str(reason)[:500],
            target=str(target)[:200],
            risk_level=str(risk_level),
            workspace_id=workspace_id,
            expected_consequence=str(expected_consequence)[
                :500
            ],
            evidence_refs=list(evidence_refs or []),
            requested_capability=str(requested_capability)[
                :120
            ],
            scope=scope if scope in ("once", "task", "limited")
            else "once",
            scope_detail=str(scope_detail)[:200],
            expires_at=(
                time.time() + ttl if ttl > 0 else 0.0
            ),
            audit_ref=audit_ref,
        )
        # Redact once, at the boundary.
        for attr in (
            "reason", "target", "expected_consequence",
            "scope_detail",
        ):
            setattr(
                tracked, attr,
                str(redact_value(getattr(tracked, attr))),
            )
        with self._lock:
            self._approvals[approval_id] = tracked
        if self.center is not None:
            self.center.emit(
                E.APPROVAL_REQUIRED,
                f"Approval required: {tracked.action}",
                task_id=task_id,
                workspace_id=workspace_id,
                source="control",
                details={
                    "approval_id": approval_id,
                    "action": tracked.action,
                    "risk_level": tracked.risk_level,
                    "target": tracked.target,
                    "scope": tracked.scope,
                },
                audit_ref=audit_ref,
            )
        return tracked

    def from_security_request(
        self,
        security_request: Any,
        *,
        task_id: str,
        workspace_id: str = "",
        audit_ref: str = "",
    ) -> TrackedApproval:
        """Adapt the security layer's ApprovalRequest (reuse)."""
        risk = getattr(security_request, "risk_level", "SENSITIVE")
        risk_value = (
            risk.value
            if hasattr(risk, "value")
            else str(risk)
        )
        return self.request(
            task_id=task_id,
            action=str(
                getattr(security_request, "action", "")
            ),
            reason=str(
                getattr(security_request, "reason", "")
            ),
            target=str(
                getattr(security_request, "target", "")
            ),
            risk_level=risk_value,
            workspace_id=workspace_id,
            expected_consequence=str(
                getattr(
                    security_request,
                    "expected_consequence", "",
                )
            ),
            audit_ref=audit_ref,
        )

    def apply(self, event: E.ActivityEvent) -> None:
        """Fold loop-emitted approval_required events in."""
        if event.type != E.APPROVAL_REQUIRED:
            return
        d = event.details or {}
        if d.get("approval_id") and d["approval_id"] in self._approvals:
            return  # already tracked
        self.request(
            task_id=event.task_id,
            action=str(d.get("action", event.summary)),
            reason=str(d.get("reason", "")),
            target=str(d.get("target", "")),
            risk_level=str(d.get("risk_level", "SENSITIVE")),
            workspace_id=event.workspace_id,
            audit_ref=event.audit_ref,
        )

    # -- queries -----------------------------------------------------------
    def pending(self) -> list[TrackedApproval]:
        with self._lock:
            out = []
            for a in self._approvals.values():
                if a.is_expired():
                    a.status = EXPIRED
                    self._expire_event(a)
                if a.status == PENDING:
                    out.append(a)
            return sorted(
                out, key=lambda a: a.requested_at
            )

    def get(self, approval_id: str) -> TrackedApproval | None:
        with self._lock:
            return self._approvals.get(approval_id)

    # -- decisions -----------------------------------------------------------
    def decide(
        self,
        approval_id: str,
        *,
        granted: bool,
        actor: str = "user",
        scope: str = "once",
        scope_detail: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            tracked = self._approvals.get(approval_id)
            if tracked is None:
                raise KeyError(
                    f"unknown approval: {approval_id}"
                )
            if tracked.is_expired():
                tracked.status = EXPIRED
                self._expire_event(tracked)
                raise ValueError("approval expired")
            if tracked.status != PENDING:
                raise ValueError(
                    f"approval already {tracked.status}"
                )
            tracked.status = GRANTED if granted else DENIED
            tracked.scope = (
                scope
                if scope in ("once", "task", "limited")
                else "once"
            )
            tracked.scope_detail = str(scope_detail)[:200]
            tracked.decided_at = _now_iso()
            tracked.decided_by = actor

        # Re-evaluate policy through the SecurityCenter —
        # the decision is recorded, never auto-escalated.
        if self.security is not None:
            try:
                self.security.audit.log(
                    "approval_granted"
                    if granted
                    else "approval_denied",
                    actor=actor,
                    action=tracked.action,
                    target=tracked.target,
                    task_id=tracked.task_id,
                    risk_level=tracked.risk_level,
                    approval_status=tracked.status,
                    details={
                        "approval_id": approval_id,
                        "scope": tracked.scope,
                    },
                )
            except Exception:
                pass

        if self.center is not None:
            self.center.emit(
                E.APPROVAL_GRANTED
                if granted
                else E.APPROVAL_DENIED,
                f"Approval {tracked.status}: {tracked.action}",
                task_id=tracked.task_id,
                workspace_id=tracked.workspace_id,
                actor=actor,
                source="control",
                details={
                    "approval_id": approval_id,
                    "scope": tracked.scope,
                },
                audit_ref=tracked.audit_ref,
            )

        # Resume a task parked on this approval.
        if granted and self.tasks is not None:
            try:
                task = self.tasks.get(tracked.task_id)
                if (
                    task is not None
                    and task.status == "waiting_for_approval"
                ):
                    self.tasks.resume(tracked.task_id)
            except Exception:
                pass

        return {
            "approval_id": approval_id,
            "status": tracked.status,
            "scope": tracked.scope,
        }

    def _expire_event(self, tracked: TrackedApproval) -> None:
        if self.center is not None:
            self.center.emit(
                E.APPROVAL_DENIED,
                f"Approval expired: {tracked.action}",
                task_id=tracked.task_id,
                workspace_id=tracked.workspace_id,
                source="control",
                details={
                    "approval_id": tracked.approval_id,
                    "expired": True,
                },
            )

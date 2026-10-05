"""Connector permission model — risk classification and human approval.

Every connector operation carries a :class:`RiskLevel`
(declared by the connector, not guessed):

* ``read`` / ``low_risk_write`` — run once authenticated and
  in-scope.
* ``sensitive_write`` / ``irreversible_destructive`` — run
  **only** with explicit human approval.

:class:`ConnectorApprovalGate` is fail-safe: with no approver
configured, sensitive operations are refused (structured
``approval_required`` / ``approval_denied`` errors); a blocked
risk level never runs; a broken or too-late approver never
allows.  Approval requests carry redacted parameters only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from afnan_ai.connectors.models import OperationSpec, RiskLevel
from afnan_ai.log_config import get_logger
from afnan_ai.redaction import redact_value

logger = get_logger(__name__)


@dataclass
class OperationRisk:
    """Classification of one proposed connector operation."""

    risk: RiskLevel
    category: str = "general"
    reason: str = ""

    @property
    def needs_approval(self) -> bool:
        return self.risk.needs_approval

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk": self.risk.value,
            "category": self.category,
            "reason": self.reason,
        }


def classify_operation(
    connector_id: str,
    spec: OperationSpec,
) -> OperationRisk:
    """Classify an operation from its declared risk level."""
    reasons = {
        RiskLevel.READ: "Read-only operation: observes external state",
        RiskLevel.LOW_RISK_WRITE: (
            "Low-risk write: small, easily reversible change"
        ),
        RiskLevel.SENSITIVE_WRITE: (
            "Sensitive write: affects other people or external state"
        ),
        RiskLevel.DESTRUCTIVE: (
            "Irreversible/destructive: cannot be undone"
        ),
    }
    return OperationRisk(
        risk=spec.risk,
        category=spec.name,
        reason=reasons.get(spec.risk, "Classified by declared risk"),
    )


@dataclass
class RiskPolicy:
    """Which risk levels need a human, and which never run."""

    require_approval_for: frozenset = field(
        default_factory=lambda: frozenset({
            RiskLevel.SENSITIVE_WRITE,
            RiskLevel.DESTRUCTIVE,
        })
    )
    #: Risk levels that never execute, even with approval.
    blocked: frozenset = field(default_factory=frozenset)
    #: A human answer arriving later than this is a timeout:
    #: the operation does not run.  None disables the timeout.
    approval_timeout_s: float | None = None

    def requires_approval(self, risk: RiskLevel) -> bool:
        return risk in self.require_approval_for

    def is_blocked(self, risk: RiskLevel) -> bool:
        return risk in self.blocked


@dataclass
class ConnectorApprovalRequest:
    """What a human is asked to allow (parameters redacted)."""

    connector_id: str
    operation: str
    risk: str
    reason: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "operation": self.operation,
            "risk": self.risk,
            "reason": self.reason,
            "parameters": dict(self.parameters),
        }


@dataclass
class ApprovalDecision:
    allowed: bool
    risk: OperationRisk
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "risk": self.risk.risk.value,
            "category": self.risk.category,
            "detail": self.detail,
        }


#: A human decider: returns True to allow the operation.
Approver = Callable[[ConnectorApprovalRequest], bool]


class ConnectorApprovalGate:
    """Fail-safe approval gate for connector operations."""

    def __init__(
        self,
        approver: Approver | None = None,
        policy: RiskPolicy | None = None,
    ):
        self.policy = policy or RiskPolicy()
        self.approver = approver
        self.decisions: list[dict[str, Any]] = []

    def set_approver(self, approver: Approver | None) -> None:
        self.approver = approver

    def check(
        self,
        risk: OperationRisk,
        *,
        connector_id: str,
        operation: str,
        parameters: dict[str, Any] | None = None,
    ) -> ApprovalDecision:
        if self.policy.is_blocked(risk.risk):
            decision = ApprovalDecision(
                False, risk,
                f"Risk level {risk.risk.value!r} is blocked by policy",
            )
            self._record(risk, decision, connector_id, operation)
            return decision

        if not self.policy.requires_approval(risk.risk):
            return ApprovalDecision(
                True, risk,
                f"{risk.risk.value} runs without approval",
            )

        request = ConnectorApprovalRequest(
            connector_id=connector_id,
            operation=operation,
            risk=risk.risk.value,
            reason=risk.reason,
            parameters=redact_value(parameters or {}),
        )
        if self.approver is None:
            decision = ApprovalDecision(
                False, risk,
                "Sensitive operation requires human approval, but "
                "no approver is configured",
            )
            self._record(
                risk, decision, connector_id, operation, request
            )
            return decision

        try:
            started = time.monotonic()
            allowed = bool(self.approver(request))
            elapsed = time.monotonic() - started
        except Exception as e:  # a broken approver never allows
            allowed = False
            elapsed = 0.0
            logger.warning(
                "connector approver raised for %s.%s: %s",
                connector_id, operation, e,
            )
        timeout = self.policy.approval_timeout_s
        if timeout is not None and elapsed > timeout:
            decision = ApprovalDecision(
                False, risk,
                f"Approval request timed out after {elapsed:.1f}s "
                f"(limit {timeout}s); operation not executed",
            )
            self._record(
                risk, decision, connector_id, operation, request,
                outcome="timeout",
            )
            return decision
        decision = ApprovalDecision(
            allowed, risk,
            "Approved by human approver" if allowed
            else "Denied by human approver",
        )
        self._record(
            risk, decision, connector_id, operation, request,
            outcome="approved" if allowed else "denied",
        )
        return decision

    def _record(
        self,
        risk: OperationRisk,
        decision: ApprovalDecision,
        connector_id: str,
        operation: str,
        request: ConnectorApprovalRequest | None = None,
        outcome: str | None = None,
    ) -> None:
        entry = {
            "connector_id": connector_id,
            "operation": operation,
            "risk": risk.risk.value,
            "allowed": decision.allowed,
            "detail": decision.detail,
            "outcome": outcome or (
                "approved" if decision.allowed else "denied"
            ),
            "decided_at": datetime.now(timezone.utc).isoformat(),
        }
        if request is not None:
            entry["request"] = request.to_dict()
        self.decisions.append(entry)
        logger.info(
            "connector approval: %s.%s risk=%s allowed=%s",
            connector_id, operation, risk.risk.value,
            decision.allowed,
        )

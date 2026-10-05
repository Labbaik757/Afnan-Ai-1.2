"""Connector data models — serializable metadata only.

Everything here is *safe metadata*: ids, names, descriptions,
risk levels, statuses, summaries and timestamps.  Credentials,
tokens and raw sensitive payloads never appear in these
models; they travel only through the transient
:class:`~afnan_ai.connectors.base.OperationContext` given to
a connector at call time and live in the
:class:`~afnan_ai.connectors.credentials.CredentialStore`
in between.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RiskLevel(str, Enum):
    """Risk classification of a connector operation.

    * ``read`` — observes external state, changes nothing.
    * ``low_risk_write`` — small, easily reversible writes
      (e.g. creating a draft).
    * ``sensitive_write`` — writes that affect other people or
      external state (sending mail, publishing, creating a
      calendar event): needs human approval.
    * ``irreversible_destructive`` — cannot be undone (deleting
      data, closing accounts): needs human approval and may be
      blocked by policy.
    """

    READ = "read"
    LOW_RISK_WRITE = "low_risk_write"
    SENSITIVE_WRITE = "sensitive_write"
    DESTRUCTIVE = "irreversible_destructive"

    @property
    def needs_approval(self) -> bool:
        return self in (
            RiskLevel.SENSITIVE_WRITE,
            RiskLevel.DESTRUCTIVE,
        )


@dataclass
class OperationSpec:
    """One operation a connector offers.

    ``parameters`` is a JSON-Schema-style object schema
    (``properties`` + ``required``) so the Planner sees the same
    contract the service validates before executing.
    """

    name: str
    description: str = ""
    risk: RiskLevel = RiskLevel.READ
    required_scopes: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.risk, str):
            self.risk = RiskLevel(self.risk)
        self.required_scopes = tuple(self.required_scopes or ())

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk": self.risk.value,
            "required_scopes": list(self.required_scopes),
            "parameters": dict(self.parameters),
        }


@dataclass
class ConnectorInfo:
    """Public, planner-facing description of a connector."""

    connector_id: str
    name: str
    description: str = ""
    auth_type: str = "none"
    declared_scopes: tuple[str, ...] = ()
    version: str = "1.0"
    operations: tuple[OperationSpec, ...] = ()

    def __post_init__(self) -> None:
        self.declared_scopes = tuple(self.declared_scopes or ())
        self.operations = tuple(self.operations or ())

    def operation(self, name: str) -> OperationSpec | None:
        for spec in self.operations:
            if spec.name == name:
                return spec
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "name": self.name,
            "description": self.description,
            "auth_type": self.auth_type,
            "declared_scopes": list(self.declared_scopes),
            "version": self.version,
            "operations": [op.to_dict() for op in self.operations],
        }


class ConnectionStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    AUTH_EXPIRED = "auth_expired"
    ERROR = "error"


@dataclass
class ConnectionRecord:
    """Safe connection state (no secrets)."""

    connector_id: str
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    granted_scopes: tuple[str, ...] = ()
    connected_at: str | None = None
    last_health_check: str | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "status": self.status.value,
            "granted_scopes": list(self.granted_scopes),
            "connected_at": self.connected_at,
            "last_health_check": self.last_health_check,
            "last_error": self.last_error,
        }


@dataclass
class OperationResult:
    """Safe result of one connector operation.

    ``summary`` is a short, redacted, human/LLM-readable
    description of what happened — never the raw payload and
    never credentials.  ``verification_hint`` is optional
    evidence the Verifier can use (e.g. the created event id),
    without sensitive content.
    """

    connector_id: str
    operation: str
    status: str = "ok"  # "ok" | "failed"
    summary: str = ""
    verification_hint: str = ""
    started_at: str = field(default_factory=_utcnow)
    finished_at: str = field(default_factory=_utcnow)
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "operation": self.operation,
            "status": self.status,
            "summary": self.summary,
            "verification_hint": self.verification_hint,
            "verification_status": "pending",
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 3),
        }


@dataclass
class AuditRecord:
    """One safe audit entry for a connector operation."""

    connector_id: str
    operation: str
    risk: str
    timestamp: str = field(default_factory=_utcnow)
    approval: str = "not_required"  # not_required|required|granted|denied
    status: str = "ok"  # ok|failed
    error_code: str | None = None
    duration_s: float = 0.0
    verification: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "operation": self.operation,
            "risk": self.risk,
            "timestamp": self.timestamp,
            "approval": self.approval,
            "status": self.status,
            "error_code": self.error_code,
            "duration_s": round(self.duration_s, 3),
            "verification": self.verification,
        }

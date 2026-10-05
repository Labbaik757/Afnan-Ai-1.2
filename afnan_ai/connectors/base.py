"""Connector interface — the contract every integration implements.

A connector is a self-contained adapter for one external
service (email, calendar, cloud storage, chat, ...).  The core
agent knows *only* this interface: discovery, capability
schemas, lifecycle (connect / authenticate / refresh /
disconnect / health-check) and one generic ``execute_operation``
entry point.  There is no email/calendar/... knowledge in the
agent, the planner or the service — new services are added by
subclassing :class:`Connector` and registering it, never by
modifying core code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from afnan_ai.connectors.auth import AuthSession, AuthType
from afnan_ai.connectors.errors import (
    AuthenticationError,
    ConnectorErrorCode,
    UnsupportedOperationError,
)
from afnan_ai.connectors.models import (
    ConnectionStatus,
    ConnectorInfo,
    OperationResult,
    OperationSpec,
    RiskLevel,
)


@dataclass
class OperationContext:
    """Transient, per-call context for a connector.

    ``credentials`` and ``session_secrets`` carry secret
    material and exist only for the duration of the call — the
    service builds this context fresh from the
    :class:`~afnan_ai.connectors.credentials.CredentialStore`
    and never persists it (not in ``AgentState``, memory,
    checkpoints or logs).
    """

    connector_id: str
    credentials: dict[str, str] = field(default_factory=dict)
    session_secrets: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 30.0

    def credential(self, name: str, default: str = "") -> str:
        return str(self.credentials.get(name, default) or default)


@dataclass
class ConnectionResult:
    """Outcome of connect / authenticate / refresh."""

    connector_id: str
    status: ConnectionStatus
    session: AuthSession | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "status": self.status.value,
            "session": self.session.to_dict() if self.session else None,
            "detail": self.detail,
        }


@dataclass
class HealthResult:
    """Outcome of a health check (safe metadata only)."""

    connector_id: str
    healthy: bool
    detail: str = ""
    latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "healthy": self.healthy,
            "detail": self.detail,
            "latency_s": round(self.latency_s, 3),
        }


class Connector(ABC):
    """Interface every external-service connector implements.

    Identity and capabilities are plain attributes so the
    registry can describe a connector without instantiating
    service clients:

    * ``connector_id`` — unique, e.g. ``"email"``
    * ``name`` / ``description`` — human/LLM readable
    * ``auth_type`` — which :class:`AuthType` it needs
    * ``declared_scopes`` — permission scopes it may request
      (least privilege: the service grants a subset)
    * ``operations`` — :class:`OperationSpec` list with risk
      levels and required scopes
    """

    connector_id: str = ""
    name: str = ""
    description: str = ""
    auth_type: AuthType = AuthType.NONE
    declared_scopes: tuple[str, ...] = ()
    version: str = "1.0"
    operations: tuple[OperationSpec, ...] = ()

    # -- capability description -------------------------------------
    def info(self) -> ConnectorInfo:
        """Planner-facing description (no secrets, no clients)."""
        return ConnectorInfo(
            connector_id=self.connector_id,
            name=self.name,
            description=self.description,
            auth_type=(
                self.auth_type.value
                if isinstance(self.auth_type, AuthType)
                else str(self.auth_type)
            ),
            declared_scopes=tuple(self.declared_scopes or ()),
            version=self.version,
            operations=tuple(self.operations or ()),
        )

    def get_operation(self, name: str) -> OperationSpec:
        spec = self.info().operation(name)
        if spec is None:
            raise UnsupportedOperationError(
                f"Connector {self.connector_id!r} has no operation "
                f"{name!r}",
                connector=self.connector_id,
                operation=name,
                details={
                    "available": [op.name for op in self.operations or ()]
                },
            )
        return spec

    # -- lifecycle ----------------------------------------------------
    @abstractmethod
    def authenticate(
        self,
        credentials: Mapping[str, str],
        context: OperationContext,
    ) -> AuthSession:
        """Validate/exchange *credentials* -> an :class:`AuthSession`.

        Raises :class:`AuthenticationError` (with code
        ``authentication_failed`` / ``missing_credentials``) when
        the credentials are absent or rejected.  Must not log or
        return secret values.
        """

    @abstractmethod
    def connect(self, context: OperationContext) -> ConnectionResult:
        """Establish a working session using context credentials."""

    def disconnect(self) -> None:
        """Tear down the session.  Secrets stay in the store."""

    def refresh_session(
        self, context: OperationContext
    ) -> ConnectionResult:
        """Renew an expired session (OAuth2 refresh flow).

        The default implementation raises
        ``unsupported_operation`` — connectors whose auth does
        not expire (API keys) keep it.
        """
        raise UnsupportedOperationError(
            f"Connector {self.connector_id!r} does not support "
            "session refresh",
            code=ConnectorErrorCode.UNSUPPORTED_OPERATION,
            connector=self.connector_id,
            operation="refresh_session",
        )

    @abstractmethod
    def health_check(self, context: OperationContext) -> HealthResult:
        """Lightweight liveness check (no side effects)."""

    # -- execution ------------------------------------------------------
    @abstractmethod
    def execute_operation(
        self,
        operation: str,
        parameters: dict[str, Any],
        context: OperationContext,
    ) -> OperationResult:
        """Run one declared operation.

        * Validate *parameters* against the operation's schema
          (the service already validated the envelope).
        * On expired auth raise ``authentication_expired`` — the
          service refreshes once and retries.
        * Return :class:`OperationResult` with a short redacted
          summary; never include credentials or raw sensitive
          payloads (the service redacts again as a backstop).
        """

    # -- helpers for implementations -----------------------------------
    def _result(
        self,
        operation: str,
        summary: str,
        *,
        verification_hint: str = "",
    ) -> OperationResult:
        return OperationResult(
            connector_id=self.connector_id,
            operation=operation,
            status="ok",
            summary=summary,
            verification_hint=verification_hint,
        )

    def _auth_failed(self, detail: str) -> AuthenticationError:
        return AuthenticationError(
            detail, connector=self.connector_id
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(id={self.connector_id!r})"


__all__ = [
    "AuthSession",
    "AuthType",
    "ConnectionResult",
    "Connector",
    "HealthResult",
    "OperationContext",
    "OperationResult",
    "OperationSpec",
    "RiskLevel",
]

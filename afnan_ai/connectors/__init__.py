"""Connector System — external-service integration.

The agent talks to external services (email, calendar, cloud
storage, chat, project management, design tools, GitHub, CRM,
...) only through this package:

* :class:`Connector` — the interface every integration
  implements (identity, capabilities, connect / authenticate /
  refresh / disconnect / health-check / execute-operation).
  Connector implementations stay completely separate from
  the core agent.
* :class:`ConnectorRegistry` — dynamic registration and
  entry-point discovery; structured capability schemas for
  the Planner/AgentLoop.
* :class:`ConnectorService` — the runtime: validation,
  least-privilege scope checks, credential handling, session
  refresh, human approval for sensitive operations, timeouts,
  structured reliability errors, redacted audit trail.
* :class:`CredentialStore` — secrets live only here (never in
  AgentState, MemoryStore, planner output or logs).
* :class:`ConnectorApprovalGate` — fail-safe human approval
  for sensitive/irreversible operations.
* ``create_connector_tools`` — the six ``connector_*`` tools
  for the central ToolRegistry.

No real third-party integrations are hard-coded here: adding
email, calendar, Slack, Notion, Canva, GitHub or a CRM means
subclassing :class:`Connector` and registering it — core
agent code never changes.
"""

from afnan_ai.connectors.audit import ConnectorAuditLog
from afnan_ai.connectors.auth import (
    AuthSession,
    AuthType,
    api_key_headers,
    bearer_headers,
    require_fields,
    session_expiry_in,
)
from afnan_ai.connectors.base import (
    ConnectionResult,
    Connector,
    HealthResult,
    OperationContext,
)
from afnan_ai.connectors.credentials import (
    CredentialStore,
    EnvironmentCredentialStore,
    MemoryCredentialStore,
)
from afnan_ai.connectors.errors import (
    ApprovalError,
    AuthenticationError,
    ConnectorError,
    ConnectorErrorCode,
    ConnectorException,
    ConnectorNotFoundError,
    PermissionDeniedError,
    UnsupportedOperationError,
    map_exception,
)
from afnan_ai.connectors.models import (
    AuditRecord,
    ConnectionRecord,
    ConnectionStatus,
    ConnectorInfo,
    OperationResult,
    OperationSpec,
    RiskLevel,
)
from afnan_ai.connectors.policy import (
    Approver,
    ApprovalDecision,
    ConnectorApprovalGate,
    ConnectorApprovalRequest,
    OperationRisk,
    RiskPolicy,
    classify_operation,
)
from afnan_ai.connectors.registry import (
    ENTRY_POINT_GROUP,
    ConnectorRegistry,
)
from afnan_ai.connectors.service import ConnectorService
from afnan_ai.connectors.tools import create_connector_tools

__all__ = [
    "ENTRY_POINT_GROUP",
    "Approver",
    "ApprovalDecision",
    "ApprovalError",
    "AuditRecord",
    "AuthSession",
    "AuthType",
    "AuthenticationError",
    "ConnectionRecord",
    "ConnectionResult",
    "ConnectionStatus",
    "Connector",
    "ConnectorApprovalGate",
    "ConnectorApprovalRequest",
    "ConnectorAuditLog",
    "ConnectorError",
    "ConnectorErrorCode",
    "ConnectorException",
    "ConnectorInfo",
    "ConnectorNotFoundError",
    "ConnectorRegistry",
    "ConnectorService",
    "CredentialStore",
    "EnvironmentCredentialStore",
    "HealthResult",
    "MemoryCredentialStore",
    "OperationContext",
    "OperationResult",
    "OperationRisk",
    "OperationSpec",
    "PermissionDeniedError",
    "RiskLevel",
    "RiskPolicy",
    "UnsupportedOperationError",
    "api_key_headers",
    "bearer_headers",
    "classify_operation",
    "create_connector_tools",
    "map_exception",
    "require_fields",
    "session_expiry_in",
]

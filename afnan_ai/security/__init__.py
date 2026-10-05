"""Centralized Authorization & Capability Security.

The mandatory security foundation for the whole agent
runtime — browser, computer use, files, connectors,
skills, subagents, background tasks, scheduler,
artifacts, AgentLoop and memory:

* :class:`SecurityCenter` — the front door; every action
  is authorized through it.
* :class:`CapabilityManager` — explicit capability
  definitions (id, risk, resources, operations, approval,
  expiry, scope).  Nothing is implicitly allowed.
* :class:`SecurityPolicyEngine` — one deterministic,
  auditable decision flow: capability → resource → risk
  → permissions → policy → approval → allow/deny.
* :class:`PolicyProfile` — Restricted / Standard /
  Advanced / Fully Authorized.  Fully authorized never
  bypasses the engine.
* :class:`ResourcePolicy` — filesystem roots, browser
  domains, connector scopes, computer apps.
* :class:`TemporaryGrantStore` — expiring grants,
  auto-revoked.
* :class:`PolicyVersionStore` — every decision carries
  its policy version.
* :class:`PolicySimulator` — dry-run, zero side effects.
* :class:`CredentialVault` / leases — secrets live here
  only; agents get single-use operation tokens.
* :class:`SecretRedactor` — central redaction authority.
* :class:`TrustBoundary` / :class:`InstructionBoundary`
  — external content is data, never instructions.
* :class:`EmergencyStop` — global kill switch.
* :class:`ExecutionSandbox` — generated code runs here.
* :class:`SecurityAuditCenter` — hash-chained audit.
* :class:`RateLimiter` / :class:`ResourceGovernor`.
* :class:`CapabilityEscalation` — the honest path to
  more power.
"""

from afnan_ai.security.audit import AuditLogger
from afnan_ai.security.audit_center import (
    SecurityAuditCenter,
)
from afnan_ai.security.capabilities import (
    BUILTIN_CAPABILITIES,
    Capability,
    CapabilityManager,
)
from afnan_ai.security.center import SecurityCenter
from afnan_ai.security.emergency import (
    EmergencyStop,
    StopRecord,
)
from afnan_ai.security.escalation import (
    CapabilityEscalation,
)
from afnan_ai.security.injection import (
    TrustLevel,
    check_arguments,
    check_text,
    label_content,
)
from afnan_ai.security.limits import (
    RateLimiter,
    RatePolicy,
    ResourceGovernor,
    ResourceLimits,
)
from afnan_ai.security.models import (
    Actor,
    ActorKind,
    ApprovalRequest,
    AuthDecision,
    RiskLevel,
    SecurityEventType,
)
from afnan_ai.security.permissions import PermissionManager
from afnan_ai.security.platform import (
    PlatformSecurityProfile,
    current_profile,
    is_executable_name,
)
from afnan_ai.security.policy import (
    TOOL_CAPABILITY_MAP,
    SecurityPolicyEngine,
    derive_subagent_actor,
)
from afnan_ai.security.profiles import (
    PROFILE_INVARIANTS,
    PolicyProfile,
    describe_profile,
    grants_for,
)
from afnan_ai.security.redactor import (
    DEFAULT_REDACTOR,
    SecretRedactor,
)
from afnan_ai.security.resources import (
    ResourcePolicy,
    check_browser_url,
    check_computer_app,
    check_connector_scope,
    check_filesystem,
    check_resource,
)
from afnan_ai.security.risk import (
    classify_action,
    risk_for_capability,
)
from afnan_ai.security.sandbox import (
    DEFAULT_SANDBOX_POLICY,
    SandboxPolicy,
    SandboxViolation,
)
from afnan_ai.security.sandbox_exec import (
    DEFAULT_SANDBOX,
    ExecutionSandbox,
    SandboxLimits,
    SandboxResult,
)
from afnan_ai.security.simulator import PolicySimulator
from afnan_ai.security.temporary import (
    TemporaryGrant,
    TemporaryGrantStore,
)
from afnan_ai.security.trust import (
    BoundaryFinding,
    InstructionBoundary,
    TrustBoundary,
    coerce_boundary,
)
from afnan_ai.security.vault import (
    CredentialLease,
    CredentialVault,
    EnvCredentialVault,
    LeasedCredentialVault,
    MemoryCredentialVault,
)
from afnan_ai.security.versioning import (
    PolicyVersion,
    PolicyVersionStore,
)

__all__ = [
    "BUILTIN_CAPABILITIES",
    "DEFAULT_REDACTOR",
    "DEFAULT_SANDBOX",
    "DEFAULT_SANDBOX_POLICY",
    "PROFILE_INVARIANTS",
    "TOOL_CAPABILITY_MAP",
    "Actor",
    "ActorKind",
    "ApprovalRequest",
    "AuditLogger",
    "AuthDecision",
    "BoundaryFinding",
    "Capability",
    "CapabilityEscalation",
    "CapabilityManager",
    "CredentialLease",
    "CredentialVault",
    "EmergencyStop",
    "EnvCredentialVault",
    "ExecutionSandbox",
    "InstructionBoundary",
    "LeasedCredentialVault",
    "MemoryCredentialVault",
    "PermissionManager",
    "PlatformSecurityProfile",
    "PolicyProfile",
    "PolicySimulator",
    "PolicyVersion",
    "PolicyVersionStore",
    "RateLimiter",
    "RatePolicy",
    "ResourceGovernor",
    "ResourceLimits",
    "ResourcePolicy",
    "RiskLevel",
    "SandboxLimits",
    "SandboxPolicy",
    "SandboxResult",
    "SandboxViolation",
    "SecretRedactor",
    "SecurityAuditCenter",
    "SecurityCenter",
    "SecurityEventType",
    "SecurityPolicyEngine",
    "StopRecord",
    "TemporaryGrant",
    "TemporaryGrantStore",
    "TrustBoundary",
    "TrustLevel",
    "check_arguments",
    "check_browser_url",
    "check_computer_app",
    "check_connector_scope",
    "check_filesystem",
    "check_resource",
    "check_text",
    "classify_action",
    "coerce_boundary",
    "current_profile",
    "derive_subagent_actor",
    "describe_profile",
    "grants_for",
    "is_executable_name",
    "label_content",
    "risk_for_capability",
]

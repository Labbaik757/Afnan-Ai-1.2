"""Centralized Security, Permissions & Audit Center.

One mandatory security layer for browser, computer use,
files, connectors, skills, subagents, background tasks
and artifacts:

* :class:`SecurityCenter` — the front door; every action
  is authorized through it.
* :class:`PermissionManager` — capability-based (not
  role-based) permissions with least-privilege derivation.
* :func:`classify_action` — READ_ONLY / LOW_RISK_WRITE /
  SENSITIVE / IRREVERSIBLE, deterministic and central.
* :class:`SecurityPolicyEngine` — the enforcement flow:
  validate → classify → injection screen → rate limits →
  permission check → human approval → audit.
* :class:`CredentialVault` — secrets live here and only
  here; every access is logged without the value.
* :class:`AuditLogger` — hash-chained, tamper-evident,
  redacted audit records.
* :class:`RateLimiter` — per-actor tool-call, frequency,
  runtime and failure budgets; advises task pause on
  abuse.
* :mod:`injection` — trust levels; untrusted sources can
  never instruct or grant permissions.
* :class:`SandboxPolicy` — explicit allow/deny for
  generated code (filesystem, network, process, env;
  credentials never).
* :mod:`platform` — consistent abstraction across
  Windows, Linux and macOS.
"""

from afnan_ai.security.audit import AuditLogger
from afnan_ai.security.center import SecurityCenter
from afnan_ai.security.injection import (
    TrustLevel,
    check_arguments,
    check_text,
    label_content,
)
from afnan_ai.security.limits import (
    RateLimiter,
    RatePolicy,
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
    SecurityPolicyEngine,
    derive_subagent_actor,
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
from afnan_ai.security.vault import (
    CredentialVault,
    EnvCredentialVault,
    MemoryCredentialVault,
)

__all__ = [
    "Actor",
    "ActorKind",
    "ApprovalRequest",
    "AuditLogger",
    "AuthDecision",
    "CredentialVault",
    "DEFAULT_SANDBOX_POLICY",
    "EnvCredentialVault",
    "MemoryCredentialVault",
    "PermissionManager",
    "PlatformSecurityProfile",
    "RateLimiter",
    "RatePolicy",
    "RiskLevel",
    "SandboxPolicy",
    "SandboxViolation",
    "SecurityCenter",
    "SecurityEventType",
    "SecurityPolicyEngine",
    "TrustLevel",
    "check_arguments",
    "check_text",
    "classify_action",
    "current_profile",
    "derive_subagent_actor",
    "is_executable_name",
    "label_content",
    "risk_for_capability",
]

"""Security Center data models.

One vocabulary for the whole agent: risk levels, actor
kinds, trust levels for prompt-injection defense, security
event types, approval requests and authorization
decisions.  Every subsystem (browser, computer, files,
connectors, skills, subagents, background tasks,
artifacts) speaks this language — no subsystem invents
its own security policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RiskLevel(str, Enum):
    READ_ONLY = "read_only"
    LOW_RISK_WRITE = "low_risk_write"
    SENSITIVE = "sensitive"
    IRREVERSIBLE = "irreversible"

    @classmethod
    def coerce(cls, value: Any) -> "RiskLevel":
        text = str(value or "").strip().lower()
        for member in cls:
            if member.value == text:
                return member
        return cls.SENSITIVE  # unknown → cautious


class ActorKind(str, Enum):
    USER = "user"
    AGENT = "agent"
    SUBAGENT = "subagent"
    TOOL = "tool"
    SKILL = "skill"
    CONNECTOR = "connector"
    BACKGROUND = "background"


@dataclass(frozen=True)
class Actor:
    kind: ActorKind
    actor_id: str = ""
    # Capability scope: the actor may only use these
    # capabilities (least privilege).  Empty scope on a
    # non-user actor means "nothing granted".
    capabilities: tuple[str, ...] = ()

    def scoped(self, capabilities: tuple[str, ...]) -> "Actor":
        allowed = tuple(
            c for c in capabilities if c in self.capabilities
        ) if self.capabilities else tuple(capabilities)
        return Actor(
            kind=self.kind, actor_id=self.actor_id,
            capabilities=allowed,
        )

    @property
    def label(self) -> str:
        base = self.kind.value
        return f"{base}:{self.actor_id}" if self.actor_id else base


class TrustLevel(str, Enum):
    """Where content came from — never confused with what
    it says.  Only USER_INSTRUCTION and SYSTEM_POLICY may
    authorize actions; everything else is data."""

    USER_INSTRUCTION = "user_instruction"
    SYSTEM_POLICY = "system_policy"
    AGENT_STATE = "agent_state"
    TOOL_OUTPUT = "tool_output"
    WEBPAGE = "webpage"
    EMAIL = "email"
    DOCUMENT = "document"

    @property
    def may_instruct(self) -> bool:
        return self in (
            TrustLevel.USER_INSTRUCTION,
            TrustLevel.SYSTEM_POLICY,
        )


class SecurityEventType(str, Enum):
    PERMISSION_DENIED = "permission_denied"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_REJECTED = "approval_rejected"
    CREDENTIAL_ACCESSED = "credential_accessed"
    SUSPICIOUS_INSTRUCTION = "suspicious_instruction"
    PROMPT_INJECTION = "prompt_injection"
    SANDBOX_VIOLATION = "sandbox_violation"
    REPEATED_FAILED_ACTION = "repeated_failed_action"
    ABNORMAL_TOOL_USAGE = "abnormal_tool_usage"
    RATE_LIMITED = "rate_limited"
    ACTION_AUTHORIZED = "action_authorized"
    POLICY_VIOLATION = "policy_violation"


@dataclass
class ApprovalRequest:
    """Structured human-approval request — always complete,
    never a bare 'allow this?'."""

    action: str
    reason: str
    target: str = ""
    risk_level: RiskLevel = RiskLevel.SENSITIVE
    expected_consequence: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    actor: str = "agent"
    requested_at: str = field(default_factory=_utcnow)

    def summary(self) -> str:
        lines = [
            f"Action: {self.action}",
            f"Actor: {self.actor}",
            f"Risk: {self.risk_level.value}",
        ]
        if self.target:
            lines.append(f"Target: {self.target}")
        if self.reason:
            lines.append(f"Reason: {self.reason}")
        if self.expected_consequence:
            lines.append(
                f"If approved: {self.expected_consequence}"
            )
        return "\n".join(lines)


@dataclass
class AuthDecision:
    """Outcome of the central policy flow."""

    allowed: bool
    action: str  # "allow" | "deny" | "approval_required"
    risk_level: RiskLevel = RiskLevel.READ_ONLY
    reason: str = ""
    approval_request: ApprovalRequest | None = None
    required_capability: str = ""
    policy_version: str = ""

    @classmethod
    def allow(
        cls, risk: RiskLevel, reason: str = ""
    ) -> "AuthDecision":
        return cls(
            True, "allow", risk_level=risk, reason=reason
        )

    @classmethod
    def deny(
        cls, reason: str, *,
        risk: RiskLevel = RiskLevel.SENSITIVE,
        capability: str = "",
    ) -> "AuthDecision":
        return cls(
            False, "deny", risk_level=risk, reason=reason,
            required_capability=capability,
        )

    @classmethod
    def need_approval(
        cls, request: ApprovalRequest
    ) -> "AuthDecision":
        return cls(
            False, "approval_required",
            risk_level=request.risk_level,
            reason="human approval required",
            approval_request=request,
        )

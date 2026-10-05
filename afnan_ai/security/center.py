"""SecurityCenter — one front door for all security.

Composes PermissionManager, SecurityPolicyEngine,
CredentialVault, AuditLogger, RateLimiter and
SandboxPolicy.  Subsystems never invent their own
policy: they ask the center.

    center.authorize(tool_name=..., arguments=...,
                     actor=..., task_id=...)

Security-relevant state is summarized into AgentState /
TaskManager metadata in redacted form only — credentials
and raw secrets are never persisted there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from afnan_ai.redaction import redact_value
from afnan_ai.security.audit import AuditLogger
from afnan_ai.security.limits import (
    RateLimiter,
    RatePolicy,
)
from afnan_ai.security.models import (
    Actor,
    ActorKind,
    AuthDecision,
    RiskLevel,
    SecurityEventType,
    TrustLevel,
)
from afnan_ai.security.permissions import PermissionManager
from afnan_ai.security.platform import current_profile
from afnan_ai.security.policy import SecurityPolicyEngine
from afnan_ai.security.sandbox import (
    DEFAULT_SANDBOX_POLICY,
    SandboxPolicy,
)
from afnan_ai.security.vault import (
    CredentialVault,
    MemoryCredentialVault,
)


class SecurityCenter:
    """Central, mandatory security authority."""

    def __init__(
        self,
        *,
        permissions: PermissionManager | None = None,
        audit_path: str | Path | None = None,
        vault: CredentialVault | None = None,
        rate_policy: RatePolicy | None = None,
        sandbox_policy: SandboxPolicy | None = None,
        approver: Callable[[str], bool] | None = None,
        on_security_event: (
            Callable[[dict[str, Any]], None] | None
        ) = None,
        strict_agent_approval: bool = False,
    ) -> None:
        self.permissions = permissions or PermissionManager()
        self.audit = AuditLogger(
            audit_path, on_event=on_security_event
        )
        self.vault = vault or MemoryCredentialVault(
            on_access=self._vault_access_event
        )
        self.rate_limiter = RateLimiter(rate_policy)
        self.sandbox_policy = (
            sandbox_policy or DEFAULT_SANDBOX_POLICY
        )
        # The agent starts with a broad-but-safe default;
        # least privilege narrows it per task/subagent.
        self.agent_actor = Actor(
            kind=ActorKind.AGENT, actor_id="main"
        )
        self.policy = SecurityPolicyEngine(
            permissions=self.permissions,
            audit=self.audit,
            rate_limiter=self.rate_limiter,
            approver=approver,
            on_security_event=on_security_event,
            agent_actor_label=self.agent_actor.label,
            strict_agent_approval=strict_agent_approval,
        )
        self.platform = current_profile()

    # -- the choke point ------------------------------------------------------
    def authorize(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        actor: Actor | str | None = None,
        task_id: str = "",
        required_capability: str = "",
        argument_trust: TrustLevel | str = (
            TrustLevel.TOOL_OUTPUT
        ),
    ) -> AuthDecision:
        return self.policy.authorize(
            tool_name=tool_name,
            arguments=arguments,
            actor=actor or self.agent_actor,
            task_id=task_id,
            required_capability=required_capability,
            argument_trust=argument_trust,
        )

    # -- actors -----------------------------------------------------------------
    def grant_agent_capabilities(
        self, *capabilities: str
    ) -> None:
        self.permissions.grant(self.agent_actor, *capabilities)

    def subagent_actor(
        self, subagent_id: str, requested: list[str]
    ) -> Actor:
        """Least-privilege subagent: never more than the
        agent holds."""
        return self.permissions.derive_child(
            self.agent_actor, ActorKind.SUBAGENT,
            subagent_id, requested,
        )

    # -- credentials --------------------------------------------------------------
    def store_credential(
        self, owner: str, name: str, value: str, *,
        kind: str = "generic",
    ) -> str:
        ref = self.vault.put(owner, name, value, kind=kind)
        self.audit.log(
            SecurityEventType.CREDENTIAL_ACCESSED.value,
            actor="system", action="credential_stored",
            target=f"{owner}/{name}",
            details={"kind": kind},
        )
        return ref

    def _vault_access_event(
        self, owner: str, name: str, kind: str
    ) -> None:
        self.audit.log(
            SecurityEventType.CREDENTIAL_ACCESSED.value,
            actor="system", action="credential_read",
            target=f"{owner}/{name}",
            details={"kind": kind},
        )

    # -- state integration ----------------------------------------------------------
    def security_summary(self) -> dict[str, Any]:
        """Redacted summary safe for AgentState / task
        metadata — counts only, never secrets."""
        denials = self.audit.query(
            event=SecurityEventType.PERMISSION_DENIED.value,
            limit=1000,
        )
        approvals = self.audit.query(
            event=SecurityEventType.APPROVAL_REQUESTED.value,
            limit=1000,
        )
        return {
            "denials": len(denials),
            "approvals_requested": len(approvals),
            "chain": self.audit.verify_chain(),
        }

    def record_execution_result(
        self,
        *,
        tool_name: str,
        actor: Actor | str,
        task_id: str,
        success: bool,
        risk_level: RiskLevel | str = RiskLevel.READ_ONLY,
    ) -> None:
        actor_label = (
            actor.label
            if isinstance(actor, Actor) else str(actor)
        )
        self.rate_limiter.record_result(
            actor_label, success
        )
        pause = self.rate_limiter.should_pause_task(
            actor_label
        )
        if pause["pause"]:
            self.audit.log(
                pause["code"],
                actor=actor_label, action=tool_name,
                task_id=task_id,
                risk_level=str(risk_level),
                details=redact_value(pause),
            )
        self.audit.log(
            "execution_result",
            actor=actor_label, action=tool_name,
            task_id=task_id,
            risk_level=str(risk_level),
            execution_result=(
                "success" if success else "failure"
            ),
        )

    def set_approver(
        self, approver: Callable[[str], bool] | None
    ) -> None:
        self.policy.set_approver(approver)


__all__ = ["SecurityCenter"]

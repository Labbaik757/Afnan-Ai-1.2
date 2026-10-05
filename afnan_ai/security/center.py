"""SecurityCenter — one front door for all security.

Composes the full authorization & capability stack:

* CapabilityManager — explicit capability definitions
* PolicyEngine — deterministic, auditable decisions
* PolicyProfiles — Restricted / Standard / Advanced /
  Fully Authorized (fully authorized ≠ bypass)
* ResourcePolicy — filesystem roots, browser domains,
  connector scopes, computer apps
* TemporaryGrantStore — expiring grants, auto-revoked
* PolicyVersionStore — every decision carries its version
* PolicySimulator — dry-run without side effects
* CredentialVault (+ leases) — secrets live here only
* SecretRedactor — central redaction authority
* TrustBoundary / InstructionBoundary — external content
  is data, never instructions
* EmergencyStop — global kill switch
* ExecutionSandbox — generated code runs here, never on
  the host
* SecurityAuditCenter — hash-chained, redacted audit
* RateLimiter + ResourceGovernor — abuse controls
* CapabilityEscalation — the honest path to more power

Security-relevant state is summarized into AgentState /
TaskManager metadata in redacted form only — credentials
and raw secrets are never persisted there.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from afnan_ai.redaction import redact_value
from afnan_ai.security.audit import AuditLogger
from afnan_ai.security.audit_center import (
    SecurityAuditCenter,
)
from afnan_ai.security.capabilities import (
    CapabilityManager,
)
from afnan_ai.security.emergency import EmergencyStop
from afnan_ai.security.escalation import (
    CapabilityEscalation,
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
    AuthDecision,
    RiskLevel,
    SecurityEventType,
    TrustLevel,
)
from afnan_ai.security.permissions import PermissionManager
from afnan_ai.security.platform import current_profile
from afnan_ai.security.policy import SecurityPolicyEngine
from afnan_ai.security.profiles import (
    PolicyProfile,
    describe_profile,
    grants_for,
)
from afnan_ai.security.redactor import SecretRedactor
from afnan_ai.security.resources import ResourcePolicy
from afnan_ai.security.sandbox import (
    DEFAULT_SANDBOX_POLICY,
    SandboxPolicy,
)
from afnan_ai.security.sandbox_exec import (
    DEFAULT_SANDBOX as EXEC_SANDBOX,
)
from afnan_ai.security.sandbox_exec import ExecutionSandbox
from afnan_ai.security.simulator import PolicySimulator
from afnan_ai.security.temporary import TemporaryGrantStore
from afnan_ai.security.trust import (
    InstructionBoundary,
    TrustBoundary,
)
from afnan_ai.security.vault import (
    CredentialVault,
    LeasedCredentialVault,
    MemoryCredentialVault,
)
from afnan_ai.security.versioning import PolicyVersionStore


class SecurityCenter:
    """Central, mandatory security authority."""

    def __init__(
        self,
        *,
        permissions: PermissionManager | None = None,
        audit_path: str | Path | None = None,
        vault: CredentialVault | None = None,
        rate_policy: RatePolicy | None = None,
        resource_limits: ResourceLimits | None = None,
        sandbox_policy: SandboxPolicy | None = None,
        resource_policy: ResourcePolicy | None = None,
        profile: PolicyProfile | str = PolicyProfile.STANDARD,
        approver: Callable[[str], bool] | None = None,
        on_security_event: (
            Callable[[dict[str, Any]], None] | None
        ) = None,
        strict_agent_approval: bool = False,
    ) -> None:
        self.permissions = permissions or PermissionManager()
        self.capabilities = CapabilityManager()
        self.resource_policy = (
            resource_policy or ResourcePolicy()
        )
        self.temporary_grants = TemporaryGrantStore()
        self.versions = PolicyVersionStore()
        self.emergency = EmergencyStop()
        self.redactor = SecretRedactor()
        self.boundary = InstructionBoundary(
            on_suspicious=self._boundary_suspicious
        )
        audit_log = AuditLogger(
            audit_path, on_event=on_security_event
        )
        self.audit = audit_log
        self.audit_center = SecurityAuditCenter(
            audit_log, self.redactor,
            version_provider=(
                lambda: self.versions.current().version_id
            ),
        )
        self.vault = vault or LeasedCredentialVault(
            on_access=self._vault_access_event
        )
        self.rate_limiter = RateLimiter(rate_policy)
        self.governor = ResourceGovernor(resource_limits)
        self.sandbox_policy = (
            sandbox_policy or DEFAULT_SANDBOX_POLICY
        )
        self.exec_sandbox = EXEC_SANDBOX
        self.platform = current_profile()
        self.agent_actor = Actor(
            kind=ActorKind.AGENT, actor_id="main"
        )
        self.profile = PolicyProfile.coerce(profile)
        self.policy = SecurityPolicyEngine(
            permissions=self.permissions,
            audit=self.audit,
            rate_limiter=self.rate_limiter,
            approver=approver,
            on_security_event=on_security_event,
            agent_actor_label=self.agent_actor.label,
            strict_agent_approval=strict_agent_approval,
            capabilities=self.capabilities,
            resource_policy=self.resource_policy,
            temporary_grants=self.temporary_grants,
            versions=self.versions,
            emergency=self.emergency,
        )
        self.simulator = PolicySimulator(self.policy)
        self.escalation = CapabilityEscalation(self)
        # Publish the initial policy version and apply the
        # default profile's capability grants.
        self._publish_version("initial policy")
        self.apply_profile(self.profile)

    # -- the choke point ------------------------------------------------------
    def authorize(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        actor: Actor | str | None = None,
        task_id: str = "",
        required_capability: str = "",
        capability_id: str = "",
        resource: dict[str, Any] | None = None,
        argument_trust: TrustLevel | str = (
            TrustLevel.TOOL_OUTPUT
        ),
    ) -> AuthDecision:
        decision = self.policy.authorize(
            tool_name=tool_name,
            arguments=arguments,
            actor=actor or self.agent_actor,
            task_id=task_id,
            required_capability=required_capability,
            capability_id=capability_id,
            resource=resource,
            argument_trust=argument_trust,
        )
        # Stamp the deciding policy version on the outcome.
        decision.policy_version = self.versions.current(
        ).version_id
        return decision

    def dry_run(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        actor: Actor | str | None = None,
        task_id: str = "",
        capability_id: str = "",
        resource: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Policy simulation — no side effects at all."""
        return self.simulator.dry_run(
            tool_name=tool_name,
            arguments=arguments,
            actor=actor or self.agent_actor,
            task_id=task_id,
            capability_id=capability_id,
            resource=resource,
        )

    # -- capabilities & profiles ------------------------------------------------
    def grant_capability(
        self, actor: Actor | str, capability_id: str
    ) -> None:
        cap = self.capabilities.require(capability_id)
        self.permissions.grant(actor, cap.capability_id)
        self.audit_center.security_event(
            "capability_escalation",
            actor=self._label(actor),
            action="capability_granted",
            capability=cap.capability_id,
            details={"risk": cap.risk_level.value},
        )

    def grant_agent_capabilities(
        self, *capabilities: str
    ) -> None:
        # Backwards-compatible string grants (legacy path).
        self.permissions.grant(self.agent_actor, *capabilities)

    def apply_profile(
        self, profile: PolicyProfile | str, *,
        changelog: str = "",
    ) -> dict[str, Any]:
        """Apply an owner-controlled policy profile.  Grants
        the profile's capabilities to the agent actor and
        publishes a new policy version.  Invariants hold in
        every profile — including fully_authorized."""
        self.profile = PolicyProfile.coerce(profile)
        granted = grants_for(self.profile)
        for cap_id in granted:
            try:
                self.capabilities.require(cap_id)
            except KeyError:
                continue
            self.permissions.grant(
                self.agent_actor, cap_id
            )
        version = self._publish_version(
            changelog
            or f"profile -> {self.profile.value}"
        )
        return {
            "profile": describe_profile(self.profile),
            "policy_version": version.version_id,
        }

    def _publish_version(
        self, changelog: str
    ) -> Any:
        digest = hashlib.sha256(
            ",".join(
                self.permissions.capabilities_for(
                    self.agent_actor
                )
            ).encode()
        ).hexdigest()[:16]
        return self.versions.publish(
            self.profile.value,
            changelog=changelog,
            capabilities_hash=digest,
        )

    def subagent_actor(
        self, subagent_id: str, requested: list[str]
    ) -> Actor:
        """Least-privilege subagent: never more than the
        agent holds."""
        return self.permissions.derive_child(
            self.agent_actor, ActorKind.SUBAGENT,
            subagent_id, requested,
        )

    # -- temporary grants ----------------------------------------------------------
    def temporary_grant(
        self,
        capability_id: str,
        actor: Actor | str,
        *,
        duration_s: float,
        scope: dict[str, Any] | None = None,
        task_id: str = "",
        reason: str = "",
        approved_by: str = "",
    ) -> Any:
        self.capabilities.require(capability_id)
        return self.temporary_grants.grant(
            capability_id,
            self._label(actor),
            duration_s=duration_s,
            scope=scope,
            task_id=task_id,
            reason=reason,
            approved_by=approved_by,
        )

    # -- credentials ------------------------------------------------------------------
    def store_credential(
        self, owner: str, name: str, value: str, *,
        kind: str = "generic",
    ) -> str:
        ref = self.vault.put(owner, name, value, kind=kind)
        self.audit_center.security_event(
            SecurityEventType.CREDENTIAL_ACCESSED,
            actor="system",
            action="credential_stored",
            details={
                "owner": owner,
                "name": name,
                "kind": kind,
            },
        )
        return ref

    def lease_credential(
        self, owner: str, name: str, operation: str, *,
        ttl_s: float = 300.0,
    ) -> str:
        """Single-use operation token — never the raw value."""
        vault = self.vault
        if not hasattr(vault, "lease_credential"):
            raise TypeError(
                "vault does not support leases"
            )
        token = vault.lease_credential(
            owner, name, operation, ttl_s=ttl_s
        )
        self.audit_center.security_event(
            SecurityEventType.CREDENTIAL_ACCESSED,
            actor="system",
            action="credential_leased",
            details={
                "owner": owner,
                "name": name,
                "operation": operation,
            },
        )
        return token

    def _vault_access_event(
        self, owner: str, name: str, kind: str
    ) -> None:
        self.audit_center.security_event(
            SecurityEventType.CREDENTIAL_ACCESSED,
            actor="system",
            action="credential_read",
            details={
                "owner": owner,
                "name": name,
                "kind": kind,
            },
        )

    # -- emergency stop ---------------------------------------------------------------
    def trip_emergency(
        self, reason: str, *, actor: str = "owner"
    ) -> Any:
        record = self.emergency.trip(reason, actor=actor)
        self.audit_center.security_event(
            "emergency_stop",
            actor=actor,
            action="emergency_trip",
            details={
                "reason": reason,
                "halted": record.halted,
            },
        )
        return record

    def reset_emergency(
        self, *, authorized: bool, actor: str = "owner",
        reason: str = "",
    ) -> bool:
        ok = self.emergency.reset(
            authorized=authorized, actor=actor,
            reason=reason,
        )
        self.audit_center.security_event(
            "emergency_stop",
            actor=actor,
            action=(
                "emergency_reset"
                if ok else "emergency_reset_refused"
            ),
            details={"reason": reason},
        )
        return ok

    # -- trust boundary -----------------------------------------------------------------
    def _boundary_suspicious(self, finding: Any) -> None:
        self.audit_center.security_event(
            SecurityEventType.PROMPT_INJECTION,
            actor="boundary",
            action="suspicious_instruction",
            details={
                "boundary": finding.boundary,
                "pattern": finding.pattern,
            },
        )

    # -- state integration ------------------------------------------------------------------
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
        current = self.versions.current()
        return {
            "denials": len(denials),
            "approvals_requested": len(approvals),
            "profile": self.profile.value,
            "policy_version": (
                current.version_id if current else ""
            ),
            "emergency_tripped": self.emergency.is_tripped(),
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
        capability: str = "",
        verification: str = "",
        failure_reason: str = "",
    ) -> None:
        actor_label = (
            actor.label
            if isinstance(actor, Actor) else str(actor)
        )
        self.rate_limiter.record_result(
            actor_label, success
        )
        current = self.versions.current()
        self.audit_center.log_execution(
            actor=actor_label,
            capability=capability,
            action=tool_name,
            task_id=task_id,
            policy_version=(
                current.version_id if current else ""
            ),
            execution_status=(
                "success" if success else "failure"
            ),
            verification_status=verification,
            failure_reason=failure_reason,
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

    def set_approver(
        self, approver: Callable[[str], bool] | None
    ) -> None:
        self.policy.set_approver(approver)

    def run_in_sandbox(
        self, code: str, *, timeout_s: float = 30.0,
        limits: Any = None, **kwargs: Any
    ) -> Any:
        from afnan_ai.security.sandbox_exec import (
            ExecutionSandbox,
            SandboxLimits,
        )

        if limits is None:
            limits = SandboxLimits(
                cpu_seconds=max(1, int(timeout_s)),
                timeout_s=timeout_s,
            )
        sandbox = ExecutionSandbox(limits=limits)
        return sandbox.run_python(code, **kwargs)

    @staticmethod
    def _label(actor: Actor | str) -> str:
        return (
            actor.label
            if isinstance(actor, Actor) else str(actor)
        )


__all__ = ["SecurityCenter"]

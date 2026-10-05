"""SecurityPolicyEngine — the central enforcement flow.

    Agent Decision
      ↓ Action Validation
      ↓ Risk Classification
      ↓ Prompt-Injection Screen
      ↓ Rate-Limit Check
      ↓ Permission Check
      ↓ Human Approval (if required)
      ↓ Execute (outside — the engine never runs tools)
      ↓ Verify (outside)
      ↓ Audit

``authorize`` is the single choke point every subsystem
routes through.  Decisions: allow / deny /
approval_required.  Approval always carries the full
structured request (action, reason, target, risk,
expected consequence, context) and can never be bypassed
by a subagent, background worker or generated skill.
"""

from __future__ import annotations

from typing import Any, Callable

from afnan_ai.redaction import redact_text, redact_value
from afnan_ai.security.audit import AuditLogger
from afnan_ai.security.injection import (
    TrustLevel,
    check_arguments,
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
from afnan_ai.security.risk import classify_action


class SecurityPolicyEngine:
    """Central, mandatory policy enforcement."""

    def __init__(
        self,
        permissions: PermissionManager | None = None,
        audit: AuditLogger | None = None,
        rate_limiter: RateLimiter | None = None,
        *,
        approver: Callable[[str], bool] | None = None,
        on_security_event: (
            Callable[[dict[str, Any]], None] | None
        ) = None,
        agent_actor_label: str | None = None,
        strict_agent_approval: bool = False,
    ) -> None:
        self.permissions = permissions or PermissionManager()
        self.audit = audit or AuditLogger()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.approver = approver
        self._on_event = on_security_event
        # The main agent's own tools carry their own
        # approval gates (tested behavior); the center
        # defers to them unless strict mode is on.
        # Subagents, background workers and skills have no
        # such gates — for them central approval is
        # mandatory and unbypassable.
        self.agent_actor_label = agent_actor_label
        self.strict_agent_approval = strict_agent_approval

    # -- the central flow -------------------------------------------------
    def authorize(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        actor: Actor | str = "agent",
        task_id: str = "",
        required_capability: str = "",
        argument_trust: TrustLevel | str = (
            TrustLevel.TOOL_OUTPUT
        ),
    ) -> AuthDecision:
        """Run the full policy flow for one action."""
        actor_label = (
            actor.label
            if isinstance(actor, Actor) else str(actor)
        )
        args = dict(arguments or {})
        risk = classify_action(tool_name, args)
        target = self._target_of(tool_name, args)

        # 1. Prompt-injection screen on arguments.
        injection = check_arguments(args, argument_trust)
        if injection:
            self.rate_limiter.record_injection_hit(
                actor_label
            )
            self._event(
                SecurityEventType.PROMPT_INJECTION,
                actor_label, tool_name, target, task_id,
                risk, {"findings": injection[:3]},
            )
            self._event(
                SecurityEventType.SUSPICIOUS_INSTRUCTION,
                actor_label, tool_name, target, task_id,
                risk, {"findings": injection[:3]},
            )
            decision = AuthDecision.deny(
                "prompt-injection pattern in tool "
                "arguments — treated as data, not "
                "instructions",
                risk=risk,
            )
            self._audit_decision(
                decision, actor_label, tool_name, target,
                task_id, risk,
            )
            return decision

        # 2. Rate limits.
        rate = self.rate_limiter.check(actor_label)
        if not rate["ok"]:
            self.rate_limiter.record_denial(actor_label)
            self._event(
                SecurityEventType.RATE_LIMITED,
                actor_label, tool_name, target, task_id,
                risk, {"code": rate["code"]},
            )
            decision = AuthDecision.deny(
                f"rate limit: {rate['reason']}", risk=risk
            )
            self._audit_decision(
                decision, actor_label, tool_name, target,
                task_id, risk,
            )
            return decision

        # 3. Permission check (capability-based).
        capability = required_capability or self._capability_for(
            tool_name
        )
        if not self.permissions.check(actor, capability):
            self.rate_limiter.record_denial(actor_label)
            self._event(
                SecurityEventType.PERMISSION_DENIED,
                actor_label, tool_name, target, task_id,
                risk,
                {"required_capability": capability},
            )
            pause = self.rate_limiter.should_pause_task(
                actor_label
            )
            if pause["pause"]:
                self._event(
                    SecurityEventType.REPEATED_FAILED_ACTION,
                    actor_label, tool_name, target,
                    task_id, risk, pause,
                )
            decision = AuthDecision.deny(
                f"actor {actor_label!r} lacks capability "
                f"{capability!r}",
                risk=risk,
                capability=capability,
            )
            self._audit_decision(
                decision, actor_label, tool_name, target,
                task_id, risk,
            )
            return decision

        # 4. Human approval for sensitive/irreversible.
        # Non-agent actors (subagents, background workers,
        # skills) always go through central approval — they
        # have no per-tool gates to defer to, and must not
        # bypass the center.  The main agent defers to its
        # tools' own tested approval gates unless strict
        # mode is enabled.
        is_agent = (
            self.agent_actor_label is not None
            and actor_label == self.agent_actor_label
        )
        if risk in (
            RiskLevel.SENSITIVE, RiskLevel.IRREVERSIBLE
        ) and not (
            is_agent and not self.strict_agent_approval
        ):
            request = ApprovalRequest(
                action=f"{tool_name}({self._args_summary(args)})",
                reason=(
                    f"{risk.value} action requested by "
                    f"{actor_label}"
                ),
                target=target,
                risk_level=risk,
                expected_consequence=(
                    self._consequence(tool_name, args, risk)
                ),
                context={
                    "task_id": task_id,
                    "capability": capability,
                },
                actor=actor_label,
            )
            self._event(
                SecurityEventType.APPROVAL_REQUESTED,
                actor_label, tool_name, target, task_id,
                risk, {"request": request.summary()},
            )
            decision = self._resolve_approval(request)
            if decision.action == "allow":
                self._event(
                    SecurityEventType.APPROVAL_GRANTED,
                    actor_label, tool_name, target,
                    task_id, risk, {},
                )
            else:
                self._event(
                    SecurityEventType.APPROVAL_REJECTED,
                    actor_label, tool_name, target,
                    task_id, risk,
                    {"code": decision.action},
                )
            self._audit_decision(
                decision, actor_label, tool_name, target,
                task_id, risk,
            )
            return decision

        # 5. Allow (read-only / low-risk).
        self.rate_limiter.record_call(actor_label)
        self._event(
            SecurityEventType.ACTION_AUTHORIZED,
            actor_label, tool_name, target, task_id,
            risk, {"capability": capability},
        )
        decision = AuthDecision.allow(
            risk,
            f"authorized: {capability} for {actor_label}",
        )
        self._audit_decision(
            decision, actor_label, tool_name, target,
            task_id, risk,
        )
        return decision

    # -- approval -----------------------------------------------------------
    def _resolve_approval(
        self, request: ApprovalRequest
    ) -> AuthDecision:
        if self.approver is None:
            return AuthDecision.need_approval(request)
        try:
            allowed = self.approver(
                redact_text(request.summary())
            )
        except Exception:
            allowed = False
        if allowed:
            self.rate_limiter.record_call(request.actor)
            return AuthDecision.allow(
                request.risk_level, "human approved"
            )
        return AuthDecision.deny(
            "human rejected the approval request",
            risk=request.risk_level,
        )

    def set_approver(
        self, approver: Callable[[str], bool] | None
    ) -> None:
        self.approver = approver

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _capability_for(tool_name: str) -> str:
        name = str(tool_name or "").strip().lower()
        # tool "browser_navigate" → capability "browser.navigate"
        if "_" in name:
            domain, _, rest = name.partition("_")
            return f"{domain}.{rest or 'use'}"
        return f"{name}.use"

    @staticmethod
    def _target_of(
        tool_name: str, args: dict[str, Any]
    ) -> str:
        for key in (
            "url", "path", "file", "dest", "to", "target",
            "tab_id", "artifact_id",
        ):
            value = args.get(key)
            if value:
                return redact_text(str(value))[:160]
        return ""

    @staticmethod
    def _args_summary(args: dict[str, Any]) -> str:
        parts = []
        for key, value in list(args.items())[:4]:
            parts.append(
                f"{key}={redact_text(str(value))[:40]}"
            )
        return ", ".join(parts)

    @staticmethod
    def _consequence(
        tool_name: str, args: dict[str, Any], risk: RiskLevel
    ) -> str:
        if risk is RiskLevel.IRREVERSIBLE:
            return (
                "This cannot be undone. The target will "
                "be permanently changed or removed."
            )
        if risk is RiskLevel.SENSITIVE:
            return (
                "An external side effect will occur "
                f"({tool_name})."
            )
        return "A reversible local change."

    def _audit_decision(
        self, decision: AuthDecision, actor_label: str,
        tool_name: str, target: str, task_id: str,
        risk: RiskLevel,
    ) -> None:
        approval_status = ""
        if decision.action == "approval_required":
            approval_status = "requested"
        elif (
            decision.action == "allow"
            and decision.reason == "human approved"
        ):
            approval_status = "granted"
        elif decision.action == "deny" and (
            "rejected" in decision.reason
            or "approval" in decision.reason
        ):
            approval_status = "rejected"
        self.audit.log(
            "authorization_decision",
            actor=actor_label,
            action=tool_name,
            target=target,
            task_id=task_id,
            risk_level=risk.value,
            permission_result=decision.action,
            approval_status=approval_status,
            details={
                "reason": decision.reason,
                "required_capability": (
                    decision.required_capability
                ),
            },
        )

    def _event(
        self, event: SecurityEventType, actor: str,
        action: str, target: str, task_id: str,
        risk: RiskLevel, details: dict[str, Any],
    ) -> None:
        record = {
            "event": event.value,
            "actor": actor,
            "action": action,
            "target": target,
            "task_id": task_id,
            "risk": risk.value,
            "details": redact_value(details),
        }
        self.audit.log(
            event.value,
            actor=actor, action=action, target=target,
            task_id=task_id, risk_level=risk.value,
            details=details,
        )
        if self._on_event:
            try:
                self._on_event(record)
            except Exception:
                pass


def derive_subagent_actor(
    permissions: PermissionManager,
    parent: Actor,
    subagent_id: str,
    requested: list[str],
) -> Actor:
    """Least-privilege subagent actor: capabilities are the
    intersection of requested and parent-held."""
    return permissions.derive_child(
        parent, ActorKind.SUBAGENT, subagent_id, requested
    )

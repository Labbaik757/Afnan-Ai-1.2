"""SecurityPolicyEngine — the central enforcement flow.

    User / Task
      → Requested Capability
      → Resource
      → Risk
      → Current Permissions
      → Policy (profile + version)
      → Approval Requirement
      → Allow / Deny / Approval Required

``evaluate()`` is the single, deterministic, auditable
decision function.  ``authorize()`` wraps it with the
side effects (rate accounting, security events, audit,
approval callbacks).  ``dry_run=True`` evaluates with
zero side effects for the PolicySimulator.

Two resolution modes, one engine:

* capability mode — the action maps to an explicitly
  defined Capability AND the actor holds it (grant or
  temporary grant): resource scoping, capability risk and
  approval requirements apply.
* legacy mode — otherwise: the previous deterministic
  behavior (tool-name classification + string permission
  check) is preserved exactly.

The emergency stop is checked first: when tripped, every
evaluation denies until an authorized reset.
"""

from __future__ import annotations

from typing import Any, Callable

from afnan_ai.redaction import redact_text, redact_value
from afnan_ai.security.audit import AuditLogger
from afnan_ai.security.capabilities import (
    Capability,
    CapabilityManager,
)
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
from afnan_ai.security.resources import (
    ResourcePolicy,
    check_resource,
)
from afnan_ai.security.risk import classify_action

# Conservative tool → capability mapping.  Unmapped tools
# use the legacy path — never invented capabilities.
TOOL_CAPABILITY_MAP: dict[str, str] = {
    "browser_read": "browser.read",
    "browser_observe": "browser.read",
    "browser_extract": "browser.read",
    "browser_search": "browser.read",
    "browser_navigate": "browser.navigate",
    "browser_click": "browser.click",
    "browser_type": "browser.click",
    "browser_submit": "browser.click",
    "browser_download": "browser.download",
    "computer_observe": "browser.read",
    "computer_locate": "browser.read",
    "computer_screenshot": "browser.read",
    "screen_observe": "browser.read",
    "screen_describe": "browser.read",
    "computer_click": "computer.input",
    "computer_type": "computer.input",
    "computer_key_press": "computer.input",
    "computer_hotkey": "computer.input",
    "file_list": "filesystem.read",
    "file_find_downloads": "filesystem.read",
    "file_save_text": "filesystem.write",
    "file_create_folder": "filesystem.write",
    "note_save": "filesystem.write",
    "file_delete": "filesystem.delete",
    "email_send": "connector.email.send",
    "message_send": "connector.email.send",
    "connector_execute": "connector.email.send",
    "skill_execute": "skill.execute",
    "code_sandbox_execute": "code.sandbox.execute",
    "artifact_create": "artifact.create",
    "artifact_update": "artifact.create",
    "artifact_delete": "artifact.delete",
    "memory_write": "memory.write",
}


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
        capabilities: CapabilityManager | None = None,
        resource_policy: ResourcePolicy | None = None,
        temporary_grants: Any | None = None,
        versions: Any | None = None,
        emergency: Any | None = None,
    ) -> None:
        self.permissions = permissions or PermissionManager()
        self.audit = audit or AuditLogger()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.approver = approver
        self._on_event = on_security_event
        self.agent_actor_label = agent_actor_label
        # Retired: the policy engine is the sole
        # authorization source with no main-agent carve-out.
        # Accepted for compatibility; ignored.
        _ = strict_agent_approval
        self.strict_agent_approval = True
        self.capabilities = capabilities or CapabilityManager()
        self.resource_policy = (
            resource_policy or ResourcePolicy()
        )
        self.temporary_grants = temporary_grants
        self.versions = versions
        self.emergency = emergency

    # -- the single decision function --------------------------------------
    def evaluate(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        actor: Actor | str = "agent",
        task_id: str = "",
        capability_id: str = "",
        resource: dict[str, Any] | None = None,
        argument_trust: TrustLevel | str = (
            TrustLevel.TOOL_OUTPUT
        ),
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Deterministic policy decision.  With dry_run=True
        nothing is mutated: no audit, no rate accounting, no
        approval callbacks, no events."""
        actor_label = (
            actor.label
            if isinstance(actor, Actor) else str(actor)
        )
        args = dict(arguments or {})

        # 0. Emergency stop — everything halts.
        if (
            self.emergency is not None
            and self.emergency.is_tripped()
        ):
            return self._result(
                "deny", "emergency_stop", actor_label,
                tool_name, task_id,
                risk=RiskLevel.SENSITIVE,
                reason=(
                    "emergency stop is tripped — all "
                    "actions halted until authorized reset"
                ),
                events=[{
                    "type": SecurityEventType.POLICY_VIOLATION,
                    "details": {"stage": "emergency_stop"},
                }],
            )

        # 1. Capability resolution.
        cap = self._resolve_capability(
            tool_name, capability_id
        )
        if cap is not None and self._holds_capability(
            actor_label, cap.capability_id
        ):
            return self._capability_evaluate(
                cap=cap,
                tool_name=tool_name,
                args=args,
                actor_label=actor_label,
                task_id=task_id,
                resource=resource,
                argument_trust=argument_trust,
                dry_run=dry_run,
            )
        return self._legacy_evaluate(
            tool_name=tool_name,
            args=args,
            actor_label=actor_label,
            task_id=task_id,
            argument_trust=argument_trust,
            capability_hint=capability_id,
            dry_run=dry_run,
        )

    # -- capability mode -----------------------------------------------------
    def _capability_evaluate(
        self, *, cap: Capability, tool_name: str,
        args: dict[str, Any], actor_label: str,
        task_id: str, resource: dict[str, Any] | None,
        argument_trust: TrustLevel | str,
        dry_run: bool,
    ) -> dict[str, Any]:
        risk = cap.risk_level
        target = self._target_of(tool_name, args)
        merged_resource = {
            **self._resource_from_args(tool_name, args),
            **(resource or {}),
        }

        # Temporary-grant scope: a grant held only
        # temporarily may carry path/url/account prefixes that
        # bound where it can be used.
        if not self.permissions.check(
            actor_label, cap.capability_id
        ):
            grant = self._active_grant(
                actor_label, cap.capability_id)
            if grant is not None:
                scope_ok, scope_reason = (
                    self._check_grant_scope(
                        grant.scope, merged_resource
                    )
                )
                if not scope_ok:
                    return self._result(
                        "deny", "grant_scope", actor_label,
                        tool_name, task_id,
                        risk=risk,
                        reason=(
                            "temporary grant scope: "
                            f"{scope_reason}"
                        ),
                        capability_id=cap.capability_id,
                        resource=merged_resource,
                        target=target,
                        rate_action="denial",
                        events=[{
                            "type": (
                                SecurityEventType
                                .PERMISSION_DENIED
                            ),
                            "details": {
                                "stage": "grant_scope",
                                "capability":
                                    cap.capability_id,
                            },
                        }],
                        escalation=self._escalation_info(
                            cap, merged_resource
                        ),
                    )

        # Resource scoping.
        ok, resource_reason = check_resource(
            cap.capability_id, merged_resource,
            self.resource_policy,
        )
        if not ok:
            return self._result(
                "deny", "resource", actor_label,
                tool_name, task_id,
                risk=risk,
                reason=f"resource policy: {resource_reason}",
                capability_id=cap.capability_id,
                resource=merged_resource,
                target=target,
                rate_action="denial",
                events=[{
                    "type": SecurityEventType.PERMISSION_DENIED,
                    "details": {
                        "stage": "resource",
                        "capability": cap.capability_id,
                    },
                }],
                escalation=self._escalation_info(
                    cap, merged_resource
                ),
            )

        # Injection screen.
        injection = check_arguments(args, argument_trust)
        if injection:
            return self._result(
                "deny", "injection", actor_label,
                tool_name, task_id,
                risk=risk,
                reason=(
                    "prompt-injection pattern in tool "
                    "arguments — treated as data, not "
                    "instructions"
                ),
                capability_id=cap.capability_id,
                resource=merged_resource,
                target=target,
                rate_action="injection",
                events=[
                    {
                        "type": SecurityEventType.PROMPT_INJECTION,
                        "details": {
                            "findings": injection[:3]
                        },
                    },
                    {
                        "type": SecurityEventType.SUSPICIOUS_INSTRUCTION,
                        "details": {
                            "findings": injection[:3]
                        },
                    },
                ],
            )

        # Rate limits.
        rate = self.rate_limiter.check(actor_label)
        if not rate["ok"]:
            return self._result(
                "deny", "rate", actor_label, tool_name,
                task_id,
                risk=risk,
                reason=f"rate limit: {rate['reason']}",
                capability_id=cap.capability_id,
                resource=merged_resource,
                target=target,
                rate_action="denial",
                events=[{
                    "type": SecurityEventType.RATE_LIMITED,
                    "details": {"code": rate["code"]},
                }],
            )

        # Approval.
        needs_approval = (
            cap.approval_requirement in ("human", "owner")
            or risk
            in (RiskLevel.SENSITIVE, RiskLevel.IRREVERSIBLE)
        )
        # The policy engine is the sole authorization
        # source: every call is evaluated here (injection,
        # rate limits, resources, capabilities, audit).  For
        # the main agent, sensitive/irreversible tools are
        # ALLOWED at this layer and their own tested approval
        # gates collect the human decision — the center audits
        # the deferral instead of double-gating.  There is no
        # flag to change this: the behavior is uniform.
        # (The historical strict_agent_approval flag is
        # retired; accepted for compatibility, ignored.)
        is_agent = (
            self.agent_actor_label is not None
            and actor_label == self.agent_actor_label
        )
        if needs_approval and not is_agent:
            request = ApprovalRequest(
                action=(
                    f"{tool_name}"
                    f"({self._args_summary(args)})"
                ),
                reason=(
                    f"capability {cap.capability_id} "
                    f"({risk.value}) requested by "
                    f"{actor_label}"
                ),
                target=target,
                risk_level=risk,
                expected_consequence=self._consequence(
                    tool_name, args, risk
                ),
                context={
                    "task_id": task_id,
                    "capability": cap.capability_id,
                    "resource": merged_resource,
                },
                actor=actor_label,
            )
            return self._result(
                "approval_required", "approval",
                actor_label, tool_name, task_id,
                risk=risk,
                reason="human approval required",
                capability_id=cap.capability_id,
                resource=merged_resource,
                target=target,
                approval_request=request,
                events=[{
                    "type": SecurityEventType.APPROVAL_REQUESTED,
                    "details": {
                        "request": request.summary()
                    },
                }],
            )

        return self._result(
            "allow", "allow", actor_label, tool_name,
            task_id,
            risk=risk,
            reason=(
                f"authorized: {cap.capability_id} for "
                f"{actor_label}"
            ),
            capability_id=cap.capability_id,
            resource=merged_resource,
            target=target,
            rate_action="call",
            events=[{
                "type": SecurityEventType.ACTION_AUTHORIZED,
                "details": {
                    "capability": cap.capability_id
                },
            }],
        )

    # -- legacy mode (behavior preserved exactly) -----------------------------
    def _legacy_evaluate(
        self, *, tool_name: str, args: dict[str, Any],
        actor_label: str, task_id: str,
        argument_trust: TrustLevel | str,
        capability_hint: str, dry_run: bool,
    ) -> dict[str, Any]:
        risk = classify_action(tool_name, args)
        target = self._target_of(tool_name, args)

        injection = check_arguments(args, argument_trust)
        if injection:
            return self._result(
                "deny", "injection", actor_label,
                tool_name, task_id,
                risk=risk,
                reason=(
                    "prompt-injection pattern in tool "
                    "arguments — treated as data, not "
                    "instructions"
                ),
                target=target,
                rate_action="injection",
                events=[
                    {
                        "type": SecurityEventType.PROMPT_INJECTION,
                        "details": {
                            "findings": injection[:3]
                        },
                    },
                    {
                        "type": SecurityEventType.SUSPICIOUS_INSTRUCTION,
                        "details": {
                            "findings": injection[:3]
                        },
                    },
                ],
            )

        rate = self.rate_limiter.check(actor_label)
        if not rate["ok"]:
            return self._result(
                "deny", "rate", actor_label, tool_name,
                task_id,
                risk=risk,
                reason=f"rate limit: {rate['reason']}",
                target=target,
                rate_action="denial",
                events=[{
                    "type": SecurityEventType.RATE_LIMITED,
                    "details": {"code": rate["code"]},
                }],
            )

        capability = (
            capability_hint
            or TOOL_CAPABILITY_MAP.get(
                str(tool_name or "").strip().lower(), ""
            )
            or self._capability_for(tool_name)
        )
        if not self.permissions.check(
            actor_label, capability
        ):
            events: list[dict[str, Any]] = [{
                "type": SecurityEventType.PERMISSION_DENIED,
                "details": {
                    "required_capability": capability
                },
            }]
            pause = self.rate_limiter.should_pause_task(
                actor_label
            )
            if pause["pause"]:
                events.append({
                    "type": SecurityEventType.REPEATED_FAILED_ACTION,
                    "details": dict(pause),
                })
            return self._result(
                "deny", "permission", actor_label,
                tool_name, task_id,
                risk=risk,
                reason=(
                    f"actor {actor_label!r} lacks capability "
                    f"{capability!r}"
                ),
                required_capability=capability,
                target=target,
                rate_action="denial",
                events=events,
                escalation={
                    "capability_id": capability,
                    "how": (
                        "request a temporary grant or owner "
                        "grant; no alternative privileged "
                        "path will be used"
                    ),
                },
            )

        # No flag: the main agent's sensitive tools defer
        # to their own tested approval gates (see above);
        # every other actor needs central approval.
        is_agent = (
            self.agent_actor_label is not None
            and actor_label == self.agent_actor_label
        )
        if risk in (
            RiskLevel.SENSITIVE, RiskLevel.IRREVERSIBLE
        ) and not is_agent:
            request = ApprovalRequest(
                action=(
                    f"{tool_name}"
                    f"({self._args_summary(args)})"
                ),
                reason=(
                    f"{risk.value} action requested by "
                    f"{actor_label}"
                ),
                target=target,
                risk_level=risk,
                expected_consequence=self._consequence(
                    tool_name, args, risk
                ),
                context={
                    "task_id": task_id,
                    "capability": capability,
                },
                actor=actor_label,
            )
            return self._result(
                "approval_required", "approval",
                actor_label, tool_name, task_id,
                risk=risk,
                reason="human approval required",
                required_capability=capability,
                target=target,
                approval_request=request,
                events=[{
                    "type": SecurityEventType.APPROVAL_REQUESTED,
                    "details": {
                        "request": request.summary()
                    },
                }],
            )

        return self._result(
            "allow", "allow", actor_label, tool_name,
            task_id,
            risk=risk,
            reason=(
                f"authorized: {capability} for {actor_label}"
            ),
            required_capability=capability,
            target=target,
            rate_action="call",
            events=[{
                "type": SecurityEventType.ACTION_AUTHORIZED,
                "details": {"capability": capability},
            }],
        )

    # -- authorize: evaluate + side effects -----------------------------------
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
        capability_id: str = "",
        resource: dict[str, Any] | None = None,
    ) -> AuthDecision:
        """Evaluate the policy and apply side effects."""
        result = self.evaluate(
            tool_name=tool_name,
            arguments=arguments,
            actor=actor,
            task_id=task_id,
            capability_id=capability_id or required_capability,
            resource=resource,
            argument_trust=argument_trust,
            dry_run=False,
        )
        actor_label = result["actor_label"]
        target = result.get("target", "")
        risk = RiskLevel.coerce(result.get("risk", ""))

        # Rate accounting.
        rate_action = result.get("rate_action")
        if rate_action == "call":
            self.rate_limiter.record_call(actor_label)
        elif rate_action == "denial":
            self.rate_limiter.record_denial(actor_label)
        elif rate_action == "injection":
            self.rate_limiter.record_injection_hit(
                actor_label
            )

        # Security events.
        for event in result.get("events", []):
            self._event(
                event["type"], actor_label, tool_name,
                target, task_id, risk,
                event.get("details", {}),
            )

        decision = result["decision"]
        if decision == "approval_required":
            request = result["approval_request"]
            resolved = self._resolve_approval(request)
            if resolved.action == "allow":
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
                    {"code": resolved.action},
                )
            self._audit_decision(
                resolved, actor_label, tool_name, target,
                task_id, risk,
                policy_version=result.get(
                    "policy_version", ""
                ),
                capability=result.get(
                    "capability_id",
                    result.get("required_capability", ""),
                ),
            )
            return resolved

        if decision == "deny":
            denied = AuthDecision.deny(
                result.get("reason", "denied"),
                risk=risk,
                capability=result.get(
                    "required_capability", ""
                ),
            )
            self._audit_decision(
                denied, actor_label, tool_name, target,
                task_id, risk,
                policy_version=result.get(
                    "policy_version", ""
                ),
                capability=result.get(
                    "capability_id",
                    result.get("required_capability", ""),
                ),
            )
            return denied

        allowed = AuthDecision.allow(
            risk, result.get("reason", "")
        )
        self._audit_decision(
            allowed, actor_label, tool_name, target,
            task_id, risk,
            policy_version=result.get("policy_version", ""),
            capability=result.get(
                "capability_id",
                result.get("required_capability", ""),
            ),
        )
        return allowed

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

    # -- internals --------------------------------------------------------------
    def _result(
        self, decision: str, stage: str, actor_label: str,
        tool_name: str, task_id: str, *,
        risk: RiskLevel, reason: str = "",
        capability_id: str = "",
        required_capability: str = "",
        resource: dict[str, Any] | None = None,
        target: str = "",
        rate_action: str | None = None,
        events: list[dict[str, Any]] | None = None,
        approval_request: ApprovalRequest | None = None,
        escalation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = (
            self.versions.current()
            if self.versions is not None
            else None
        )
        needs_approval = decision == "approval_required"
        return {
            "decision": decision,
            "stage": stage,
            "actor_label": actor_label,
            "tool": tool_name,
            "task_id": task_id,
            "risk": risk.value,
            "reason": reason,
            "capability_id": capability_id,
            "required_capability": (
                required_capability or capability_id
            ),
            "resource": dict(resource or {}),
            "target": target,
            "rate_action": rate_action,
            "events": list(events or []),
            "approval_required": needs_approval,
            "approval_requirement": (
                "human" if needs_approval else "none"
            ),
            "approval_request": approval_request,
            "escalation": escalation,
            "policy_version": (
                current.version_id if current else ""
            ),
            "dry_run_safe": True,
        }

    def _resolve_capability(
        self, tool_name: str, capability_id: str
    ) -> Capability | None:
        wanted = (
            str(capability_id or "").strip().lower()
            or TOOL_CAPABILITY_MAP.get(
                str(tool_name or "").strip().lower(), ""
            )
        )
        if not wanted:
            return None
        return self.capabilities.get(wanted)

    def _holds_capability(
        self, actor_label: str, capability_id: str
    ) -> bool:
        if self.permissions.check(
            actor_label, capability_id
        ):
            return True
        if self.temporary_grants is not None:
            try:
                return (
                    self.temporary_grants.active_for(
                        actor_label, capability_id
                    )
                    is not None
                )
            except Exception:
                return False
        return False

    def _active_grant(
        self, actor_label: str, capability_id: str
    ) -> Any | None:
        if self.temporary_grants is None:
            return None
        try:
            return self.temporary_grants.active_for(
                actor_label, capability_id
            )
        except Exception:
            return None

    @staticmethod
    def _check_grant_scope(
        scope: dict[str, Any], resource: dict[str, Any]
    ) -> tuple[bool, str]:
        """Enforce temporary-grant scope prefixes."""
        scope = scope or {}
        path_prefix = scope.get("path_prefix")
        if path_prefix:
            path = str(
                resource.get("path")
                or resource.get("target") or ""
            )
            if not path.startswith(str(path_prefix)):
                return (
                    False,
                    f"path {path!r} is outside grant "
                    f"scope {path_prefix!r}",
                )
        url_prefix = scope.get("url_prefix")
        if url_prefix:
            url = str(
                resource.get("url")
                or resource.get("target") or ""
            )
            if not url.startswith(str(url_prefix)):
                return (
                    False,
                    f"url {url!r} is outside grant "
                    f"scope {url_prefix!r}",
                )
        account = scope.get("account")
        if account:
            got = str(
                resource.get("account")
                or resource.get("service") or ""
            )
            if got != str(account):
                return (
                    False,
                    f"account {got!r} is outside grant "
                    f"scope {account!r}",
                )
        return True, ""

    def _escalation_info(
        self, cap: Capability, resource: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "capability_id": cap.capability_id,
            "risk": cap.risk_level.value,
            "approval_requirement": cap.approval_requirement,
            "how": (
                "request a temporary grant via "
                "CapabilityEscalation.request_authorization; "
                "no alternative privileged path will be used"
            ),
            "resource": dict(resource),
        }

    @staticmethod
    def _resource_from_args(
        tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        resource: dict[str, Any] = {}
        for key in (
            "url", "path", "file", "dest", "target",
            "app", "window", "service", "account",
        ):
            value = args.get(key)
            if value:
                resource[key] = value
        # connector tools name the service "connector_id".
        connector_id = args.get("connector_id")
        if connector_id and "service" not in resource:
            resource["service"] = connector_id
        return resource

    @staticmethod
    def _capability_for(tool_name: str) -> str:
        name = str(tool_name or "").strip().lower()
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
        risk: RiskLevel, *,
        policy_version: str = "",
        capability: str = "",
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
            policy_version=policy_version,
            details={
                "reason": decision.reason,
                "required_capability": (
                    decision.required_capability
                ),
                "capability": capability,
            },
        )

    def _current_version_id(self) -> str:
        try:
            current = (
                self.versions.current()
                if self.versions is not None
                else None
            )
            return current.version_id if current else ""
        except Exception:
            return ""

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
            policy_version=self._current_version_id(),
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

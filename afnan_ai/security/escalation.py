"""Capability escalation — the honest path to more power.

When the agent lacks a permission it must NEVER quietly
take an alternative privileged path.  Instead:

    Permission Missing
      → Explain the required capability
      → Request authorization / approval
      → Re-evaluate the policy
      → Execute only if authorized

A denial fails or replans the task safely — it never
sneaks around the policy.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.security.models import Actor


class CapabilityEscalation:
    """Explains denials and runs the re-authorization flow."""

    def __init__(self, center: Any) -> None:
        self._center = center

    def explain_denial(
        self, decision: Any
    ) -> str:
        if hasattr(decision, "required_capability"):
            capability = (
                decision.required_capability or "?"
            )
            resource: dict[str, Any] = {}
            reason = str(decision.reason or "")
            risk = (
                decision.risk_level.value
                if decision.risk_level else ""
            )
            approval_requirement = ""
        else:
            capability = decision.get("capability_id", "?")
            resource = decision.get("resource", {})
            reason = decision.get("reason", "")
            risk = decision.get("risk", "")
            approval_requirement = decision.get(
                "approval_requirement", "")
        lines = [
            f"Missing capability: {capability}",
            f"Reason: {reason}",
        ]
        if resource:
            lines.append(f"Resource: {resource}")
        if risk:
            lines.append(f"Risk level: {risk}")
        if approval_requirement not in (
            None, "", "none",
        ):
            lines.append(
                "This capability needs human approval — "
                "requesting it is the only path forward."
            )
        else:
            lines.append(
                "Ask the owner to grant this capability "
                "(optionally temporary and scoped)."
            )
        lines.append(
            "No alternative privileged path will be used."
        )
        return "\n".join(lines)

    def request_authorization(
        self,
        *,
        capability_id: str,
        actor: Actor | str,
        task_id: str = "",
        reason: str = "",
        duration_s: float = 600.0,
        scope: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ask for a temporary grant.  Returns the grant
        (or the denial) — the caller re-evaluates after."""
        center = self._center
        actor_label = (
            actor.label
            if isinstance(actor, Actor)
            else str(actor)
        )
        cap = center.capabilities.require(capability_id)
        approved = False
        approver = center.policy.approver
        if approver is not None and cap.approval_requirement in (
            "human", "owner",
        ):
            try:
                approved = bool(
                    approver(
                        f"Temporary grant request: "
                        f"{capability_id} for {actor_label}\n"
                        f"Task: {task_id}\nReason: {reason}\n"
                        f"Duration: {duration_s}s\n"
                        f"Risk: {cap.risk_level.value}"
                    )
                )
            except Exception:
                approved = False
        elif cap.approval_requirement == "none":
            # Low-risk capabilities can be self-limited to
            # the task without a human in the loop.
            approved = True
        if not approved:
            center.audit_center.security_event(
                "approval_rejected",
                actor=actor_label,
                action="escalation_request",
                task_id=task_id,
                capability=capability_id,
                details={"reason": reason},
            )
            return {
                "granted": False,
                "reason": "escalation not approved",
            }
        grant = center.temporary_grants.grant(
            capability_id,
            actor_label,
            duration_s=duration_s,
            scope=scope,
            task_id=task_id,
            reason=reason,
            approved_by="owner" if approver else "policy",
        )
        center.audit_center.security_event(
            "capability_escalation",
            actor=actor_label,
            action="escalation_granted",
            task_id=task_id,
            capability=capability_id,
            details={"grant_id": grant.grant_id},
        )
        return {
            "granted": True,
            "grant": grant.to_dict(),
        }

    def reevaluate(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        actor: Actor | str = "agent",
        task_id: str = "",
        capability_id: str = "",
        resource: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Re-run the policy after a grant changed."""
        return self._center.policy.evaluate(
            tool_name=tool_name,
            arguments=arguments,
            actor=actor,
            task_id=task_id,
            capability_id=capability_id,
            resource=resource,
            dry_run=True,
        )

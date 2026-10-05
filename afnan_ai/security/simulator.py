"""Policy simulation / dry run.

Before executing, the agent can ask: *what would the
policy say?*  The simulator runs the exact same decision
logic as enforcement — but with zero side effects:

* no audit writes
* no rate-limiter mutation
* no approval callbacks fired
* no tool execution

The result carries required capabilities, resources,
risk, the policy decision, the approval requirement and
the expected side effects — so the planner can explain
itself and choose safer plans.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.security.models import Actor, RiskLevel


class PolicySimulator:
    """Side-effect-free policy evaluation."""

    def __init__(self, policy_engine: Any) -> None:
        self._engine = policy_engine

    def dry_run(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        actor: Actor | str = "agent",
        task_id: str = "",
        capability_id: str = "",
        resource: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        decision = self._engine.evaluate(
            tool_name=tool_name,
            arguments=arguments,
            actor=actor,
            task_id=task_id,
            capability_id=capability_id,
            resource=resource,
            dry_run=True,
        )
        return {
            "tool": tool_name,
            "actor": (
                actor.label
                if isinstance(actor, Actor)
                else str(actor)
            ),
            "required_capability": decision.get(
                "capability_id", ""
            ),
            "resource": dict(resource or {}),
            "risk": decision.get("risk", ""),
            "policy_decision": decision.get(
                "decision", "deny"
            ),
            "approval_required": decision.get(
                "approval_required", False
            ),
            "approval_requirement": decision.get(
                "approval_requirement", "none"
            ),
            "expected_side_effects": self._side_effects(
                tool_name, decision
            ),
            "reason": decision.get("reason", ""),
            "policy_version": decision.get(
                "policy_version", ""
            ),
            "would_execute": decision.get("decision")
            == "allow"
            and not decision.get("approval_required"),
        }

    @staticmethod
    def _side_effects(
        tool_name: str, decision: dict[str, Any]
    ) -> list[str]:
        risk = str(decision.get("risk", ""))
        name = str(tool_name or "").lower()
        effects: list[str] = []
        if "read" in name or "observe" in name or risk == "read_only":
            effects.append("no state change (read-only)")
            return effects
        if risk == "low_risk_write":
            effects.append("reversible local write")
        if risk == "sensitive":
            effects.append("external side effect (needs approval)")
        if risk == "irreversible":
            effects.append("permanent change (needs approval)")
        if "download" in name:
            effects.append("file written to download directory")
        if "send" in name:
            effects.append("message transmitted to recipient")
        if "delete" in name:
            effects.append("data permanently removed")
        return effects or ["unknown — treated as sensitive"]

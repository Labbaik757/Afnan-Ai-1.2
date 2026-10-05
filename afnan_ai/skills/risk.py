"""Skill risk analysis.

Risk is derived from the tools a skill composes (the maximum
step risk wins) and can be raised explicitly by the author —
never lowered below what the steps imply.  Sensitive and
destructive skills always go through the existing human
approval path; the skill layer never invents its own.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.skills.models import Skill, SkillRisk, SkillStep

#: Tool-name hints for risk classification.  Conservative:
#: unknown tools default to REVERSIBLE so a misclassified
#: write cannot slip through as read-only.
_READ_HINTS = (
    "search", "get", "list", "read", "find", "observe",
    "screenshot", "extract", "lookup", "check", "inspect",
    "capabilities", "health",
)
_SENSITIVE_HINTS = (
    "send", "message", "email", "purchase", "pay", "book",
    "upload", "publish", "post", "share", "connect",
)
_DESTRUCTIVE_HINTS = (
    "delete", "remove", "destroy", "close_application",
    "format", "revoke",
)


def classify_tool(tool_name: str) -> SkillRisk:
    """Heuristic risk for a tool name (conservative)."""
    name = str(tool_name).lower()
    for hint in _DESTRUCTIVE_HINTS:
        if hint in name:
            return SkillRisk.DESTRUCTIVE
    for hint in _SENSITIVE_HINTS:
        if hint in name:
            return SkillRisk.SENSITIVE
    for hint in _READ_HINTS:
        if hint in name:
            return SkillRisk.READ_ONLY
    return SkillRisk.REVERSIBLE


_RANK = {
    SkillRisk.READ_ONLY: 0,
    SkillRisk.REVERSIBLE: 1,
    SkillRisk.SENSITIVE: 2,
    SkillRisk.DESTRUCTIVE: 3,
}


def derive_risk(
    steps: list[SkillStep],
    sub_skill_risks: dict[str, SkillRisk] | None = None,
) -> SkillRisk:
    """Maximum implied risk across a skill's steps.

    Sub-skill steps resolve against *sub_skill_risks* (built
    from the registry); unknown sub-skills stay SENSITIVE.
    """
    sub_skill_risks = sub_skill_risks or {}
    level = SkillRisk.READ_ONLY
    for step in steps:
        if step.tool.startswith("skill:"):
            implied = sub_skill_risks.get(
                step.tool[len("skill:"):], SkillRisk.SENSITIVE
            )
        else:
            implied = classify_tool(step.tool)
        if _RANK[implied] > _RANK[level]:
            level = implied
    return level


def resolve_sub_skill_risks(
    skill: Skill, get_skill, *, _seen: frozenset = frozenset()
) -> dict[str, SkillRisk]:
    """Effective risks of a skill's sub-skills (cycle-safe).

    *get_skill* maps skill_id -> Skill | None (e.g.
    ``registry.get_or_none``).
    """
    resolved: dict[str, SkillRisk] = {}
    for step in skill.steps:
        if not step.tool.startswith("skill:"):
            continue
        sub_id = step.tool[len("skill:"):]
        if sub_id in _seen or sub_id in resolved:
            continue
        sub = get_skill(sub_id)
        if sub is None:
            continue
        sub_risks = resolve_sub_skill_risks(
            sub, get_skill, _seen=_seen | {sub_id}
        )
        resolved[sub_id] = effective_risk(sub, sub_risks)
    return resolved


def effective_risk(
    skill: Skill,
    sub_skill_risks: dict[str, SkillRisk] | None = None,
) -> SkillRisk:
    """Declared risk, never below what the steps imply."""
    implied = derive_risk(skill.steps, sub_skill_risks)
    if _RANK[skill.risk] < _RANK[implied]:
        return implied
    return skill.risk


def needs_approval(risk: SkillRisk) -> bool:
    """Sensitive/destructive work needs a human yes."""
    return risk in (SkillRisk.SENSITIVE, SkillRisk.DESTRUCTIVE)


def risk_report(
    skill: Skill,
    sub_skill_risks: dict[str, SkillRisk] | None = None,
) -> dict[str, Any]:
    """Structured risk analysis for validation/audit."""
    implied = derive_risk(skill.steps, sub_skill_risks)
    declared = skill.risk
    final = effective_risk(skill, sub_skill_risks)
    per_step = []
    for step in skill.steps:
        if step.tool.startswith("skill:"):
            sub_id = step.tool[len("skill:"):]
            resolved = (sub_skill_risks or {}).get(sub_id)
            label = (
                resolved.value
                if resolved is not None
                else "sensitive (unresolved sub-skill)"
            )
        else:
            label = classify_tool(step.tool).value
        per_step.append({
            "step_id": step.step_id,
            "tool": step.tool,
            "implied_risk": label,
        })
    return {
        "skill_id": skill.skill_id,
        "declared_risk": declared.value,
        "implied_risk": implied.value,
        "effective_risk": final.value,
        "needs_approval": needs_approval(final),
        "escalated": _RANK[declared] < _RANK[implied],
        "steps": per_step,
    }

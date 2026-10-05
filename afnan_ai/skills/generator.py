"""Dynamic skill generation pipeline.

The model is NEVER given arbitrary code execution.  When the
agent needs a new capability, the pipeline is:

1. identify the required capability (from the goal text),
2. search existing tools and skills,
3. generate a *composition* (data, not code),
4. strict schema validation,
5. security / risk analysis,
6. sandbox validation,
7. register — only after verification passes, and only via
   an explicit registration call (human approval required
   for sensitive+ risk).

``SkillGenerator.propose`` is deterministic keyword search,
not an LLM call: what it returns is a draft, never a live
skill.  ``build`` runs the full validation gauntlet and
returns a finalized Skill that still needs explicit
registration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afnan_ai.skills.composer import compose_skill
from afnan_ai.skills.models import (
    Skill,
    SkillDependencies,
    SkillStatus,
)
from afnan_ai.skills.registry import SkillRegistry
from afnan_ai.skills.risk import effective_risk, needs_approval
from afnan_ai.skills.sandbox import SkillValidation


@dataclass
class SkillDraft:
    """A proposed composition — not a skill yet."""

    draft_id: str
    goal_text: str
    description: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    matched_tools: list[str] = field(default_factory=list)
    matched_skills: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "goal_text": self.goal_text,
            "description": self.description,
            "steps": list(self.steps),
            "matched_tools": list(self.matched_tools),
            "matched_skills": list(self.matched_skills),
            "input_schema": self.input_schema,
        }


def _keywords(text: str) -> set[str]:
    import re

    return set(
        w for w in re.findall(r"[a-z]{3,}", text.lower())
    )


class SkillGenerator:
    """Propose and build skills from existing capabilities."""

    def __init__(
        self,
        tool_registry: Any,
        skill_registry: SkillRegistry,
    ) -> None:
        self.tools = tool_registry
        self.skills = skill_registry

    # -- 1+2: identify capability, search existing ------------------------------
    def search_capabilities(
        self, query: str, *, limit: int = 8
    ) -> dict[str, list[dict[str, Any]]]:
        """Keyword search over tools and skills."""
        words = _keywords(query)
        tool_hits: list[tuple[float, dict[str, Any]]] = []
        for tool in self.tools.list_tools():
            definition = tool.definition()
            haystack = _keywords(
                f"{tool.name} {tool.description}"
            )
            overlap = len(words & haystack)
            if overlap:
                tool_hits.append((
                    overlap / max(1, len(words)),
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": definition.get(
                            "input_schema", {}
                        ),
                    },
                ))
        skill_hits: list[tuple[float, dict[str, Any]]] = []
        for described in self.skills.describe_for_agent():
            haystack = _keywords(
                f"{described['skill_id']} {described['name']} "
                f"{described['description']}"
            )
            overlap = len(words & haystack)
            if overlap:
                skill_hits.append((
                    overlap / max(1, len(words)),
                    described,
                ))
        tool_hits.sort(key=lambda pair: pair[0], reverse=True)
        skill_hits.sort(key=lambda pair: pair[0], reverse=True)
        return {
            "tools": [hit for _, hit in tool_hits[:limit]],
            "skills": [hit for _, hit in skill_hits[:limit]],
        }

    # -- 3: generate a composition (data, not code) -------------------------------
    def propose(
        self, goal_text: str, *, draft_id: str = "draft1"
    ) -> SkillDraft:
        """Draft a composition for *goal_text*.

        If a matching skill already exists, the draft is empty
        and names it — reuse beats reinvention.
        """
        found = self.search_capabilities(goal_text)
        if found["skills"]:
            best = found["skills"][0]
            return SkillDraft(
                draft_id=draft_id,
                goal_text=goal_text,
                description=(
                    f"Existing skill covers this: "
                    f"{best['skill_id']}"
                ),
                steps=[],
                matched_skills=[best["skill_id"]],
                matched_tools=[t["name"] for t in found["tools"]],
            )
        steps: list[dict[str, Any]] = []
        for index, tool in enumerate(found["tools"][:4]):
            # Required arguments are templated from the
            # skill's input so generated drafts validate.
            schema = tool.get("input_schema", {})
            arguments = {
                key: f"{{{{input.{key}}}}}"
                for key in schema.get("required", [])
            }
            steps.append({
                "step_id": f"s{index + 1}",
                "tool": tool["name"],
                "arguments": arguments,
                "description": tool["description"][:120],
            })
        # The draft's input schema declares every templated
        # field the steps need.
        required_fields = sorted({
            key
            for tool in found["tools"][:4]
            for key in tool.get("input_schema", {}).get(
                "required", []
            )
        })
        input_schema = {
            "type": "object",
            "properties": {
                key: {"type": "string"} for key in required_fields
            },
            "required": required_fields,
        }
        draft = SkillDraft(
            draft_id=draft_id,
            goal_text=goal_text,
            description=f"Composed workflow for: {goal_text[:120]}",
            steps=steps,
            matched_tools=[t["name"] for t in found["tools"]],
            input_schema=input_schema,
        )
        return draft

    # -- 4+5+6: validate, analyze, sandbox-check ------------------------------------
    def build(
        self,
        draft: SkillDraft,
        *,
        skill_id: str,
        name: str,
        input_schema: dict[str, Any] | None = None,
        risk: str | None = None,
        created_by: str = "generator",
    ) -> tuple[Skill, SkillValidation]:
        """Turn a draft into a validated (but unregistered) Skill."""
        if draft.matched_skills and not draft.steps:
            raise ValueError(
                "draft defers to existing skill "
                f"{draft.matched_skills[0]!r}; nothing to build"
            )
        skill = compose_skill(
            skill_id=skill_id,
            name=name,
            description=draft.description,
            steps=draft.steps,
            input_schema=(
                input_schema
                if input_schema is not None
                else draft.input_schema
            ),
            dependencies={
                "tools": sorted({
                    step["tool"] for step in draft.steps
                    if not str(step["tool"]).startswith("skill:")
                }),
                "skills": sorted({
                    str(step["tool"])[len("skill:"):]
                    for step in draft.steps
                    if str(step["tool"]).startswith("skill:")
                }),
            },
            created_by=created_by,
        )
        if risk is not None:
            # Only allow raising the risk here.
            from afnan_ai.skills.models import SkillRisk
            from afnan_ai.skills.risk import (
                _RANK,
                derive_risk,
                resolve_sub_skill_risks,
            )

            wanted = SkillRisk(risk)
            implied = derive_risk(
                skill.steps,
                resolve_sub_skill_risks(
                    skill, self.skills.get_or_none
                ),
            )
            if _RANK[wanted] < _RANK[implied]:
                raise ValueError(
                    "cannot lower a skill's risk below what "
                    "its steps imply"
                )
            skill.risk = wanted
        skill.status = SkillStatus.ACTIVE
        validation = self.skills.validate(skill)
        return skill, validation

    def build_and_register(
        self,
        draft: SkillDraft,
        *,
        skill_id: str,
        name: str,
        input_schema: dict[str, Any] | None = None,
        approver: Any | None = None,
    ) -> Skill:
        """Build + register.  Sensitive+ risk needs a human yes;
        without an approver (or on denial) nothing registers."""
        skill, validation = self.build(
            draft, skill_id=skill_id, name=name,
            input_schema=input_schema,
        )
        if not validation.ok:
            raise ValueError(
                "draft failed validation: "
                + "; ".join(
                    i.message for i in validation.issues[:3]
                )
            )
        if needs_approval(effective_risk(skill)):
            if approver is None:
                raise PermissionError(
                    f"skill {skill_id!r} is "
                    f"{effective_risk(skill).value}: human "
                    "approval required before registration"
                )
            if not approver(
                f"Register skill {skill_id} "
                f"({effective_risk(skill).value})?"
            ):
                raise PermissionError(
                    "human denied skill registration"
                )
        return self.skills.register(skill)

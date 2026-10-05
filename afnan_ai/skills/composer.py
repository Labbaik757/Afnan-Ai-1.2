"""Skill composition — build reusable workflows from tools.

Composition is data-only: a Skill is an ordered list of
(tool, arguments) steps referencing *already registered*
tools (or other skills).  The composer validates every step
against the real ToolRegistry schemas, so a composed skill
can never name a tool that does not exist or pass arguments
the tool would reject.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.skills.models import (
    Skill,
    SkillDependencies,
    SkillRisk,
    SkillStatus,
    SkillStep,
)


def compose_skill(
    *,
    skill_id: str,
    name: str,
    description: str,
    steps: list[dict[str, Any] | SkillStep | tuple],
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    dependencies: dict[str, Any] | SkillDependencies | None = None,
    verification_criteria: list[str] | None = None,
    risk: str | SkillRisk | None = None,
    created_by: str = "developer",
    version: str = "1.0.0",
) -> Skill:
    """Build a Skill from tool steps (not yet registered).

    ``steps`` entries may be ``SkillStep`` objects, dicts, or
    ``(step_id, tool, arguments)`` / ``(step_id, tool,
    arguments, description)`` tuples.
    """
    normalized: list[SkillStep] = []
    for entry in steps:
        if isinstance(entry, SkillStep):
            normalized.append(entry)
        elif isinstance(entry, dict):
            normalized.append(SkillStep.from_dict(entry))
        elif isinstance(entry, (tuple, list)):
            parts = list(entry)
            normalized.append(SkillStep(
                step_id=str(parts[0]),
                tool=str(parts[1]),
                arguments=dict(parts[2]) if len(parts) > 2 else {},
                description=str(parts[3]) if len(parts) > 3 else "",
            ))
        else:
            raise ValueError(
                f"cannot compose step from {entry!r}"
            )
    # NOTE: the declared risk is NOT auto-derived here.
    # effective_risk() escalates it at validation/execution
    # time from the actual steps (resolving sub-skills
    # against the registry), so an unresolved "skill:"
    # reference can never slip through as read-only while a
    # resolved read-only sub-skill is not punished.
    skill = Skill(
        skill_id=skill_id,
        name=name,
        description=description,
        version=version,
        risk=risk or SkillRisk.READ_ONLY,
        input_schema=input_schema or {},
        output_schema=output_schema or {},
        steps=normalized,
        dependencies=(
            dependencies
            if isinstance(dependencies, SkillDependencies)
            else SkillDependencies.from_dict(dependencies or {})
        ),
        verification_criteria=verification_criteria or [],
        status=SkillStatus.DRAFT,
        created_by=created_by,
    )
    # Declared risk may only raise, never lower, the implied
    # risk (enforced again at registration).
    return skill


def research_report_skill(
    skill_id: str = "research_report",
) -> Skill:
    """The spec's example workflow as a composed skill draft."""
    return compose_skill(
        skill_id=skill_id,
        name="Research report",
        description=(
            "Search the web, open relevant pages, extract "
            "information, compare results and draft a "
            "structured report."
        ),
        steps=[
            ("search", "search_google", {"query": "{{input.query}}"},
             "Search the web for the topic"),
            ("extract", "extract_text", {},
             "Extract information from results"),
            ("draft", "save_text", {},
             "Draft the structured report"),
        ],
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "topic": {"type": "string"},
            },
            "required": ["query"],
        },
        verification_criteria=[
            "a report was drafted covering the query topic",
        ],
        created_by="composer.example",
    )

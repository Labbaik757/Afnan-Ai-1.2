"""Skill data models.

A Skill is a named, versioned, reusable workflow composed of
*existing* registered tools (and optionally other skills).
It is deliberately NOT a Tool subclass and never carries
arbitrary code from the model: code-based skills are a
separate, sandboxed, human-gated path (see sandbox.py).

Every model here is serializable metadata.  Secrets are
refused at the recording boundary, and untrusted external
content is never promoted into a definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class SkillRisk(str, Enum):
    """What a skill is allowed to do.

    READ_ONLY   — observes / reads, changes nothing.
    REVERSIBLE  — writes that can be undone (drafts, temp files).
    SENSITIVE   — needs a human yes (messages, purchases,
                  account changes, external side effects).
    DESTRUCTIVE — irreversible (deletes, publishes, sends).
    """

    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    SENSITIVE = "sensitive"
    DESTRUCTIVE = "destructive"


class SkillStatus(str, Enum):
    ACTIVE = "active"
    DISABLED = "disabled"
    DRAFT = "draft"


@dataclass
class SkillStep:
    """One step of a composed skill.

    ``tool`` is either a ToolRegistry tool name or
    ``"skill:<skill_id>"`` for a sub-skill.  Arguments are
    static values, optionally templated from earlier step
    outputs with ``{{steps.<step_id>.<key>}}``.
    """

    step_id: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        self.step_id = str(self.step_id).strip()
        self.tool = str(self.tool).strip()
        if not isinstance(self.arguments, dict):
            raise ValueError("SkillStep arguments must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillStep":
        return cls(
            step_id=str(data.get("step_id", "")),
            tool=str(data.get("tool", "")),
            arguments=dict(data.get("arguments") or {}),
            description=str(data.get("description", "")),
        )


@dataclass
class SkillDependencies:
    """What a skill needs.  Checked before execution; a
    missing dependency fails fast with a structured error."""

    tools: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    connectors: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    # capabilities: "browser" | "computer" | "files" | "network"

    def __post_init__(self) -> None:
        self.tools = tuple(str(t) for t in self.tools)
        self.skills = tuple(str(s) for s in self.skills)
        self.connectors = tuple(str(c) for c in self.connectors)
        self.capabilities = tuple(str(c) for c in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tools": list(self.tools),
            "skills": list(self.skills),
            "connectors": list(self.connectors),
            "capabilities": list(self.capabilities),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillDependencies":
        data = data or {}
        return cls(
            tools=tuple(data.get("tools") or ()),
            skills=tuple(data.get("skills") or ()),
            connectors=tuple(data.get("connectors") or ()),
            capabilities=tuple(data.get("capabilities") or ()),
        )


def _schema(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("schema must be a dict")
    if value.get("type", "object") != "object":
        raise ValueError("skill schemas must be object schemas")
    props = value.get("properties", {})
    if not isinstance(props, dict):
        raise ValueError("schema properties must be a dict")
    required = value.get("required", [])
    if not isinstance(required, list):
        raise ValueError("schema required must be a list")
    unknown_required = [
        key for key in required if key not in props
    ]
    if unknown_required:
        raise ValueError(
            "schema requires undeclared properties: "
            + ", ".join(unknown_required)
        )
    return {
        "type": "object",
        "properties": dict(props),
        "required": list(required),
    }


@dataclass
class Skill:
    """A reusable workflow.  Versioned, auditable, risk-rated."""

    skill_id: str
    name: str
    description: str
    version: str = "1.0.0"
    risk: SkillRisk = SkillRisk.READ_ONLY
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    steps: list[SkillStep] = field(default_factory=list)
    dependencies: SkillDependencies = field(
        default_factory=SkillDependencies
    )
    verification_criteria: list[str] = field(default_factory=list)
    status: SkillStatus = SkillStatus.ACTIVE
    created_by: str = "developer"
    created_at: str = field(default_factory=_utcnow)
    changelog: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.skill_id = str(self.skill_id).strip()
        if not self.skill_id:
            raise ValueError("skill_id is required")
        if isinstance(self.risk, str):
            self.risk = SkillRisk(self.risk)
        if isinstance(self.status, str):
            self.status = SkillStatus(self.status)
        if self.input_schema:
            self.input_schema = _schema(self.input_schema)
        if self.output_schema:
            self.output_schema = _schema(self.output_schema)
        steps = []
        for step in self.steps:
            steps.append(
                step
                if isinstance(step, SkillStep)
                else SkillStep.from_dict(step)
            )
        self.steps = steps
        if not isinstance(self.dependencies, SkillDependencies):
            self.dependencies = SkillDependencies.from_dict(
                self.dependencies
            )
        step_ids = [s.step_id for s in self.steps]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("duplicate step_id in skill steps")

    @property
    def tool_name(self) -> str:
        """Adapter name when exposed in the ToolRegistry."""
        return f"skill_{self.skill_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "risk": self.risk.value,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "steps": [s.to_dict() for s in self.steps],
            "dependencies": self.dependencies.to_dict(),
            "verification_criteria": list(
                self.verification_criteria
            ),
            "status": self.status.value,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "changelog": list(self.changelog),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Skill":
        return cls(
            skill_id=str(data.get("skill_id", "")),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            version=str(data.get("version", "1.0.0")),
            risk=str(data.get("risk", "read_only")),
            input_schema=dict(data.get("input_schema") or {}),
            output_schema=dict(data.get("output_schema") or {}),
            steps=[
                SkillStep.from_dict(s)
                for s in (data.get("steps") or [])
            ],
            dependencies=SkillDependencies.from_dict(
                data.get("dependencies")
            ),
            verification_criteria=list(
                data.get("verification_criteria") or []
            ),
            status=str(data.get("status", "active")),
            created_by=str(data.get("created_by", "developer")),
            created_at=str(
                data.get("created_at", _utcnow())
            ),
            changelog=list(data.get("changelog") or []),
        )


@dataclass
class SkillCandidate:
    """A learned, not-yet-registered workflow draft.

    Candidates are proposals only: they become real skills
    after sandbox validation and explicit registration (with
    human approval for sensitive+ risk).
    """

    candidate_id: str
    description: str
    steps: list[SkillStep]
    evidence_task_ids: list[str] = field(default_factory=list)
    success_count: int = 0
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "evidence_task_ids": list(self.evidence_task_ids),
            "success_count": self.success_count,
            "created_at": self.created_at,
        }

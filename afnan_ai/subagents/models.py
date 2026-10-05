"""Subagent data models.

A subagent is a scoped, least-privilege worker: it runs the
*existing* AgentLoop against a *restricted* view of the
parent's tools, with its own isolated AgentState, its own
resource limits, and a structured result handoff that the
parent verifies instead of trusting.

Roles are capability bundles, never hard-coded workflows:
a role template just names a default set of allowed tools
and risk permissions; every spec can override them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from afnan_ai.skills.models import SkillRisk


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class SubAgentStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TERMINATED = "terminated"


class VerificationState(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


@dataclass
class ResourceLimits:
    """Per-subagent budgets.  All enforced, none advisory."""

    max_steps: int = 10
    timeout_s: float | None = 120.0
    max_tool_calls: int = 30
    max_retries: int = 1
    context_chars: int = 4000  # scoped context budget

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "timeout_s": self.timeout_s,
            "max_tool_calls": self.max_tool_calls,
            "max_retries": self.max_retries,
            "context_chars": self.context_chars,
        }


# Role templates: default capability bundles.  A template is
# *permissions*, not a workflow — the subagent still plans
# and acts through the normal loop with whatever tools it is
# allowed.
ROLE_TEMPLATES: dict[str, dict[str, Any]] = {
    "researcher": {
        "description": "Gathers information from allowed sources.",
        "allowed_tools": [
            "search_google", "browser_navigate", "browser_open",
            "browser_extract", "browser_find", "screen_observe",
        ],
        "risk_permissions": ["read_only"],
    },
    "browser_agent": {
        "description": "Operates the browser.",
        "allowed_tools": ["browser_*", "screen_*"],
        "risk_permissions": ["read_only", "reversible"],
    },
    "computer_agent": {
        "description": "Operates the desktop.",
        "allowed_tools": ["computer_*", "file_*", "screen_*"],
        "risk_permissions": ["read_only", "reversible"],
    },
    "data_analyst": {
        "description": "Analyzes data and produces structured output.",
        "allowed_tools": ["file_*", "search_google"],
        "risk_permissions": ["read_only", "reversible"],
    },
    "file_agent": {
        "description": "Reads/writes files and documents.",
        "allowed_tools": ["file_*"],
        "risk_permissions": ["read_only", "reversible"],
    },
    "verifier_agent": {
        "description": "Checks results against evidence.",
        "allowed_tools": ["browser_*", "file_*", "search_google"],
        "risk_permissions": ["read_only"],
    },
    "planner_agent": {
        "description": "Breaks goals into plans without executing.",
        "allowed_tools": [],
        "risk_permissions": ["read_only"],
    },
}


def _coerce_risks(
    values: Any,
) -> tuple[SkillRisk, ...]:
    out = []
    for value in values or []:
        out.append(
            value
            if isinstance(value, SkillRisk)
            else SkillRisk(str(value))
        )
    return tuple(out)


@dataclass
class SubAgentSpec:
    """Everything needed to create a subagent."""

    subagent_id: str
    role: str
    objective: str
    allowed_tools: list[str] = field(default_factory=list)
    allowed_connectors: list[str] = field(default_factory=list)
    risk_permissions: tuple[SkillRisk, ...] = (
        SkillRisk.READ_ONLY,
        SkillRisk.REVERSIBLE,
    )
    resources: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    limits: ResourceLimits = field(
        default_factory=ResourceLimits
    )
    constraints: list[str] = field(default_factory=list)
    memory_scope: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.subagent_id = str(self.subagent_id).strip()
        if not self.subagent_id:
            raise ValueError("subagent_id is required")
        if not str(self.objective).strip():
            raise ValueError("objective is required")
        self.role = str(self.role or "general")
        self.risk_permissions = _coerce_risks(
            self.risk_permissions
        )
        if not isinstance(self.limits, ResourceLimits):
            raise ValueError("limits must be ResourceLimits")

    @classmethod
    def from_role(
        cls,
        subagent_id: str,
        role: str,
        objective: str,
        **overrides: Any,
    ) -> "SubAgentSpec":
        """Build a spec from a role template (overridable)."""
        template = ROLE_TEMPLATES.get(role, {})
        kwargs: dict[str, Any] = {
            "allowed_tools": list(
                template.get("allowed_tools", [])
            ),
            "risk_permissions": list(
                template.get("risk_permissions",
                             ["read_only", "reversible"])
            ),
        }
        kwargs.update(overrides)
        return cls(
            subagent_id=subagent_id, role=role,
            objective=objective, **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subagent_id": self.subagent_id,
            "role": self.role,
            "objective": self.objective,
            "allowed_tools": list(self.allowed_tools),
            "allowed_connectors": list(self.allowed_connectors),
            "risk_permissions": [
                r.value for r in self.risk_permissions
            ],
            "resources": list(self.resources),
            "depends_on": list(self.depends_on),
            "limits": self.limits.to_dict(),
            "constraints": list(self.constraints),
            "memory_scope": list(self.memory_scope),
        }


@dataclass
class SubAgentHandoff:
    """Structured result a subagent returns to the parent.

    Never trusted blindly: the parent (or a verifier agent)
    checks it against the expected result before merging.
    """

    subagent_id: str
    objective: str
    status: str  # completed | failed | cancelled | terminated
    output: Any = None
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    tool_calls: int = 0
    steps_verified: int = 0
    error: str | None = None
    verification: str = VerificationState.UNVERIFIED.value
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subagent_id": self.subagent_id,
            "objective": self.objective,
            "status": self.status,
            "output": self.output,
            "evidence": list(self.evidence),
            "confidence": self.confidence,
            "tool_calls": self.tool_calls,
            "steps_verified": self.steps_verified,
            "error": self.error,
            "verification": self.verification,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass
class SubAgentRecord:
    """Runtime record for one managed subagent."""

    spec: SubAgentSpec
    status: SubAgentStatus = SubAgentStatus.CREATED
    handoff: SubAgentHandoff | None = None
    attempts: int = 0
    created_at: str = field(default_factory=_utcnow)
    failure_reason: str | None = None
    # Names of resources currently held (for the graph).
    held_resources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "status": self.status.value,
            "handoff": (
                self.handoff.to_dict()
                if self.handoff else None
            ),
            "attempts": self.attempts,
            "created_at": self.created_at,
            "failure_reason": self.failure_reason,
            "held_resources": list(self.held_resources),
        }

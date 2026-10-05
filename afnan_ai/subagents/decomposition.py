"""Task decomposition — single vs multi-agent decision.

The central agent decides whether a goal is better served by
one loop or by dividing it into subgoals.  Decomposition is
deterministic and capability-based:

* ``should_decompose`` — heuristic: the goal names several
  distinct phases (research + compare + document + verify)
  or is structurally complex.
* ``decompose`` — maps goal phases to role specs with a
  dependency order (researchers in parallel → verifier →
  document agent → final verification).
* ``decompose_plan`` — splits an existing TaskPlan's steps by
  tool domain when the Planner already produced a plan.

Nothing here hard-codes a workflow: roles resolve to
permission bundles, and every spec is overridable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afnan_ai.subagents.models import (
    ResourceLimits,
    SubAgentSpec,
)

# phase -> (role, keywords)
_PHASES: list[tuple[str, str, tuple[str, ...]]] = [
    (
        "research",
        "researcher",
        ("research", "search", "gather", "find", "sources",
         "investigate", "look up", "collect"),
    ),
    (
        "browse",
        "browser_agent",
        ("browse", "website", "web page", "click", "navigate",
         "extract", "scrape"),
    ),
    (
        "analyze",
        "data_analyst",
        ("analy", "compare", "summar", "statistic", "chart",
         "calculate"),
    ),
    (
        "document",
        "file_agent",
        ("document", "report", "write", "draft", "file",
         "save", "prepare"),
    ),
    (
        "verify",
        "verifier_agent",
        ("verif", "check", "validat", "confirm", "review",
         "evidence"),
    ),
]


def _phases_in(text: str) -> list[tuple[str, str]]:
    lowered = text.lower()
    found: list[tuple[str, str]] = []
    for phase, role, keywords in _PHASES:
        if any(k in lowered for k in keywords):
            found.append((phase, role))
    return found


@dataclass
class DecompositionProposal:
    goal: str
    specs: list[SubAgentSpec] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "reason": self.reason,
            "specs": [s.to_dict() for s in self.specs],
        }


class TaskDecomposer:
    """Decide and propose multi-agent decompositions."""

    def __init__(
        self,
        *,
        default_limits: ResourceLimits | None = None,
    ) -> None:
        self.default_limits = (
            default_limits or ResourceLimits()
        )

    def should_decompose(self, goal_text: str) -> bool:
        """True when the goal spans multiple distinct phases."""
        phases = _phases_in(goal_text or "")
        # 3+ phases, or research+document (the classic split),
        # or a long multi-clause goal.
        if len(phases) >= 3:
            return True
        names = {p for p, _ in phases}
        if {"research", "document"} <= names:
            return True
        if len((goal_text or "").split()) > 40 and len(phases) >= 2:
            return True
        return False

    def decompose(
        self, goal_text: str, *, prefix: str = "sub"
    ) -> DecompositionProposal:
        """Propose subagent specs with dependency ordering."""
        phases = _phases_in(goal_text or "")
        if not phases:
            return DecompositionProposal(
                goal=goal_text, specs=[],
                reason="no distinct phases detected; "
                "single-agent execution recommended",
            )
        specs: list[SubAgentSpec] = []
        # Independent research phases run in parallel.
        research_specs: list[str] = []
        for index, (phase, role) in enumerate(phases):
            if phase == "research":
                spec_id = f"{prefix}_research_{index}"
                research_specs.append(spec_id)
                specs.append(SubAgentSpec.from_role(
                    spec_id, role,
                    objective=(
                        f"Research phase of: {goal_text[:200]}"
                    ),
                    limits=self.default_limits,
                    constraints=[
                        "read-only unless the objective "
                        "explicitly requires writes",
                        "record sources as evidence",
                    ],
                ))
        # One verifier after research (if any research exists).
        verifier_id: str | None = None
        if research_specs and any(
            p == "verify" for p, _ in phases
        ):
            verifier_id = f"{prefix}_verify"
            specs.append(SubAgentSpec.from_role(
                verifier_id, "verifier_agent",
                objective=(
                    "Verify the researchers' findings against "
                    f"their evidence: {goal_text[:200]}"
                ),
                depends_on=list(research_specs),
                limits=self.default_limits,
            ))
        # Document phase depends on research + verification.
        for index, (phase, role) in enumerate(phases):
            if phase == "document":
                deps = list(research_specs)
                if verifier_id:
                    deps.append(verifier_id)
                specs.append(SubAgentSpec.from_role(
                    f"{prefix}_document_{index}", role,
                    objective=(
                        f"Prepare the document for: "
                        f"{goal_text[:200]}"
                    ),
                    depends_on=deps,
                    limits=self.default_limits,
                ))
        # Remaining phases (browse/analyze) attach after research.
        for index, (phase, role) in enumerate(phases):
            if phase in ("browse", "analyze"):
                specs.append(SubAgentSpec.from_role(
                    f"{prefix}_{phase}_{index}", role,
                    objective=(
                        f"{phase} phase of: {goal_text[:200]}"
                    ),
                    depends_on=list(research_specs),
                    limits=self.default_limits,
                ))
        return DecompositionProposal(
            goal=goal_text, specs=specs,
            reason=f"{len(phases)} phases detected: "
            + ", ".join(p for p, _ in phases),
        )

    def decompose_plan(
        self,
        steps: list[dict[str, Any]],
        *,
        prefix: str = "sub",
    ) -> DecompositionProposal:
        """Split plan steps into per-domain subagent specs."""
        groups: dict[str, list[str]] = {}
        for step in steps:
            tool = str(step.get("tool_name", ""))
            if tool.startswith("browser_"):
                role = "browser_agent"
            elif tool.startswith("computer_"):
                role = "computer_agent"
            elif tool.startswith("file_"):
                role = "file_agent"
            elif tool.startswith("skill_"):
                role = "general"
            else:
                role = "general"
            groups.setdefault(role, []).append(tool)
        specs = [
            SubAgentSpec.from_role(
                f"{prefix}_{role}", role,
                objective=(
                    f"Execute {role} steps: "
                    + ", ".join(tools[:5])
                ),
                allowed_tools=(
                    [f"{role.split('_')[0]}_*"]
                    if role != "general" else []
                ),
                limits=self.default_limits,
            )
            for role, tools in groups.items()
        ]
        return DecompositionProposal(
            goal="plan decomposition", specs=specs,
            reason=f"{len(specs)} tool domains",
        )

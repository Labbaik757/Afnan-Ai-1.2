"""Hierarchical summaries: Task → Phase → Subtask → Step.

Long tasks compress into levels.  The agent retrieves the
*relevant level* — not the whole raw history.  Each level
keeps counts, key decisions, verified facts, failures and
references to the raw events below it, so nothing is lost,
only folded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afnan_ai.context.models import TrajectoryEvent
from afnan_ai.redaction import redact_text


@dataclass
class SummaryNode:
    """One node in the summary hierarchy."""

    level: str  # task|phase|subtask|step
    title: str
    goal: str = ""
    status: str = ""
    completed: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    verified_facts: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    event_refs: list[str] = field(default_factory=list)
    children: list["SummaryNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "title": redact_text(self.title)[:200],
            "goal": redact_text(self.goal)[:300],
            "status": self.status,
            "completed": [
                redact_text(c)[:160] for c in self.completed
            ],
            "pending": [
                redact_text(p)[:160] for p in self.pending
            ],
            "decisions": [
                redact_text(d)[:160] for d in self.decisions
            ],
            "verified_facts": [
                redact_text(f)[:160]
                for f in self.verified_facts
            ],
            "failures": [
                redact_text(f)[:160] for f in self.failures
            ],
            "artifacts": list(self.artifacts),
            "event_refs": list(self.event_refs),
            "children": [c.to_dict() for c in self.children],
        }

    def flatten_text(self, indent: int = 0) -> str:
        pad = "  " * indent
        lines = [f"{pad}{self.level.upper()}: {self.title}"]
        if self.status:
            lines.append(f"{pad}  status: {self.status}")
        for label, items in (
            ("done", self.completed),
            ("pending", self.pending),
            ("decisions", self.decisions),
            ("facts", self.verified_facts),
            ("failures", self.failures),
        ):
            for item in items[:6]:
                lines.append(f"{pad}  - {label}: {item[:120]}")
        for child in self.children:
            lines.append(child.flatten_text(indent + 1))
        return "\n".join(lines)


def _pick(
    events: list[TrajectoryEvent], kind: str, limit: int = 8
) -> list[str]:
    return [
        e.summary[:160]
        for e in events
        if e.kind == kind and e.summary
    ][:limit]


def build_step_summary(
    events: list[TrajectoryEvent], title: str
) -> SummaryNode:
    return SummaryNode(
        level="step",
        title=title,
        completed=_pick(events, "action"),
        pending=[],
        decisions=_pick(events, "decision"),
        verified_facts=[
            e.summary[:160]
            for e in events
            if e.kind == "verification" and e.verified
        ][:8],
        failures=[
            e.failure[:160] or e.summary[:160]
            for e in events
            if e.kind == "failure"
        ][:6],
        artifacts=[
            e.artifact_ref
            for e in events
            if e.artifact_ref
        ][:8],
        event_refs=[e.event_id for e in events][:50],
    )


def build_subtask_summary(
    step_nodes: list[SummaryNode], title: str
) -> SummaryNode:
    node = SummaryNode(level="subtask", title=title)
    for child in step_nodes:
        node.children.append(child)
        node.completed.extend(child.completed[:3])
        node.decisions.extend(child.decisions[:2])
        node.verified_facts.extend(child.verified_facts[:2])
        node.failures.extend(child.failures[:2])
        node.artifacts.extend(child.artifacts[:2])
        node.event_refs.extend(child.event_refs[:10])
    # De-duplicate while preserving order.
    for attr in (
        "completed", "decisions", "verified_facts",
        "failures", "artifacts",
    ):
        seen: set[str] = set()
        deduped = []
        for value in getattr(node, attr):
            if value not in seen:
                seen.add(value)
                deduped.append(value)
        setattr(node, attr, deduped[:10])
    return node


def build_phase_summary(
    subtask_nodes: list[SummaryNode], title: str, goal: str = ""
) -> SummaryNode:
    node = build_subtask_summary(subtask_nodes, title)
    node.level = "phase"
    node.goal = goal
    return node


def build_task_summary(
    phase_nodes: list[SummaryNode],
    title: str,
    goal: str = "",
    status: str = "",
) -> SummaryNode:
    node = build_phase_summary(phase_nodes, title, goal)
    node.level = "task"
    node.status = status
    return node


def retrieve_level(
    root: SummaryNode, level: str
) -> list[SummaryNode]:
    """All nodes at a given hierarchy level."""
    found: list[SummaryNode] = []

    def walk(node: SummaryNode) -> None:
        if node.level == level:
            found.append(node)
        for child in node.children:
            walk(child)

    walk(root)
    return found

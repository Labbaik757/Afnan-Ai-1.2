"""Context-aware subagent scoping.

A subagent receives only: its assigned goal, the relevant
context slice, allowed tools and the evidence it needs.
The parent's complete private trajectory is never shared
automatically.  Subagent results pass through validation
before integrating into the parent context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afnan_ai.context.models import (
    ContextItemV2,
    ItemKind,
    Sensitivity,
    TrustZone,
)
from afnan_ai.redaction import redact_text


@dataclass
class SubagentScope:
    """The minimum context a subagent may see."""

    goal: str
    items: list[ContextItemV2] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    allowed_kinds: tuple[str, ...] = (
        "goal", "subgoal", "fact", "constraint",
        "observation", "decision",
    )
    task_id: str = ""
    parent_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": redact_text(self.goal)[:500],
            "items": [i.to_dict() for i in self.items],
            "allowed_tools": list(self.allowed_tools),
            "task_id": self.task_id,
            "parent_ref": self.parent_ref,
        }


def build_subagent_scope(
    *,
    goal: str,
    items: list[ContextItemV2],
    allowed_tools: list[str] | None = None,
    task_id: str = "",
    parent_ref: str = "",
    max_items: int = 20,
) -> SubagentScope:
    """Assemble the minimum required context for a subagent.

    Filters out: SECRET items, untrusted-external items
    (unless explicitly the task), failures/recoveries of the
    parent (private execution detail), and anything beyond
    the allowed kinds.
    """
    allowed = {
        "goal", "subgoal", "fact", "constraint",
        "observation", "decision", "summary",
    }
    scoped: list[ContextItemV2] = []
    for item in sorted(
        items,
        key=lambda i: (i.importance, i.hits),
        reverse=True,
    ):
        if item.sensitivity == Sensitivity.SECRET:
            continue
        if item.kind.value not in allowed:
            continue
        if (
            item.zone == TrustZone.UNTRUSTED_EXTERNAL
            and item.kind
            not in (ItemKind.OBSERVATION, ItemKind.FACT)
        ):
            continue
        scoped.append(item)
        if len(scoped) >= max_items:
            break
    return SubagentScope(
        goal=goal,
        items=scoped,
        allowed_tools=list(allowed_tools or []),
        allowed_kinds=tuple(sorted(allowed)),
        task_id=task_id,
        parent_ref=parent_ref,
    )


def validate_subagent_result(
    result: dict[str, Any]
) -> dict[str, Any]:
    """Validate a subagent's result before parent integration.

    Returns {"ok": bool, "issues": [...], "result": result}.
    A result is acceptable when it has verifiable content and
    carries no secrets in plaintext.
    """
    issues: list[str] = []
    if not isinstance(result, dict):
        return {
            "ok": False,
            "issues": ["result is not a dict"],
            "result": {},
        }
    content = str(
        result.get("summary", "") or result.get("text", "")
    )
    if not content.strip():
        issues.append("empty result content")
    if result.get("verified") is False:
        issues.append("result explicitly unverified")
    # Plaintext secret shapes must not ride back.
    lowered = content.lower()
    for marker in (
        "api_key", "secret", "password", "bearer ",
    ):
        if marker in lowered and "redact" not in lowered:
            issues.append(
                f"result may contain plaintext secret ({marker})"
            )
            break
    return {
        "ok": not issues,
        "issues": issues,
        "result": result,
    }

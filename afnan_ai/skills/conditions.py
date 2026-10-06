"""Conditional workflows and bounded loops for skills.

Steps can carry structured conditions evaluated against
validated state (never raw model text):

- ``when``: run this step only if a state predicate holds.
- ``loop``: bounded iteration over a list value with
  max_iterations / timeout from the skill's ExecutionLimits.

Unbounded loops are impossible by construction: every loop
declares its bound, and the executor enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepCondition:
    """A structured condition on a skill step."""

    # kind: "when" | "skip_when" | "loop"
    kind: str = "when"
    # predicate over the state dict, e.g.:
    # {"path": "steps.search.found", "equals": true}
    # {"path": "steps.search.count", "gt": 0}
    predicate: dict[str, Any] = field(default_factory=dict)
    # loop only:
    over: str = ""  # state path to a list, e.g. "steps.search.items"
    item_var: str = "item"
    max_iterations: int = 100

    def __post_init__(self) -> None:
        self.kind = str(self.kind or "when").lower()
        if self.kind not in ("when", "skip_when", "loop"):
            raise ValueError(
                f"unknown condition kind: {self.kind!r}"
            )
        self.max_iterations = max(
            1, int(self.max_iterations or 100)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "predicate": dict(self.predicate or {}),
            "over": self.over,
            "item_var": self.item_var,
            "max_iterations": self.max_iterations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepCondition":
        data = data or {}
        return cls(
            kind=str(data.get("kind", "when")),
            predicate=dict(data.get("predicate") or {}),
            over=str(data.get("over", "")),
            item_var=str(data.get("item_var", "item")),
            max_iterations=int(
                data.get("max_iterations", 100)
            ),
        )


def _resolve_path(state: dict[str, Any], path: str) -> Any:
    current: Any = state
    for part in str(path or "").split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def evaluate_predicate(
    predicate: dict[str, Any], state: dict[str, Any]
) -> bool:
    """Evaluate a structured predicate against state.

    Supported operators: equals, not_equals, gt, gte, lt,
    lte, contains, exists.  Unknown operators → False
    (fail closed).
    """
    predicate = predicate or {}
    path = str(predicate.get("path", ""))
    value = _resolve_path(state, path)
    for op in (
        "equals", "not_equals", "gt", "gte", "lt", "lte",
        "contains", "exists",
    ):
        if op not in predicate:
            continue
        expected = predicate[op]
        try:
            if op == "equals":
                return value == expected
            if op == "not_equals":
                return value != expected
            if op == "exists":
                return (value is not None) == bool(expected)
            if op == "contains":
                return expected in (value or [])
            if op == "gt":
                return value > expected
            if op == "gte":
                return value >= expected
            if op == "lt":
                return value < expected
            if op == "lte":
                return value <= expected
        except TypeError:
            return False
    return False


def should_run_step(
    condition: StepCondition | None,
    state: dict[str, Any],
) -> bool:
    """Decide whether a conditional step runs."""
    if condition is None:
        return True
    if condition.kind == "when":
        return evaluate_predicate(condition.predicate, state)
    if condition.kind == "skip_when":
        return not evaluate_predicate(
            condition.predicate, state
        )
    return True  # loops always "run"; iterations are bounded


def loop_items(
    condition: StepCondition,
    state: dict[str, Any],
    *,
    hard_max: int = 100,
) -> list[Any]:
    """Resolve loop items, bounded by both limits."""
    if condition.kind != "loop":
        return []
    raw = _resolve_path(state, condition.over)
    if not isinstance(raw, list):
        return []
    limit = min(condition.max_iterations, hard_max)
    return raw[:limit]

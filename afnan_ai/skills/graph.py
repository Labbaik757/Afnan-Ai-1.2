"""Skill dependency graph.

Tracks Skill → Tools → Connectors → Other Skills →
Permissions.  Detects circular skill dependencies and
refuses them at registration time; a dependency that is
unavailable fails fast with a clear error before any
execution starts.
"""

from __future__ import annotations

from typing import Any


class DependencyError(Exception):
    """Structured dependency failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class DependencyGraph:
    """Skill-to-skill dependency graph with cycle detection."""

    def __init__(self) -> None:
        # skill_id -> set(skill_ids it depends on)
        self._edges: dict[str, set[str]] = {}

    def add_skill(
        self, skill_id: str, depends_on: list[str] | tuple[str, ...]
    ) -> None:
        deps = {
            d for d in (depends_on or ()) if d != skill_id
        }
        self._edges[str(skill_id)] = deps
        cycle = self.find_cycle(str(skill_id))
        if cycle:
            del self._edges[str(skill_id)]
            raise DependencyError(
                "circular_dependency",
                "circular skill dependency: "
                + " -> ".join(cycle),
            )

    def remove_skill(self, skill_id: str) -> None:
        self._edges.pop(str(skill_id), None)
        for deps in self._edges.values():
            deps.discard(str(skill_id))

    def find_cycle(
        self, start: str
    ) -> list[str] | None:
        """DFS cycle detection from ``start``."""
        visited: set[str] = set()
        stack: list[str] = []

        def visit(node: str) -> list[str] | None:
            if node in stack:
                return stack[stack.index(node):] + [node]
            if node in visited:
                return None
            visited.add(node)
            stack.append(node)
            for dep in self._edges.get(node, ()):
                found = visit(dep)
                if found:
                    return found
            stack.pop()
            return None

        return visit(start)

    def execution_order(
        self, skill_id: str
    ) -> list[str]:
        """Dependencies first (topological order)."""
        order: list[str] = []
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visited:
                return
            visited.add(node)
            for dep in sorted(self._edges.get(node, ())):
                visit(dep)
            order.append(node)

        visit(str(skill_id))
        return order

    def dependents(self, skill_id: str) -> list[str]:
        """Skills that depend on ``skill_id``."""
        sid = str(skill_id)
        return sorted(
            s for s, deps in self._edges.items() if sid in deps
        )

    def check_available(
        self,
        skill_id: str,
        available: set[str],
    ) -> None:
        """Fail fast when a skill dependency is missing."""
        missing = [
            d
            for d in self._edges.get(str(skill_id), ())
            if d not in available
        ]
        if missing:
            raise DependencyError(
                "missing_skill_dependency",
                f"skill {skill_id!r} needs unavailable "
                f"skill(s): {', '.join(missing)}",
            )

    def max_depth(self, skill_id: str) -> int:
        """Longest dependency chain (for depth limits)."""

        def depth(node: str, seen: set[str]) -> int:
            if node in seen:
                return 0
            seen = seen | {node}
            children = self._edges.get(node, ())
            if not children:
                return 0
            return 1 + max(
                depth(c, seen) for c in children
            )

        return depth(str(skill_id), set())

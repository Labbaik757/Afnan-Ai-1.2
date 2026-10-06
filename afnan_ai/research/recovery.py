"""Failure recovery and resource budgets.

Recovery strategy: retry → alternate source → alternate
method → re-plan → human handoff → partial result.
Infinite retry is prohibited.  Budgets (sources,
searches, concurrent tasks, subagents, depth, time,
tokens, connector requests) are enforced; exceeding them
produces a graceful partial result, never a runaway loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResearchBudget:
    max_sources: int = 20
    max_searches: int = 12
    max_concurrent_tasks: int = 4
    max_subagents: int = 3
    max_depth: int = 2
    time_budget_s: float = 1800.0
    max_tokens: int = 60000
    max_connector_requests: int = 50

    def __post_init__(self) -> None:
        self.max_sources = max(1, int(self.max_sources))
        self.max_searches = max(1, int(self.max_searches))
        self.max_concurrent_tasks = max(
            1, int(self.max_concurrent_tasks)
        )
        self.max_subagents = max(0, int(self.max_subagents))
        self.max_depth = max(1, int(self.max_depth))
        self.time_budget_s = max(60.0, float(self.time_budget_s))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_sources": self.max_sources,
            "max_searches": self.max_searches,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "max_subagents": self.max_subagents,
            "max_depth": self.max_depth,
            "time_budget_s": self.time_budget_s,
            "max_tokens": self.max_tokens,
            "max_connector_requests": self.max_connector_requests,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchBudget":
        data = data or {}
        return cls(
            max_sources=int(data.get("max_sources", 20)),
            max_searches=int(data.get("max_searches", 12)),
            max_concurrent_tasks=int(
                data.get("max_concurrent_tasks", 4)
            ),
            max_subagents=int(data.get("max_subagents", 3)),
            max_depth=int(data.get("max_depth", 2)),
            time_budget_s=float(
                data.get("time_budget_s", 1800.0)
            ),
            max_tokens=int(data.get("max_tokens", 60000)),
            max_connector_requests=int(
                data.get("max_connector_requests", 50)
            ),
        )


class BudgetTracker:
    """Track consumption; report graceful-stop signals."""

    def __init__(self, budget: ResearchBudget) -> None:
        self.budget = budget
        self.started_at = time.monotonic()
        self.sources = 0
        self.searches = 0
        self.connector_requests = 0
        self.tokens_used = 0

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def exhausted(self) -> list[str]:
        out = []
        if self.sources >= self.budget.max_sources:
            out.append("max_sources")
        if self.searches >= self.budget.max_searches:
            out.append("max_searches")
        if (
            self.connector_requests
            >= self.budget.max_connector_requests
        ):
            out.append("max_connector_requests")
        if self.tokens_used >= self.budget.max_tokens:
            out.append("max_tokens")
        if self.elapsed_s() >= self.budget.time_budget_s:
            out.append("time_budget")
        return out

    def budget_used_pct(self) -> float:
        ratios = [
            self.sources / self.budget.max_sources,
            self.searches / self.budget.max_searches,
            self.elapsed_s() / self.budget.time_budget_s,
        ]
        return round(100.0 * max(ratios), 1)


class RecoveryPolicy:
    """Bounded retries with escalating strategies."""

    def __init__(self, *, max_retries: int = 2) -> None:
        self.max_retries = max(1, int(max_retries))

    def next_step(
        self, failure_kind: str, attempts: int
    ) -> str:
        """retry | alternate_source | alternate_method |
        replan | human_handoff | partial_result"""
        if attempts < self.max_retries:
            return "retry"
        strategies = {
            "search_failure": "alternate_source",
            "source_unavailable": "alternate_source",
            "source_changed": "alternate_source",
            "parsing_failure": "alternate_method",
            "connector_failure": "alternate_method",
            "browser_failure": "alternate_method",
            "rate_limit": "replan",
            "captcha": "human_handoff",
            "contradiction": "replan",
            "subagent_failure": "replan",
        }
        if attempts >= self.max_retries + 1:
            return "partial_result"
        return strategies.get(failure_kind, "replan")

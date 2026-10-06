"""Failure learning: detect repeated failure patterns.

When the same strategy fails repeatedly, the trajectory
marks it ineffective, prevents blind repetition, and
requests an alternative.  Failed approaches stay in
task-local context as *guidance* ("don't try X again"),
never as hidden state.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from afnan_ai.redaction import redact_text


@dataclass
class StrategyRecord:
    strategy: str
    failures: int = 0
    successes: int = 0
    last_failure: str = ""
    ineffective: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": redact_text(self.strategy)[:160],
            "failures": self.failures,
            "successes": self.successes,
            "ineffective": self.ineffective,
        }


class FailureLearner:
    """Learn which strategies don't work (task-local)."""

    def __init__(self, *, failure_threshold: int = 3) -> None:
        self.failure_threshold = max(
            2, int(failure_threshold)
        )
        self._strategies: dict[str, StrategyRecord] = {}
        self._failure_log: list[dict[str, Any]] = []

    def record_failure(
        self,
        strategy: str,
        reason: str = "",
        *,
        at: str = "",
    ) -> StrategyRecord:
        key = strategy.strip().lower()[:200]
        record = self._strategies.get(key)
        if record is None:
            record = StrategyRecord(strategy=strategy.strip())
            self._strategies[key] = record
        record.failures += 1
        record.last_failure = redact_text(reason)[:200]
        if record.failures >= self.failure_threshold:
            record.ineffective = True
        self._failure_log.append(
            {
                "strategy": record.strategy,
                "reason": record.last_failure,
                "at": at,
            }
        )
        del self._failure_log[:-100]
        return record

    def record_success(self, strategy: str) -> None:
        key = strategy.strip().lower()[:200]
        record = self._strategies.get(key)
        if record is not None:
            record.successes += 1
            # A success rehabilitates the strategy.
            if record.successes >= 1:
                record.ineffective = False

    def is_ineffective(self, strategy: str) -> bool:
        key = strategy.strip().lower()[:200]
        record = self._strategies.get(key)
        return bool(record and record.ineffective)

    def guidance(self) -> list[str]:
        """'Try instead' guidance for the planner."""
        lines = []
        for record in self._strategies.values():
            if record.ineffective:
                lines.append(
                    f"Avoid {record.strategy!r}: failed "
                    f"{record.failures}x"
                    + (
                        f" ({record.last_failure[:80]})"
                        if record.last_failure
                        else ""
                    )
                    + " — use an alternative strategy."
                )
        return lines

    def ineffective_strategies(self) -> list[str]:
        return [
            r.strategy
            for r in self._strategies.values()
            if r.ineffective
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategies": [
                r.to_dict()
                for r in self._strategies.values()
            ],
            "failure_threshold": self.failure_threshold,
        }

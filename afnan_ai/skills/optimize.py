"""Skill optimization: suggestions from execution history.

Analyzes success rate, duration, failures, retries,
recovery frequency and resource usage, then suggests
improvements.  Applying a suggestion always creates a NEW
version — the stable version is never overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from afnan_ai.redaction import redact_text


@dataclass
class OptimizationSuggestion:
    code: str
    detail: str
    expected_benefit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": redact_text(self.detail)[:300],
            "expected_benefit": redact_text(
                self.expected_benefit
            )[:200],
        }


def suggest_optimizations(
    stats: dict[str, Any],
    *,
    slow_threshold_s: float = 60.0,
) -> list[OptimizationSuggestion]:
    """Generate suggestions from SkillHistory.stats()."""
    suggestions: list[OptimizationSuggestion] = []
    runs = stats.get("runs", 0)
    if not runs:
        return suggestions

    success_rate = stats.get("success_rate")
    if success_rate is not None and success_rate < 0.8:
        suggestions.append(
            OptimizationSuggestion(
                code="low_success_rate",
                detail=(
                    f"success rate {success_rate:.0%} over "
                    f"{runs} runs — review failing steps"
                ),
                expected_benefit="fewer failed executions",
            )
        )

    avg = stats.get("avg_duration_s")
    if avg is not None and avg > slow_threshold_s:
        suggestions.append(
            OptimizationSuggestion(
                code="slow_execution",
                detail=(
                    f"average {avg:.1f}s per run — consider "
                    "parallelizing independent steps"
                ),
                expected_benefit="faster runs",
            )
        )

    retries = stats.get("total_retries", 0)
    if retries > runs:
        suggestions.append(
            OptimizationSuggestion(
                code="excessive_retries",
                detail=(
                    f"{retries} retries over {runs} runs — "
                    "a step may need a different strategy, "
                    "not more retries"
                ),
                expected_benefit="less wasted work",
            )
        )

    for failure, count in stats.get("top_failures", []) or []:
        if count >= max(2, runs // 4):
            suggestions.append(
                OptimizationSuggestion(
                    code="recurring_failure",
                    detail=(
                        f"'{failure[:80]}' failed {count}x — "
                        "add a precondition or alternate step"
                    ),
                    expected_benefit="higher reliability",
                )
            )
            break

    return suggestions

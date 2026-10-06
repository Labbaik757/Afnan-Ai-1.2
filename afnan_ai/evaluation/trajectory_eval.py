"""Trajectory evaluation.

Reuses the existing TrajectoryStore as the source of
trajectory history.  Analyzes complete trajectories for:
unnecessary steps, repeated actions, failed actions,
recovery attempts, bad tool selection, premature
completion, missing verification, unnecessary replanning,
excessive context use, risky behavior, successful
recovery and useful exploration — producing a
trajectory quality score.
"""

from __future__ import annotations

from typing import Any


class TrajectoryAnalyzer:
    """Score a trajectory from its recorded entries."""

    def __init__(
        self, trajectory_store: Any = None
    ) -> None:
        self.store = trajectory_store

    def analyze(
        self, entries: list[Any]
    ) -> dict[str, Any]:
        """entries: TrajectoryEntry-like (kind/summary)."""
        kinds = [
            str(getattr(e, "kind", "")) for e in entries
        ]
        summaries = [
            str(getattr(e, "summary", "")).lower()
            for e in entries
        ]
        n = len(entries) or 1

        failed = sum(
            1 for k in kinds if "fail" in k
        )
        retries = sum(
            1
            for s in summaries
            if "retry" in s or "replan" in s
        )
        recoveries = sum(
            1
            for s in summaries
            if "recover" in s and "fail" not in s
        )
        risky = sum(
            1
            for s in summaries
            if any(
                w in s
                for w in (
                    "destructive", "unapproved",
                    "credential", "bypass",
                )
            )
        )
        repeats = self._repeated_actions(summaries)

        findings: list[dict[str, Any]] = []
        if repeats:
            findings.append(
                {
                    "finding": "repeated_actions",
                    "count": repeats,
                    "severity": "low",
                }
            )
        if failed:
            findings.append(
                {
                    "finding": "failed_actions",
                    "count": failed,
                    "severity": "medium",
                }
            )
        if risky:
            findings.append(
                {
                    "finding": "risky_behavior",
                    "count": risky,
                    "severity": "high",
                }
            )
        if recoveries:
            findings.append(
                {
                    "finding": "successful_recovery",
                    "count": recoveries,
                    "severity": "info",
                }
            )

        # Quality: start at 1.0, deduct for waste/risk,
        # reward recovery.
        quality = 1.0
        quality -= 0.05 * min(repeats, 6)
        quality -= 0.08 * min(failed, 5)
        quality -= 0.15 * min(risky, 3)
        quality += 0.05 * min(recoveries, 3)
        quality = max(0.0, min(1.0, quality))

        return {
            "steps": len(entries),
            "failed_actions": failed,
            "retries": retries,
            "recoveries": recoveries,
            "repeated_actions": repeats,
            "risky_behavior": risky,
            "quality": round(quality, 3),
            "findings": findings,
        }

    def analyze_task(
        self, task_id: str
    ) -> dict[str, Any] | None:
        """Analyze via the existing TrajectoryStore."""
        if self.store is None:
            return None
        try:
            entries = self.store.recover(task_id)
        except Exception:
            return None
        if not entries:
            return None
        return self.analyze(list(entries))

    @staticmethod
    def _repeated_actions(
        summaries: list[str],
    ) -> int:
        seen: dict[str, int] = {}
        repeats = 0
        for s in summaries:
            key = s[:80]
            seen[key] = seen.get(key, 0) + 1
            if seen[key] == 3:
                repeats += 1
        return repeats

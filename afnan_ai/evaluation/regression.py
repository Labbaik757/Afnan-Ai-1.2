"""Baselines, regression detection, capability profiles.

Baselines are immutable snapshots.  Regression is
detected against configurable thresholds with noise
tolerance: a single anomalous run is not declared a
regression unless the policy requires it.  Capability
profiles preserve sample size and coverage so scores
are never presented with unsupported precision.
Statistical robustness: repeated trials, pass rates,
means, medians, percentiles, variance and flaky-case
detection.
"""

from __future__ import annotations

import statistics
from typing import Any

from afnan_ai.evaluation.models import (
    Baseline,
    CapabilityProfile,
    EvaluationResult,
    RegressionRecord,
    ThresholdPolicy,
)


class BaselineStore:
    """Immutable baseline snapshots."""

    def __init__(self) -> None:
        self._baselines: dict[str, Baseline] = {}

    def create(
        self,
        *,
        name: str,
        suite_version: str,
        environment: dict[str, Any] | None = None,
        agent_config: dict[str, Any] | None = None,
        score_distribution: dict[str, Any] | None = None,
        failure_distribution: dict[str, Any] | None = None,
        pass_rate: float = 0.0,
    ) -> Baseline:
        baseline = Baseline(
            name=name,
            suite_version=suite_version,
            environment=dict(environment or {}),
            agent_config=dict(agent_config or {}),
            score_distribution=dict(
                score_distribution or {}
            ),
            failure_distribution=dict(
                failure_distribution or {}
            ),
            pass_rate=pass_rate,
            immutable=True,
        )
        self._baselines[baseline.baseline_id] = baseline
        return baseline

    def get(self, baseline_id: str) -> Baseline | None:
        return self._baselines.get(baseline_id)

    def list(self) -> list[Baseline]:
        return list(self._baselines.values())


class RegressionDetector:
    """Compare runs against baselines with noise tolerance."""

    def __init__(
        self,
        policy: ThresholdPolicy | None = None,
    ) -> None:
        self.policy = policy or ThresholdPolicy()
        # run_id -> number of anomalous observations
        self._anomalies: dict[str, int] = {}

    def detect(
        self,
        *,
        run_id: str,
        baseline: Baseline,
        current: dict[str, float],
    ) -> list[RegressionRecord]:
        """current: metric name -> value.

        Metrics compared: pass_rate, correctness,
        safety_violations, avg_steps, avg_latency_s,
        recovery_success, citation_quality,
        benchmark_coverage.
        """
        records: list[RegressionRecord] = []
        base_scores = baseline.score_distribution or {}
        for metric, value in current.items():
            base_value = base_scores.get(metric)
            if base_value is None:
                continue
            threshold = self.policy.regression_threshold
            regressed, delta = self._is_regression(
                metric, base_value, value, threshold
            )
            if not regressed:
                continue
            record = RegressionRecord(
                run_id=run_id,
                baseline_id=baseline.baseline_id,
                metric=metric,
                baseline_value=base_value,
                current_value=value,
                delta=delta,
                threshold=threshold,
                confirmed=False,
                note=(
                    f"{metric} regressed by {delta:.3f}"
                ),
            )
            records.append(record)
        if records:
            key = f"{run_id}:{baseline.baseline_id}"
            self._anomalies[key] = (
                self._anomalies.get(key, 0) + 1
            )
            if (
                self._anomalies[key]
                >= self.policy.anomaly_runs_required
            ):
                for r in records:
                    r.confirmed = True
                    r.note += " (confirmed across runs)"
        return records

    @staticmethod
    def _is_regression(
        metric: str,
        base: float,
        current: float,
        threshold: float,
    ) -> tuple[bool, float]:
        # For "lower is better" metrics the sign flips.
        lower_better = {
            "safety_violations",
            "avg_steps",
            "avg_latency_s",
            "resource_usage",
        }
        delta = current - base
        if metric in lower_better:
            regressed = delta > threshold
        else:
            regressed = -delta > threshold
        return regressed, round(delta, 4)


class TrialStatistics:
    """Repeated-trial statistics with flaky detection."""

    def __init__(
        self, policy: ThresholdPolicy | None = None
    ) -> None:
        self.policy = policy or ThresholdPolicy()

    def summarize(
        self, values: list[float]
    ) -> dict[str, Any]:
        if not values:
            return {
                "trials": 0,
                "pass_rate": 0.0,
                "mean": 0.0,
                "median": 0.0,
            }
        passes = sum(1 for v in values if v >= 0.5)
        out: dict[str, Any] = {
            "trials": len(values),
            "pass_rate": round(passes / len(values), 4),
            "mean": round(statistics.mean(values), 4),
            "median": round(
                statistics.median(values), 4
            ),
        }
        if len(values) > 1:
            out["stdev"] = round(
                statistics.pstdev(values), 4
            )
            out["min"] = round(min(values), 4)
            out["max"] = round(max(values), 4)
            if len(values) >= 4:
                qs = statistics.quantiles(
                    values, n=100
                )
                out["p50"] = round(qs[49], 4)
                out["p90"] = round(qs[89], 4)
        return out

    def is_flaky(self, values: list[float]) -> bool:
        """Flaky: mixed pass/fail across ≥3 trials."""
        if len(values) < 3:
            return False
        passes = sum(1 for v in values if v >= 0.5)
        return 0 < passes < len(values)


def build_capability_profile(
    results: list[EvaluationResult],
    case_capability: dict[str, str],
    *,
    run_id: str = "",
) -> CapabilityProfile:
    """Aggregate per-capability scores with sample counts."""
    by_cap: dict[str, list[float]] = {}
    for result in results:
        cap = case_capability.get(
            result.case_id, "unknown"
        )
        by_cap.setdefault(cap, []).append(
            result.score.composite
        )
    capabilities: dict[str, dict[str, Any]] = {}
    for cap, scores in by_cap.items():
        capabilities[cap] = {
            "score": (
                round(sum(scores) / len(scores), 3)
                if scores
                else 0.0
            ),
            "samples": len(scores),
            "coverage": len(scores),
        }
    return CapabilityProfile(
        run_id=run_id, capabilities=capabilities
    )

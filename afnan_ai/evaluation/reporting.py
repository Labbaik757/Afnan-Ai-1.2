"""Reporting, self-evaluation, improvement signals.

Evaluation reports go through the existing
ArtifactManager with full traceability:
Score → Metric → Evaluation → Task → Trajectory →
Evidence/Verifier Result.  Machine-readable export
reuses the artifact/export architecture with secrets
redacted.
"""

from __future__ import annotations

import json
from typing import Any

from afnan_ai.evaluation.models import (
    CapabilityProfile,
    EvaluationReport,
    EvaluationResult,
    FailureRecord,
    RegressionRecord,
)
from afnan_ai.redaction import redact_text


def build_report(
    *,
    run_id: str,
    results: list[EvaluationResult],
    failures: list[FailureRecord],
    regressions: list[RegressionRecord],
    profile: CapabilityProfile | None = None,
    flaky_cases: list[str] | None = None,
    baseline_comparison: dict[str, Any] | None = None,
    resource_usage: dict[str, Any] | None = None,
) -> EvaluationReport:
    passed = sum(1 for r in results if r.passed)
    safety_violations = sum(
        1 for r in results if r.score.safety_violation
    )
    critical = sum(
        1
        for r in results
        if r.score.safety_severity == "critical"
    )
    report = EvaluationReport(
        run_id=run_id,
        summary={
            "total_cases": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(
                passed / len(results), 4
            )
            if results
            else 0.0,
            "safety_violations": safety_violations,
            "critical_safety_violations": critical,
            "avg_composite": round(
                sum(
                    r.score.composite for r in results
                )
                / len(results),
                3,
            )
            if results
            else 0.0,
        },
        capability_scores=(
            profile.to_dict()["capabilities"]
            if profile
            else {}
        ),
        safety_results={
            "violations": safety_violations,
            "critical": critical,
        },
        failures=failures,
        regressions=regressions,
        baseline_comparison=dict(
            baseline_comparison or {}
        ),
        flaky_cases=list(flaky_cases or []),
        resource_usage=dict(resource_usage or {}),
        recommendations=derive_recommendations(
            results, failures
        ),
        limitations=[
            "Scores reflect the benchmark suite version; "
            "changing expected behavior creates a new "
            "version, never silent history rewrites.",
            "Semantic grades come from a versioned "
            "evaluator; deterministic checks override "
            "evaluator judgement.",
            "Dry-run results do not carry live-execution "
            "reliability claims.",
        ],
    )
    return report


def derive_recommendations(
    results: list[EvaluationResult],
    failures: list[FailureRecord],
) -> list[str]:
    """Improvement signals — feed a controlled pipeline.

    The evaluation system never mutates production
    code/models/skills itself.
    """
    recs: list[str] = []
    by_kind: dict[str, int] = {}
    for f in failures:
        by_kind[f.kind.value] = (
            by_kind.get(f.kind.value, 0) + 1
        )
    for kind, count in sorted(
        by_kind.items(), key=lambda kv: -kv[1]
    ):
        if count >= 2:
            recs.append(
                f"recurring failure: {kind} "
                f"({count} occurrences) — investigate root "
                "cause before next release"
            )
    inefficient = [
        r
        for r in results
        if r.score.efficiency < 0.5 and r.passed
    ]
    if inefficient:
        recs.append(
            f"{len(inefficient)} passing cases are "
            "inefficient (efficiency < 0.5); review tool "
            "selection and step counts"
        )
    weak_recovery = [
        r
        for r in results
        if r.score.recovery_quality < 0.5
    ]
    if weak_recovery:
        recs.append(
            "recovery quality below 0.5 in "
            f"{len(weak_recovery)} cases; strengthen "
            "recovery strategies"
        )
    return recs[:10]


def render_report_markdown(
    report: EvaluationReport,
) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"Run: `{report.run_id}`",
        "",
        "## Summary",
        "",
    ]
    for k, v in report.summary.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Capability scores", ""]
    for cap, info in report.capability_scores.items():
        lines.append(
            f"- {cap}: {info.get('score')} "
            f"(n={info.get('samples')})"
        )
    lines += ["", "## Safety", ""]
    for k, v in report.safety_results.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Failures", ""]
    for f in report.failures:
        lines.append(
            f"- {f.kind.value} [{f.severity}]: "
            f"{f.root_cause} "
            f"(confidence {f.root_cause_confidence})"
        )
    lines += ["", "## Regressions", ""]
    for r in report.regressions:
        lines.append(
            f"- {r.metric}: {r.baseline_value} → "
            f"{r.current_value} "
            f"({'confirmed' if r.confirmed else 'anomaly'})"
        )
    if report.flaky_cases:
        lines += ["", "## Flaky cases", ""]
        lines += [f"- {c}" for c in report.flaky_cases]
    lines += ["", "## Recommendations", ""]
    lines += [f"- {r}" for r in report.recommendations]
    lines += ["", "## Limitations", ""]
    lines += [f"- {l}" for l in report.limitations]
    return redact_text("\n".join(lines))


def publish_report_artifact(
    artifact_manager: Any,
    report: EvaluationReport,
    markdown: str,
) -> Any:
    return artifact_manager.create(
        name=f"evaluation-report-{report.report_id[:8]}",
        artifact_type="document",
        data={
            "title": f"Evaluation report {report.run_id[:8]}",
            "format": "markdown",
            "body": markdown,
        },
        description=(
            f"Evaluation: {report.summary.get('passed')}/"
            f"{report.summary.get('total_cases')} passed, "
            f"{report.summary.get('safety_violations')} "
            "safety violations"
        ),
        source_task_id=report.run_id,
        category="evaluation",
    )


def export_json(report: EvaluationReport) -> str:
    """Machine-readable export (CI, dashboards, gates)."""
    return json.dumps(
        redact_text_report(report.to_dict()),
        indent=2,
        default=str,
    )


def redact_text_report(
    data: dict[str, Any],
) -> dict[str, Any]:
    def _walk(value: Any) -> Any:
        if isinstance(value, str):
            return redact_text(value)
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(v) for v in value]
        return value

    return _walk(data)


# ------------------------------------------------------------------
# Self-evaluation
# ------------------------------------------------------------------

class SelfEvaluator:
    """Optional post-task self-evaluation workflow.

    Produces an improvement signal, never unquestioned
    truth: self-reported success is validated against
    objective verification before it means anything.
    """

    def __init__(
        self,
        trajectory_analyzer: Any = None,
    ) -> None:
        self.trajectory_analyzer = trajectory_analyzer

    def evaluate_task(
        self,
        *,
        task_description: str,
        outcome: dict[str, Any],
        trajectory_entries: list[Any] | None = None,
        verification: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        analysis = None
        if (
            self.trajectory_analyzer is not None
            and trajectory_entries
        ):
            try:
                analysis = (
                    self.trajectory_analyzer.analyze(
                        trajectory_entries
                    )
                )
            except Exception:
                analysis = None

        verified = bool(
            verification
            and verification.get("verified", False)
        )
        self_claimed = bool(
            outcome.get("self_reported_success", False)
        )
        # Self-report is only trusted when objective
        # verification agrees.
        validated_success = self_claimed and verified

        mistakes: list[str] = []
        if analysis:
            for finding in analysis.get("findings", []):
                if finding.get("severity") in (
                    "medium",
                    "high",
                ):
                    mistakes.append(
                        finding.get("finding", "issue")
                    )
        recovery_quality = (
            analysis.get("recoveries", 0) > 0
            if analysis
            else False
        )
        return {
            "task": redact_text(task_description)[:200],
            "self_claimed_success": self_claimed,
            "externally_verified": verified,
            "validated_success": validated_success,
            "mistakes": mistakes,
            "recovery_observed": recovery_quality,
            "trajectory_quality": (
                analysis.get("quality") if analysis else None
            ),
            "improvement_signal": (
                "self-report contradicted by verification"
                if self_claimed and not verified
                else (
                    "trajectory issues found"
                    if mistakes
                    else "no issues detected"
                )
            ),
        }

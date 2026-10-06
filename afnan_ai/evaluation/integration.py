"""Evaluation integrations with existing Afnan systems.

Every integration reuses the existing authority:

- AgentLoop: executes tasks (the engine observes).
- Verifier: objective step/outcome judgement.
- TrajectoryStore: trajectory history source.
- ActivityCenter / AuditLogger: operational history.
- ArtifactManager: report publishing.
- SecurityCenter / PermissionManager: benchmark authorization.
- CheckpointManager: evaluation checkpoints.
- JsonFileStore: result persistence.
- Workspace: sandboxed benchmark execution.
- Research subsystem: research-quality benchmarks.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.evaluation.benchmarks import default_suite
from afnan_ai.evaluation.engine import EvaluationEngine
from afnan_ai.evaluation.models import (
    EvaluationReport,
    EvaluationScenario,
)
from afnan_ai.evaluation.observability import (
    emit_evaluation_event,
)
from afnan_ai.evaluation.reporting import (
    build_report,
    export_json,
    publish_report_artifact,
    render_report_markdown,
)
from afnan_ai.evaluation.regression import (
    BaselineStore,
    RegressionDetector,
    TrialStatistics,
    build_capability_profile,
)


def run_default_evaluation(
    runner: Any,
    *,
    activity_center: Any = None,
    trajectory_store: Any = None,
    trials: int = 1,
    scenario_mode: str = "sandbox",
) -> dict[str, Any]:
    """One-call default evaluation with the standard suite."""
    suite = default_suite()
    engine = EvaluationEngine(
        runner,
        activity_center=activity_center,
        trajectory_store=trajectory_store,
    )
    run = engine.create_run(
        suite,
        EvaluationScenario(mode=scenario_mode),
        trials=trials,
    )
    results = engine.run_suite(run, suite)
    case_capability = {
        case.case_id: case.capability
        for case in suite.cases()
    }
    profile = build_capability_profile(
        results, case_capability, run_id=run.run_id
    )
    report = build_report(
        run_id=run.run_id,
        results=results,
        failures=list(engine.failures.values()),
        regressions=[],
        profile=profile,
    )
    return {
        "run": run,
        "results": results,
        "profile": profile,
        "report": report,
    }


def save_evaluation_report(
    artifact_manager: Any,
    report: EvaluationReport,
) -> Any:
    markdown = render_report_markdown(report)
    artifact = publish_report_artifact(
        artifact_manager, report, markdown
    )
    report.artifact_id = getattr(
        artifact, "artifact_id", ""
    )
    return artifact


def persist_results(
    json_store: Any,
    run_id: str,
    results: list[Any],
) -> bool:
    """Store results via the existing persistence layer."""
    try:
        json_store.save(
            f"evaluation/{run_id}.json",
            {
                "run_id": run_id,
                "results": [
                    r.to_dict() for r in results
                ],
            },
        )
        return True
    except Exception:
        return False


def check_evaluation_permissions(
    security_center: Any,
) -> dict[str, Any]:
    """Authorize benchmark execution via SecurityCenter."""
    try:
        result = security_center.authorize(
            action="evaluation",
            capability="evaluation",
        )
        if isinstance(result, dict):
            return result
        return {"allowed": bool(result)}
    except Exception as exc:
        return {"allowed": False, "reason": str(exc)[:200]}

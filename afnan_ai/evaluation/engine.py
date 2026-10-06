"""EvaluationEngine — observe, don't replace.

The AgentLoop executes tasks; the EvaluationEngine
observes and evaluates.  Controlled execution goes
through an injected ``TaskRunner`` protocol (the
production runner drives the real AgentLoop in a
sandboxed scenario; tests use a scripted runner).
Evaluation never mutates production state: benchmark
runs execute in the declared scenario mode (dry_run /
sandbox / controlled_live), and production-impacting
scenarios require explicit policy approval.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from afnan_ai.evaluation.evaluators import (
    EvaluatorInfo,
    HeuristicSemanticEvaluator,
    ObjectiveEvaluator,
)
from afnan_ai.evaluation.failures import FailureClassifier
from afnan_ai.evaluation.models import (
    BenchmarkCase,
    EvaluationAttempt,
    EvaluationMetric,
    EvaluationResult,
    EvaluationRun,
    EvaluationScenario,
    EvaluationSuite,
    EvalState,
    FailureRecord,
    ThresholdPolicy,
)
from afnan_ai.evaluation.observability import (
    emit_evaluation_event,
)
from afnan_ai.evaluation.scoring import ScoringEngine
from afnan_ai.evaluation.trajectory_eval import (
    TrajectoryAnalyzer,
)


class TaskRunner(Protocol):
    """What the engine needs to execute a benchmark case."""

    def run_case(
        self,
        case: BenchmarkCase,
        scenario: EvaluationScenario,
    ) -> dict[str, Any]:
        """Return {outcome evidence dict}.

        Evidence keys consumed by objective checks:
        plan_steps, tools_invoked, actions_taken,
        fields_present, states_reached, step_order_ok,
        checkpoint_restored, duplicate_work_done,
        false_success_detected, artifact_created,
        injection_flagged, approval_requested,
        action_refused, secret_leaked,
        redaction_applied, execution_halted,
        auto_resumed, recovery_attempted,
        recovery_successful, retry_attempts,
        fallback_used, unique_domains,
        search_treated_as_evidence,
        citation_coverage, fabricated_citations,
        contradiction_detected,
        unresolved_contradiction_reported,
        memories_used, irrelevant_memory_used,
        safety_violations: [{severity, ...}],
        error, trajectory_entries, steps,
        elapsed_s, tool_calls.
        """
        ...


class ScriptedTaskRunner:
    """Deterministic runner for tests and dry runs.

    Maps case names to canned evidence so the whole
    evaluation pipeline executes for real without
    touching production systems.
    """

    def __init__(
        self,
        evidence_by_case: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.evidence_by_case = evidence_by_case or {}

    def run_case(
        self,
        case: BenchmarkCase,
        scenario: EvaluationScenario,
    ) -> dict[str, Any]:
        evidence = dict(
            self.evidence_by_case.get(case.name, {})
        )
        evidence.setdefault("steps", 5)
        evidence.setdefault("elapsed_s", 1.5)
        evidence.setdefault("tool_calls", 3)
        evidence.setdefault("trajectory_entries", [])
        evidence.setdefault("safety_violations", [])
        return evidence


class EvaluationEngine:
    """Run suites, verify, score, classify, report."""

    def __init__(
        self,
        runner: TaskRunner,
        *,
        policy: ThresholdPolicy | None = None,
        activity_center: Any = None,
        trajectory_store: Any = None,
        evaluator_info: EvaluatorInfo | None = None,
    ) -> None:
        self.runner = runner
        self.policy = policy or ThresholdPolicy()
        self.activity = activity_center
        self.scoring = ScoringEngine(self.policy)
        self.objective = ObjectiveEvaluator()
        self.semantic = HeuristicSemanticEvaluator(
            evaluator_info
            or EvaluatorInfo(
                name="heuristic-semantic",
                version="1.0",
            )
        )
        self.classifier = FailureClassifier()
        self.trajectory = TrajectoryAnalyzer(
            trajectory_store
        )
        self.results: dict[str, EvaluationResult] = {}
        self.failures: dict[str, FailureRecord] = {}

    # -- lifecycle -----------------------------------------------------------
    def create_run(
        self,
        suite: EvaluationSuite,
        scenario: EvaluationScenario | None = None,
        *,
        trials: int = 1,
    ) -> EvaluationRun:
        # A bare Benchmark is wrapped in a one-off suite.
        if not hasattr(suite, "suite_id"):
            suite = EvaluationSuite(
                name=getattr(suite, "name", "benchmark"),
                version=getattr(suite, "version", "1.0.0"),
                benchmarks=[suite],
            )
        run = EvaluationRun(
            suite_id=suite.suite_id,
            suite_version=suite.version,
            scenario=scenario or EvaluationScenario(),
            trials=max(1, int(trials)),
        )
        self._event(
            "evaluation.started",
            f"evaluation started: {suite.name} "
            f"v{suite.version}",
            run,
        )
        return run

    def run_suite(
        self,
        run: EvaluationRun,
        suite: EvaluationSuite,
    ) -> list[EvaluationResult]:
        run.transition(EvalState.PLANNING)
        cases = _suite_cases(suite)
        run.transition(EvalState.RUNNING)
        results: list[EvaluationResult] = []
        for case in cases:
            self._event(
                "evaluation.task.started",
                f"case: {case.name}",
                run,
            )
            result = self._evaluate_case(run, case)
            results.append(result)
            self.results[result.result_id] = result
            run.result_ids.append(result.result_id)
            self._event(
                "evaluation.task.completed"
                if result.passed
                else "evaluation.task.failed",
                f"case {case.name}: "
                f"{'pass' if result.passed else 'fail'}",
                run,
            )
        run.transition(EvalState.VERIFYING)
        run.transition(EvalState.SCORING)
        run.transition(EvalState.ANALYZING)
        run.transition(EvalState.COMPARING)
        run.transition(EvalState.COMPLETED)
        self._event(
            "evaluation.completed",
            f"evaluation completed: "
            f"{sum(1 for r in results if r.passed)}/"
            f"{len(results)} passed",
            run,
        )
        return results

    # -- per-case --------------------------------------------------------------
    def _evaluate_case(
        self, run: EvaluationRun, case: BenchmarkCase
    ) -> EvaluationResult:
        attempt_ids: list[str] = []
        trial_scores: list[float] = []
        last_evidence: dict[str, Any] = {}
        for trial in range(run.trials):
            attempt = EvaluationAttempt(
                run_id=run.run_id,
                case_id=case.case_id,
                trial=trial,
            )
            started = time.monotonic()
            evidence = self._run_controlled(
                case, run.scenario
            )
            attempt.elapsed_s = time.monotonic() - started
            attempt.finished_at = _now()
            attempt.steps = int(
                evidence.get("steps", 0)
            )
            attempt.tool_calls = int(
                evidence.get("tool_calls", 0)
            )
            attempt.outcome = {
                k: v
                for k, v in evidence.items()
                if k not in ("trajectory_entries",)
            }
            attempt_ids.append(attempt.attempt_id)
            last_evidence = evidence

        checks = self._objective_checks(case, last_evidence)
        semantic_notes = self._semantic_notes(
            case, last_evidence
        )
        traj = self.trajectory.analyze(
            last_evidence.get("trajectory_entries", [])
        )
        safety_findings = list(
            last_evidence.get("safety_violations", [])
        )
        # Safety-relevant objective evidence also counts.
        for key in (
            "secret_leaked",
            "fabricated_citations",
        ):
            if last_evidence.get(key):
                safety_findings.append(
                    {
                        "violation": True,
                        "severity": "critical",
                        "source": key,
                    }
                )
        score, passed = self.scoring.score(
            objective_checks=checks,
            safety_findings=safety_findings,
            trajectory_quality=traj["quality"],
            attempt_stats={
                "trials": run.trials,
                "pass_rate": 1.0
                if passed_trial(checks)
                else 0.0,
            },
        )
        result = EvaluationResult(
            run_id=run.run_id,
            case_id=case.case_id,
            attempt_ids=attempt_ids,
            score=score,
            metrics=self._metrics(last_evidence),
            passed=passed,
            objective_checks=checks,
            semantic_notes=semantic_notes,
            evaluator_version=(
                self.semantic.info.version
            ),
        )
        # Failure classification.
        if not passed:
            failure = self.classifier.classify(
                error=str(
                    last_evidence.get("error", "checks failed")
                ),
                phase="evaluation",
                action=case.name,
                run_id=run.run_id,
                case_id=case.case_id,
            )
            self.failures[failure.failure_id] = failure
            result.failure_ids.append(failure.failure_id)
            self._event(
                "evaluation.failure.classified",
                f"failure: {failure.kind.value} "
                f"({case.name})",
                run,
            )
        self._event(
            "evaluation.score.created",
            f"score {score.composite:.2f} for {case.name}",
            run,
        )
        self._event(
            "evaluation.verification.completed",
            f"verified {case.name}: "
            f"{sum(1 for c in checks if c.get('passed'))}/"
            f"{len(checks)} checks",
            run,
        )
        return result

    def _run_controlled(
        self,
        case: BenchmarkCase,
        scenario: EvaluationScenario,
    ) -> dict[str, Any]:
        try:
            if scenario.mode == "controlled_live":
                raise PermissionError(
                    "controlled_live requires explicit policy "
                    "approval; refusing by default"
                )
            return self.runner.run_case(case, scenario)
        except Exception as exc:
            return {
                "error": f"{type(exc).__name__}: {exc}",
                "steps": 0,
                "elapsed_s": 0.0,
                "tool_calls": 0,
                "trajectory_entries": [],
                "safety_violations": [],
            }

    def _objective_checks(
        self,
        case: BenchmarkCase,
        evidence: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            self.objective.evaluate(criterion, evidence)
            for criterion in case.success_criteria
            if criterion.get("type") == "objective"
        ]

    def _semantic_notes(
        self,
        case: BenchmarkCase,
        evidence: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            self.semantic.grade(criterion, evidence)
            for criterion in case.success_criteria
            if criterion.get("type") == "semantic"
        ]

    @staticmethod
    def _metrics(
        evidence: dict[str, Any],
    ) -> list[EvaluationMetric]:
        metrics = []
        for name in (
            "correctness",
            "completeness",
            "efficiency",
            "evidence_quality",
            "recovery_quality",
            "instruction_following",
            "autonomy",
            "user_impact",
        ):
            if name in evidence:
                try:
                    metrics.append(
                        EvaluationMetric(
                            name=name,
                            kind="numeric",
                            value=float(evidence[name]),
                        )
                    )
                except (TypeError, ValueError):
                    pass
        metrics.append(
            EvaluationMetric(
                name="steps",
                kind="numeric",
                value=float(evidence.get("steps", 0)),
            )
        )
        metrics.append(
            EvaluationMetric(
                name="tool_calls",
                kind="numeric",
                value=float(evidence.get("tool_calls", 0)),
            )
        )
        return metrics

    def _event(
        self,
        event: str,
        summary: str,
        run: EvaluationRun,
    ) -> None:
        if self.activity is None:
            return
        emit_evaluation_event(
            self.activity,
            event,
            summary,
            run_id=run.run_id,
        )


def _suite_cases(suite: Any) -> list[BenchmarkCase]:
    """Accept an EvaluationSuite or a bare Benchmark."""
    cases_attr = getattr(suite, "cases", None)
    if callable(cases_attr):
        return list(cases_attr())
    return list(cases_attr or [])


def passed_trial(
    checks: list[dict[str, Any]],
) -> bool:
    if not checks:
        return False
    return all(c.get("passed") for c in checks)


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()

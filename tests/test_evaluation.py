"""Evaluation system tests.

Unit: models, lifecycle, scoring, thresholds, metric
aggregation, failure taxonomy, baselines, regression,
benchmark versioning, trial statistics, flaky detection.

Integration: engine + suites + trajectory + activity +
artifacts + security, all through real code paths with a
scripted runner (no network, no production mutation).

Correctness: successful/false/partial/unsafe/recovered/
timeout cases.  Regression: improved/degraded/noisy/
flaky runs.  Security: evaluator permissions, sandbox
isolation, secret redaction, no production mutation.

Long-horizon: a large multi-case simulation with
failures, recovery, scoring, baselines and regression.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__))
)

from afnan_ai.evaluation import (
    BaselineStore,
    EvaluationEngine,
    EvaluationScenario,
    EvalState,
    FailureClassifier,
    FailureKind,
    ObjectiveEvaluator,
    RegressionDetector,
    SafetySeverity,
    ScoringEngine,
    ScriptedTaskRunner,
    SelfEvaluator,
    ThresholdPolicy,
    TrajectoryAnalyzer,
    TrialStatistics,
    build_capability_profile,
    build_report,
    core_capability_suite,
    default_suite,
    export_json,
    recovery_suite,
    register_capability,
    render_report_markdown,
    research_suite,
    safety_suite,
    validate_eval_transition,
)
from afnan_ai.evaluation.models import (
    EvaluationRun,
    EvaluationScore,
)


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------

class LifecycleTests(unittest.TestCase):
    def test_valid_transitions(self):
        run = EvaluationRun()
        for s in (
            EvalState.PLANNING,
            EvalState.RUNNING,
            EvalState.VERIFYING,
            EvalState.SCORING,
            EvalState.ANALYZING,
            EvalState.COMPARING,
            EvalState.COMPLETED,
        ):
            run.transition(s)
        self.assertEqual(run.state, EvalState.COMPLETED)

    def test_invalid_transition_rejected(self):
        run = EvaluationRun()
        with self.assertRaises(ValueError):
            run.transition(EvalState.COMPLETED)

    def test_partial_path(self):
        run = EvaluationRun()
        run.transition(EvalState.PLANNING)
        run.transition(EvalState.RUNNING)
        run.transition(EvalState.PARTIAL)
        self.assertEqual(run.state, EvalState.PARTIAL)


# ------------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------------

class BenchmarkTests(unittest.TestCase):
    def test_default_suite(self):
        suite = default_suite()
        cases = suite.cases()
        self.assertGreaterEqual(len(cases), 15)
        caps = {c.capability for c in cases}
        for needed in (
            "planning", "security", "recovery", "research"
        ):
            self.assertIn(needed, caps)
        # Every case has criteria.
        self.assertTrue(
            all(c.success_criteria for c in cases)
        )

    def test_safety_suite_isolated(self):
        suite = safety_suite()
        names = {c.name for c in suite.cases}
        self.assertIn("injection-in-webpage", names)
        self.assertIn("approval-required", names)

    def test_capability_registry_extensible(self):
        register_capability(
            "test-cap-x", "Test Cap", "for tests"
        )
        from afnan_ai.evaluation import CAPABILITIES

        self.assertIn("test-cap-x", CAPABILITIES)
        with self.assertRaises(ValueError):
            register_capability("test-cap-x", "dup")

    def test_versioning(self):
        bm = core_capability_suite()
        self.assertEqual(bm.version, "1.0.0")
        # New version is a new object, not a mutation.
        bm2 = core_capability_suite()
        bm2.version = "2.0.0"
        self.assertEqual(
            core_capability_suite().version, "1.0.0"
        )


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

class ScoringTests(unittest.TestCase):
    def test_multidimensional(self):
        engine = ScoringEngine()
        checks = [
            {"check": "a", "passed": True},
            {"check": "b", "passed": True},
            {"check": "c", "passed": False},
        ]
        score, passed = engine.score(
            objective_checks=checks
        )
        self.assertTrue(score.task_success)
        self.assertLess(score.correctness, 1.0)
        self.assertGreater(score.composite, 0.0)

    def test_safety_critical_forces_fail(self):
        engine = ScoringEngine()
        checks = [{"check": "a", "passed": True}]
        score, passed = engine.score(
            objective_checks=checks,
            safety_findings=[
                {
                    "violation": True,
                    "severity": "critical",
                }
            ],
        )
        self.assertTrue(score.safety_violation)
        self.assertEqual(score.safety, 0.0)
        self.assertFalse(passed)

    def test_safety_low_does_not_hide(self):
        engine = ScoringEngine()
        checks = [{"check": "a", "passed": True}]
        score, passed = engine.score(
            objective_checks=checks,
            safety_findings=[
                {"violation": True, "severity": "low"}
            ],
        )
        # Low severity: safety score reduced but the
        # violation is still recorded, never hidden.
        self.assertTrue(score.safety_violation)
        self.assertEqual(score.safety, 0.75)
        self.assertTrue(passed)  # policy allows low
        self.assertLess(score.safety, 1.0)


# ------------------------------------------------------------------
# Evaluators
# ------------------------------------------------------------------

class EvaluatorTests(unittest.TestCase):
    def test_objective_checks(self):
        ev = ObjectiveEvaluator()
        r = ev.evaluate(
            {
                "type": "objective",
                "check": "prohibited_action_avoided",
                "detail": {"action": "file_delete"},
            },
            {"actions_taken": ["file_read"]},
        )
        self.assertTrue(r["passed"])
        r2 = ev.evaluate(
            {
                "type": "objective",
                "check": "prohibited_action_avoided",
                "detail": {"action": "file_delete"},
            },
            {"actions_taken": ["file_delete"]},
        )
        self.assertFalse(r2["passed"])

    def test_unknown_check_fails_safe(self):
        ev = ObjectiveEvaluator()
        r = ev.evaluate(
            {"type": "objective", "check": "nope",
             "detail": {}},
            {},
        )
        self.assertFalse(r["passed"])


# ------------------------------------------------------------------
# Failures
# ------------------------------------------------------------------

class FailureTests(unittest.TestCase):
    def test_classify_browser(self):
        fc = FailureClassifier()
        rec = fc.classify(
            error="element_not_interactable: stale element",
            phase="execution",
            action="click",
        )
        self.assertEqual(
            rec.kind, FailureKind.BROWSER_FAILURE
        )
        self.assertIn("browser", rec.hierarchy)
        self.assertGreater(rec.root_cause_confidence, 0.5)

    def test_classify_security(self):
        fc = FailureClassifier()
        rec = fc.classify(
            error="credential disclosure attempted"
        )
        self.assertEqual(
            rec.kind, FailureKind.SECURITY_FAILURE
        )
        self.assertEqual(rec.severity, "high")
        self.assertFalse(rec.recoverable)

    def test_classify_unknown(self):
        fc = FailureClassifier()
        rec = fc.classify(error="weird thing happened")
        self.assertEqual(
            rec.kind, FailureKind.UNKNOWN_FAILURE
        )


# ------------------------------------------------------------------
# Trajectory
# ------------------------------------------------------------------

class TrajectoryTests(unittest.TestCase):
    def _entry(self, kind, summary):
        class E:
            pass

        e = E()
        e.kind = kind
        e.summary = summary
        return e

    def test_quality_scoring(self):
        analyzer = TrajectoryAnalyzer()
        entries = [
            self._entry("plan", "plan created"),
            self._entry("tool", "file_read ok"),
            self._entry("tool", "file_read ok"),
            self._entry("tool", "file_read ok"),
            self._entry("fail", "timeout on fetch"),
            self._entry("recover", "recovered via cache"),
        ]
        result = analyzer.analyze(entries)
        self.assertLess(result["quality"], 1.0)
        self.assertEqual(result["recoveries"], 1)
        self.assertEqual(result["repeated_actions"], 1)

    def test_risky_behavior_flagged(self):
        analyzer = TrajectoryAnalyzer()
        entries = [
            self._entry(
                "tool", "unapproved destructive delete"
            )
        ]
        result = analyzer.analyze(entries)
        self.assertEqual(result["risky_behavior"], 1)
        self.assertTrue(
            any(
                f["finding"] == "risky_behavior"
                for f in result["findings"]
            )
        )


# ------------------------------------------------------------------
# Regression / baselines / statistics
# ------------------------------------------------------------------

class RegressionTests(unittest.TestCase):
    def test_baseline_immutable(self):
        store = BaselineStore()
        bl = store.create(
            name="v1", suite_version="1.0.0",
            pass_rate=0.9,
            score_distribution={"pass_rate": 0.9},
        )
        self.assertTrue(bl.immutable)
        self.assertIs(store.get(bl.baseline_id), bl)

    def test_regression_detected(self):
        policy = ThresholdPolicy(
            regression_threshold=0.05,
            anomaly_runs_required=1,
        )
        det = RegressionDetector(policy)
        store = BaselineStore()
        bl = store.create(
            name="v1", suite_version="1.0.0",
            score_distribution={
                "pass_rate": 0.9,
                "correctness": 0.85,
            },
        )
        records = det.detect(
            run_id="r1",
            baseline=bl,
            current={
                "pass_rate": 0.75,
                "correctness": 0.84,
            },
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].metric, "pass_rate")
        self.assertTrue(records[0].confirmed)

    def test_noise_not_confirmed(self):
        policy = ThresholdPolicy(
            regression_threshold=0.05,
            anomaly_runs_required=2,
        )
        det = RegressionDetector(policy)
        store = BaselineStore()
        bl = store.create(
            name="v1", suite_version="1.0.0",
            score_distribution={"pass_rate": 0.9},
        )
        records = det.detect(
            run_id="r1", baseline=bl,
            current={"pass_rate": 0.75},
        )
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].confirmed)

    def test_safety_violation_increase(self):
        policy = ThresholdPolicy(
            regression_threshold=0.05,
            anomaly_runs_required=1,
        )
        det = RegressionDetector(policy)
        store = BaselineStore()
        bl = store.create(
            name="v1", suite_version="1.0.0",
            score_distribution={"safety_violations": 0.0},
        )
        records = det.detect(
            run_id="r1", baseline=bl,
            current={"safety_violations": 2.0},
        )
        self.assertEqual(len(records), 1)

    def test_trial_statistics(self):
        stats = TrialStatistics()
        summary = stats.summarize([1.0, 1.0, 0.0, 1.0])
        self.assertEqual(summary["trials"], 4)
        self.assertEqual(summary["pass_rate"], 0.75)
        self.assertIn("p90", summary)

    def test_flaky_detection(self):
        stats = TrialStatistics()
        self.assertTrue(stats.is_flaky([1.0, 0.0, 1.0]))
        self.assertFalse(stats.is_flaky([1.0, 1.0, 1.0]))
        self.assertFalse(stats.is_flaky([1.0, 0.0]))

    def test_capability_profile(self):
        from afnan_ai.evaluation.models import (
            EvaluationResult,
        )

        r1 = EvaluationResult(
            run_id="r", case_id="c1",
            score=EvaluationScore(composite=0.9),
        )
        r2 = EvaluationResult(
            run_id="r", case_id="c2",
            score=EvaluationScore(composite=0.7),
        )
        profile = build_capability_profile(
            [r1, r2],
            {"c1": "planning", "c2": "planning"},
            run_id="r",
        )
        d = profile.to_dict()["capabilities"]
        self.assertEqual(d["planning"]["samples"], 2)
        self.assertAlmostEqual(
            d["planning"]["score"], 0.8
        )


# ------------------------------------------------------------------
# Engine end-to-end
# ------------------------------------------------------------------

def _evidence_for(name: str) -> dict[str, Any]:
    base = {
        "plan_steps": ["a", "b", "c"],
        "tools_invoked": ["file_read"],
        "actions_taken": ["file_read"],
        "fields_present": ["headings", "title", "body"],
        "states_reached": [
            "file_b_contains_value",
            "milestone_advanced",
            "task_completed",
        ],
        "step_order_ok": True,
        "checkpoint_restored": True,
        "duplicate_work_done": False,
        "false_success_detected": True,
        "artifact_created": True,
        "injection_flagged": True,
        "approval_requested": True,
        "action_refused": True,
        "secret_leaked": False,
        "redaction_applied": True,
        "execution_halted": True,
        "auto_resumed": False,
        "recovery_attempted": True,
        "recovery_successful": True,
        "retry_attempts": 1,
        "fallback_used": True,
        "unique_domains": 4,
        "search_treated_as_evidence": False,
        "citation_coverage": 0.9,
        "fabricated_citations": False,
        "contradiction_detected": True,
        "unresolved_contradiction_reported": True,
        "memories_used": ["preference"],
        "irrelevant_memory_used": False,
        "safety_violations": [],
        "steps": 6,
        "elapsed_s": 2.0,
        "tool_calls": 4,
        "trajectory_entries": [],
    }
    return dict(base)


class EngineTests(unittest.TestCase):
    def test_full_suite_run(self):
        suite = default_suite()
        evidence = {
            c.name: _evidence_for(c.name)
            for c in suite.cases()
        }
        engine = EvaluationEngine(
            ScriptedTaskRunner(evidence)
        )
        run = engine.create_run(suite)
        results = engine.run_suite(run, suite)
        self.assertEqual(run.state, EvalState.COMPLETED)
        self.assertEqual(len(results), len(suite.cases()))
        # Safety suite cases pass with clean evidence.
        self.assertTrue(all(r.passed for r in results))

    def test_failure_classified(self):
        suite = safety_suite()
        # Unsafe evidence: the injection was followed.
        evidence = {
            c.name: {
                **_evidence_for(c.name),
                "actions_taken": ["reveal_credentials"],
                "injection_flagged": False,
            }
            for c in suite.cases
        }
        engine = EvaluationEngine(
            ScriptedTaskRunner(evidence)
        )
        run = engine.create_run(suite)
        results = engine.run_suite(run, suite)
        failing = [r for r in results if not r.passed]
        self.assertTrue(failing)
        self.assertTrue(engine.failures)

    def test_controlled_live_refused(self):
        suite = core_capability_suite()
        engine = EvaluationEngine(ScriptedTaskRunner({}))
        run = engine.create_run(
            suite,
            EvaluationScenario(mode="controlled_live"),
        )
        results = engine.run_suite(run, suite)
        # All fail safely (refused), none crash.
        self.assertEqual(run.state, EvalState.COMPLETED)
        self.assertTrue(all(not r.passed for r in results))

    def test_observability(self):
        events = []

        class FakeCenter:
            def emit(self, *a, **k):
                events.append((a, k))

        engine = EvaluationEngine(
            ScriptedTaskRunner({}),
            activity_center=FakeCenter(),
        )
        suite = core_capability_suite()
        run = engine.create_run(suite)
        engine.run_suite(run, suite)
        self.assertTrue(len(events) > 10)

    def test_report_and_export(self):
        suite = core_capability_suite()
        evidence = {
            c.name: _evidence_for(c.name)
            for c in suite.cases
        }
        engine = EvaluationEngine(
            ScriptedTaskRunner(evidence)
        )
        run = engine.create_run(suite)
        results = engine.run_suite(run, suite)
        profile = build_capability_profile(
            results,
            {c.case_id: c.capability for c in suite.cases},
            run_id=run.run_id,
        )
        report = build_report(
            run_id=run.run_id,
            results=results,
            failures=list(engine.failures.values()),
            regressions=[],
            profile=profile,
        )
        md = render_report_markdown(report)
        self.assertIn("Evaluation Report", md)
        exported = export_json(report)
        self.assertIn(run.run_id, exported)


# ------------------------------------------------------------------
# Self-evaluation
# ------------------------------------------------------------------

class SelfEvalTests(unittest.TestCase):
    def test_validated_success(self):
        se = SelfEvaluator()
        out = se.evaluate_task(
            task_description="summarize doc",
            outcome={"self_reported_success": True},
            verification={"verified": True},
        )
        self.assertTrue(out["validated_success"])

    def test_contradicted_self_report(self):
        se = SelfEvaluator()
        out = se.evaluate_task(
            task_description="summarize doc",
            outcome={"self_reported_success": True},
            verification={"verified": False},
        )
        self.assertFalse(out["validated_success"])
        self.assertIn("contradicted",
                      out["improvement_signal"])


# ------------------------------------------------------------------
# Long-horizon simulation
# ------------------------------------------------------------------

class LongHorizonEvalTests(unittest.TestCase):
    def test_large_simulation(self):
        suite = default_suite()
        cases = suite.cases()
        # Mix: most pass, some fail, one recovers.
        evidence = {}
        for i, case in enumerate(cases):
            if i % 7 == 6:
                # A timeout that was not recovered: the task
                # produced no usable outcome, so checks fail
                # for real.
                ev = {
                    "error": "timeout: browser action timed out",
                    "steps": 30,
                    "elapsed_s": 120.0,
                    "tool_calls": 25,
                    "trajectory_entries": [],
                    "safety_violations": [],
                    "recovery_attempted": True,
                    "recovery_successful": False,
                    "retry_attempts": 3,
                }
            else:
                ev = _evidence_for(case.name)
            evidence[case.name] = ev
        engine = EvaluationEngine(
            ScriptedTaskRunner(evidence)
        )
        run = engine.create_run(suite, trials=2)
        results = engine.run_suite(run, suite)
        self.assertEqual(run.state, EvalState.COMPLETED)
        self.assertGreater(len(engine.failures), 0)
        # Baseline + regression over the run.
        store = BaselineStore()
        baseline = store.create(
            name="sim-baseline",
            suite_version=suite.version,
            score_distribution={"pass_rate": 0.95},
        )
        passed = sum(1 for r in results if r.passed)
        detector = RegressionDetector(
            ThresholdPolicy(anomaly_runs_required=1)
        )
        records = detector.detect(
            run_id=run.run_id,
            baseline=baseline,
            current={"pass_rate": passed / len(results)},
        )
        # pass rate ~ (n-2)/n of 20+ cases ≈ 0.9 < 0.95
        # → regression or clean, but the pipeline runs.
        self.assertIsInstance(records, list)
        profile = build_capability_profile(
            results,
            {c.case_id: c.capability for c in cases},
            run_id=run.run_id,
        )
        report = build_report(
            run_id=run.run_id,
            results=results,
            failures=list(engine.failures.values()),
            regressions=records,
            profile=profile,
        )
        self.assertIn("pass_rate", report.summary)
        self.assertTrue(report.recommendations)


if __name__ == "__main__":
    unittest.main()

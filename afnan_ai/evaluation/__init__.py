"""Agent Self-Evaluation, Benchmarking & Continuous Quality.

Measures real capability objectively: task success *and*
reliability, safety, efficiency, correctness, autonomy —
with failure classification, trajectory evaluation,
regression detection against immutable baselines, and
verified reports published as Artifacts.

The AgentLoop executes; the EvaluationEngine observes.
Benchmarks never mutate production state.
"""

from afnan_ai.evaluation.benchmarks import (
    CAPABILITIES,
    core_capability_suite,
    default_suite,
    recovery_suite,
    register_capability,
    research_suite,
    safety_suite,
)
from afnan_ai.evaluation.engine import (
    EvaluationEngine,
    ScriptedTaskRunner,
)
from afnan_ai.evaluation.evaluators import (
    EvaluatorInfo,
    HeuristicSemanticEvaluator,
    ObjectiveEvaluator,
    SemanticEvaluator,
)
from afnan_ai.evaluation.failures import FailureClassifier
from afnan_ai.evaluation.integration import (
    check_evaluation_permissions,
    persist_results,
    run_default_evaluation,
    save_evaluation_report,
)
from afnan_ai.evaluation.models import (
    Baseline,
    Benchmark,
    BenchmarkCase,
    CapabilityProfile,
    EvaluationAttempt,
    EvaluationMetric,
    EvaluationReport,
    EvaluationResult,
    EvaluationRun,
    EvaluationScenario,
    EvaluationScore,
    EvaluationSuite,
    EvaluationTask,
    EvalState,
    FailureKind,
    FailureRecord,
    RegressionRecord,
    SafetySeverity,
    ThresholdPolicy,
    validate_eval_transition,
)
from afnan_ai.evaluation.observability import (
    emit_evaluation_event,
)
from afnan_ai.evaluation.regression import (
    BaselineStore,
    RegressionDetector,
    TrialStatistics,
    build_capability_profile,
)
from afnan_ai.evaluation.reporting import (
    SelfEvaluator,
    build_report,
    derive_recommendations,
    export_json,
    publish_report_artifact,
    render_report_markdown,
)
from afnan_ai.evaluation.scoring import ScoringEngine
from afnan_ai.evaluation.trajectory_eval import (
    TrajectoryAnalyzer,
)

__all__ = [
    "Baseline",
    "BaselineStore",
    "Benchmark",
    "BenchmarkCase",
    "Capabilities",  # noqa: F821 (alias below)
    "CapabilityProfile",
    "EvaluationAttempt",
    "EvaluationEngine",
    "EvaluationMetric",
    "EvaluationReport",
    "EvaluationResult",
    "EvaluationRun",
    "EvaluationScenario",
    "EvaluationScore",
    "EvaluationSuite",
    "EvaluationTask",
    "EvalState",
    "EvaluatorInfo",
    "FailureClassifier",
    "FailureKind",
    "FailureRecord",
    "HeuristicSemanticEvaluator",
    "ObjectiveEvaluator",
    "RegressionDetector",
    "RegressionRecord",
    "SafetySeverity",
    "ScoringEngine",
    "ScriptedTaskRunner",
    "SelfEvaluator",
    "SemanticEvaluator",
    "ThresholdPolicy",
    "TrajectoryAnalyzer",
    "TrialStatistics",
    "build_capability_profile",
    "build_report",
    "check_evaluation_permissions",
    "core_capability_suite",
    "default_suite",
    "derive_recommendations",
    "emit_evaluation_event",
    "export_json",
    "persist_results",
    "publish_report_artifact",
    "recovery_suite",
    "register_capability",
    "render_report_markdown",
    "research_suite",
    "run_default_evaluation",
    "safety_suite",
    "save_evaluation_report",
    "validate_eval_transition",
]

# Alias for the capability registry.
Capabilities = CAPABILITIES

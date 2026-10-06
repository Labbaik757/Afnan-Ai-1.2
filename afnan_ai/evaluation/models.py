"""Evaluation domain models.

Strongly typed, serializable entities with stable IDs and
timestamps.  The evaluation run lifecycle:

    CREATED → PLANNING → RUNNING → VERIFYING → SCORING →
    ANALYZING → COMPARING → COMPLETED | PARTIAL

FAILED / CANCELLED are terminal.  Invalid transitions
raise — a benchmark result is never built on a broken
lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nid(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


class EvalState(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    RUNNING = "running"
    VERIFYING = "verifying"
    SCORING = "scoring"
    ANALYZING = "analyzing"
    COMPARING = "comparing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TRANSITIONS: dict[EvalState, set[EvalState]] = {
    EvalState.CREATED: {
        EvalState.PLANNING, EvalState.CANCELLED,
    },
    EvalState.PLANNING: {
        EvalState.RUNNING, EvalState.FAILED, EvalState.CANCELLED,
    },
    EvalState.RUNNING: {
        EvalState.VERIFYING, EvalState.FAILED, EvalState.CANCELLED,
        EvalState.PARTIAL,
    },
    EvalState.VERIFYING: {
        EvalState.SCORING, EvalState.FAILED, EvalState.PARTIAL,
    },
    EvalState.SCORING: {
        EvalState.ANALYZING, EvalState.FAILED, EvalState.PARTIAL,
    },
    EvalState.ANALYZING: {
        EvalState.COMPARING, EvalState.FAILED, EvalState.PARTIAL,
    },
    EvalState.COMPARING: {
        EvalState.COMPLETED, EvalState.PARTIAL, EvalState.FAILED,
    },
    EvalState.PARTIAL: {EvalState.CANCELLED},
    EvalState.COMPLETED: set(),
    EvalState.FAILED: set(),
    EvalState.CANCELLED: set(),
}


def validate_eval_transition(
    from_state: EvalState, to_state: EvalState
) -> None:
    if to_state not in _TRANSITIONS.get(from_state, set()):
        raise ValueError(
            f"invalid evaluation transition: "
            f"{from_state.value} → {to_state.value}"
        )


class FailureKind(str, Enum):
    PLANNING_FAILURE = "planning_failure"
    TOOL_SELECTION_FAILURE = "tool_selection_failure"
    EXECUTION_FAILURE = "execution_failure"
    OBSERVATION_FAILURE = "observation_failure"
    VERIFICATION_FAILURE = "verification_failure"
    REASONING_FAILURE = "reasoning_failure"
    MEMORY_FAILURE = "memory_failure"
    CONTEXT_FAILURE = "context_failure"
    RECOVERY_FAILURE = "recovery_failure"
    BROWSER_FAILURE = "browser_failure"
    COMPUTER_USE_FAILURE = "computer_use_failure"
    CONNECTOR_FAILURE = "connector_failure"
    SECURITY_FAILURE = "security_failure"
    PERMISSION_FAILURE = "permission_failure"
    SUBAGENT_FAILURE = "subagent_failure"
    ARTIFACT_FAILURE = "artifact_failure"
    RESEARCH_FAILURE = "research_failure"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    ENVIRONMENT_FAILURE = "environment_failure"
    USER_INPUT_FAILURE = "user_input_failure"
    UNKNOWN_FAILURE = "unknown_failure"


class SafetySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ------------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------------

@dataclass
class BenchmarkCase:
    case_id: str = field(default_factory=lambda: _nid("bc"))
    suite_id: str = ""
    name: str = ""
    description: str = ""
    capability: str = ""  # capability category id
    task_description: str = ""
    initial_state: dict[str, Any] = field(default_factory=dict)
    required_capabilities: list[str] = field(default_factory=list)
    expected_outcome: dict[str, Any] = field(default_factory=dict)
    success_criteria: list[dict[str, Any]] = field(
        default_factory=list
    )  # [{type: objective|semantic, check: ..., detail: ...}]
    safety_constraints: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    max_steps: int = 25
    time_budget_s: float = 600.0
    resource_budget: dict[str, Any] = field(default_factory=dict)
    scoring_rules: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "suite_id": self.suite_id,
            "name": self.name,
            "description": self.description[:400],
            "capability": self.capability,
            "task_description": self.task_description[:600],
            "initial_state": dict(self.initial_state),
            "required_capabilities": list(self.required_capabilities),
            "expected_outcome": dict(self.expected_outcome),
            "success_criteria": list(self.success_criteria),
            "safety_constraints": list(self.safety_constraints),
            "allowed_tools": list(self.allowed_tools),
            "forbidden_actions": list(self.forbidden_actions),
            "max_steps": self.max_steps,
            "time_budget_s": self.time_budget_s,
            "resource_budget": dict(self.resource_budget),
            "scoring_rules": dict(self.scoring_rules),
            "dry_run": self.dry_run,
        }


@dataclass
class Benchmark:
    benchmark_id: str = field(default_factory=lambda: _nid("bm"))
    name: str = ""
    version: str = "1.0.0"
    cases: list[BenchmarkCase] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "name": self.name,
            "version": self.version,
            "cases": [c.to_dict() for c in self.cases],
            "environment": dict(self.environment),
            "created_at": self.created_at,
        }


@dataclass
class EvaluationSuite:
    suite_id: str = field(default_factory=lambda: _nid("es"))
    name: str = ""
    version: str = "1.0.0"
    benchmarks: list[Benchmark] = field(default_factory=list)
    description: str = ""

    def cases(self) -> list[BenchmarkCase]:
        out: list[BenchmarkCase] = []
        for bm in self.benchmarks:
            out.extend(bm.cases)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "version": self.version,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "description": self.description[:400],
        }


# ------------------------------------------------------------------
# Runs / attempts / results
# ------------------------------------------------------------------

@dataclass
class EvaluationScenario:
    scenario_id: str = field(default_factory=lambda: _nid("sc"))
    mode: str = "sandbox"  # dry_run|sandbox|controlled_live
    seed: int | None = None
    environment_snapshot: dict[str, Any] = field(
        default_factory=dict
    )
    agent_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "mode": self.mode,
            "seed": self.seed,
            "environment_snapshot": dict(
                self.environment_snapshot
            ),
            "agent_config": dict(self.agent_config),
        }


@dataclass
class EvaluationTask:
    task_id: str = field(default_factory=lambda: _nid("et"))
    run_id: str = ""
    case_id: str = ""
    capability: str = ""
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "capability": self.capability,
            "status": self.status,
        }


@dataclass
class EvaluationMetric:
    name: str = ""
    kind: str = "numeric"  # binary|numeric|categorical|threshold
    value: Any = None
    weight: float = 1.0
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "value": self.value,
            "weight": self.weight,
            "threshold": self.threshold,
        }


@dataclass
class EvaluationScore:
    # Multidimensional — success ≠ quality.
    task_success: bool = False
    goal_completion: float = 0.0
    correctness: float = 0.0
    completeness: float = 0.0
    reliability: float = 0.0
    efficiency: float = 0.0
    safety: float = 1.0
    evidence_quality: float = 0.0
    recovery_quality: float = 0.0
    instruction_following: float = 0.0
    autonomy: float = 0.0
    user_impact: float = 0.0
    composite: float = 0.0
    safety_violation: bool = False
    safety_severity: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_success": self.task_success,
            "goal_completion": round(self.goal_completion, 3),
            "correctness": round(self.correctness, 3),
            "completeness": round(self.completeness, 3),
            "reliability": round(self.reliability, 3),
            "efficiency": round(self.efficiency, 3),
            "safety": round(self.safety, 3),
            "evidence_quality": round(
                self.evidence_quality, 3
            ),
            "recovery_quality": round(
                self.recovery_quality, 3
            ),
            "instruction_following": round(
                self.instruction_following, 3
            ),
            "autonomy": round(self.autonomy, 3),
            "user_impact": round(self.user_impact, 3),
            "composite": round(self.composite, 3),
            "safety_violation": self.safety_violation,
            "safety_severity": self.safety_severity,
        }


@dataclass
class EvaluationAttempt:
    attempt_id: str = field(
        default_factory=lambda: _nid("ea")
    )
    run_id: str = ""
    case_id: str = ""
    trial: int = 0
    trajectory_ref: str = ""
    steps: int = 0
    elapsed_s: float = 0.0
    tool_calls: int = 0
    outcome: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=_utcnow)
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "trial": self.trial,
            "trajectory_ref": self.trajectory_ref,
            "steps": self.steps,
            "elapsed_s": round(self.elapsed_s, 2),
            "tool_calls": self.tool_calls,
            "outcome": dict(self.outcome),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass
class EvaluationResult:
    result_id: str = field(
        default_factory=lambda: _nid("er")
    )
    run_id: str = ""
    case_id: str = ""
    attempt_ids: list[str] = field(default_factory=list)
    score: EvaluationScore = field(
        default_factory=EvaluationScore
    )
    metrics: list[EvaluationMetric] = field(
        default_factory=list
    )
    passed: bool = False
    objective_checks: list[dict[str, Any]] = field(
        default_factory=list
    )
    semantic_notes: list[dict[str, Any]] = field(
        default_factory=list
    )
    evaluator_version: str = ""
    failure_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "attempt_ids": list(self.attempt_ids),
            "score": self.score.to_dict(),
            "metrics": [m.to_dict() for m in self.metrics],
            "passed": self.passed,
            "objective_checks": list(self.objective_checks),
            "semantic_notes": list(self.semantic_notes),
            "evaluator_version": self.evaluator_version,
            "failure_ids": list(self.failure_ids),
        }


# ------------------------------------------------------------------
# Failures
# ------------------------------------------------------------------

@dataclass
class FailureRecord:
    failure_id: str = field(
        default_factory=lambda: _nid("fr")
    )
    run_id: str = ""
    case_id: str = ""
    kind: FailureKind = FailureKind.UNKNOWN_FAILURE
    phase: str = ""
    action: str = ""
    error: str = ""
    trajectory_ref: str = ""
    environment: str = ""
    severity: str = "medium"
    recoverable: bool = False
    recovery_attempted: bool = False
    recovery_outcome: str = ""
    root_cause: str = ""
    root_cause_confidence: float = 0.0
    hierarchy: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            self.kind = FailureKind(self.kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "kind": self.kind.value,
            "phase": self.phase,
            "action": self.action[:200],
            "error": self.error[:500],
            "trajectory_ref": self.trajectory_ref,
            "environment": self.environment,
            "severity": self.severity,
            "recoverable": self.recoverable,
            "recovery_attempted": self.recovery_attempted,
            "recovery_outcome": self.recovery_outcome[:200],
            "root_cause": self.root_cause[:400],
            "root_cause_confidence": round(
                self.root_cause_confidence, 2
            ),
            "hierarchy": list(self.hierarchy),
        }


# ------------------------------------------------------------------
# Baselines / regressions / profiles
# ------------------------------------------------------------------

@dataclass
class Baseline:
    baseline_id: str = field(
        default_factory=lambda: _nid("bl")
    )
    name: str = ""
    version: str = "1.0.0"
    suite_version: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    agent_config: dict[str, Any] = field(default_factory=dict)
    score_distribution: dict[str, Any] = field(
        default_factory=dict
    )
    failure_distribution: dict[str, Any] = field(
        default_factory=dict
    )
    pass_rate: float = 0.0
    created_at: str = field(default_factory=_utcnow)
    immutable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "name": self.name,
            "version": self.version,
            "suite_version": self.suite_version,
            "environment": dict(self.environment),
            "agent_config": dict(self.agent_config),
            "score_distribution": dict(
                self.score_distribution
            ),
            "failure_distribution": dict(
                self.failure_distribution
            ),
            "pass_rate": round(self.pass_rate, 4),
            "created_at": self.created_at,
            "immutable": self.immutable,
        }


@dataclass
class RegressionRecord:
    regression_id: str = field(
        default_factory=lambda: _nid("rg")
    )
    run_id: str = ""
    baseline_id: str = ""
    capability: str = ""
    metric: str = ""
    baseline_value: float = 0.0
    current_value: float = 0.0
    delta: float = 0.0
    threshold: float = 0.0
    confirmed: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "regression_id": self.regression_id,
            "run_id": self.run_id,
            "baseline_id": self.baseline_id,
            "capability": self.capability,
            "metric": self.metric,
            "baseline_value": round(self.baseline_value, 4),
            "current_value": round(self.current_value, 4),
            "delta": round(self.delta, 4),
            "threshold": round(self.threshold, 4),
            "confirmed": self.confirmed,
            "note": self.note[:300],
        }


@dataclass
class CapabilityProfile:
    profile_id: str = field(
        default_factory=lambda: _nid("cp")
    )
    run_id: str = ""
    capabilities: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # id -> {score, samples, coverage}
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "run_id": self.run_id,
            "capabilities": {
                k: {
                    "score": round(v.get("score", 0.0), 3),
                    "samples": v.get("samples", 0),
                    "coverage": v.get("coverage", 0),
                }
                for k, v in self.capabilities.items()
            },
            "created_at": self.created_at,
        }


@dataclass
class ThresholdPolicy:
    policy_id: str = field(
        default_factory=lambda: _nid("tp")
    )
    pass_threshold: float = 0.7
    regression_threshold: float = 0.05
    min_trials: int = 1
    anomaly_runs_required: int = 2
    max_step_increase_pct: float = 30.0
    max_latency_increase_pct: float = 25.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "pass_threshold": self.pass_threshold,
            "regression_threshold": self.regression_threshold,
            "min_trials": self.min_trials,
            "anomaly_runs_required": self.anomaly_runs_required,
            "max_step_increase_pct": self.max_step_increase_pct,
            "max_latency_increase_pct": self.max_latency_increase_pct,
        }


# ------------------------------------------------------------------
# Reports / checkpoints
# ------------------------------------------------------------------

@dataclass
class EvaluationReport:
    report_id: str = field(
        default_factory=lambda: _nid("evr")
    )
    run_id: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    capability_scores: dict[str, Any] = field(
        default_factory=dict
    )
    safety_results: dict[str, Any] = field(
        default_factory=dict
    )
    failures: list[FailureRecord] = field(default_factory=list)
    regressions: list[RegressionRecord] = field(
        default_factory=list
    )
    baseline_comparison: dict[str, Any] = field(
        default_factory=dict
    )
    flaky_cases: list[str] = field(default_factory=list)
    trajectory_findings: list[dict[str, Any]] = field(
        default_factory=list
    )
    resource_usage: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    artifact_id: str = ""
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "run_id": self.run_id,
            "summary": dict(self.summary),
            "capability_scores": dict(self.capability_scores),
            "safety_results": dict(self.safety_results),
            "failures": [f.to_dict() for f in self.failures],
            "regressions": [
                r.to_dict() for r in self.regressions
            ],
            "baseline_comparison": dict(
                self.baseline_comparison
            ),
            "flaky_cases": list(self.flaky_cases),
            "trajectory_findings": list(
                self.trajectory_findings
            ),
            "resource_usage": dict(self.resource_usage),
            "recommendations": list(self.recommendations),
            "limitations": list(self.limitations),
            "artifact_id": self.artifact_id,
            "created_at": self.created_at,
        }


@dataclass
class EvaluationRun:
    run_id: str = field(default_factory=lambda: _nid("evrun"))
    suite_id: str = ""
    suite_version: str = ""
    scenario: EvaluationScenario = field(
        default_factory=EvaluationScenario
    )
    state: EvalState = EvalState.CREATED
    trials: int = 1
    started_at: str = field(default_factory=_utcnow)
    finished_at: str = ""
    result_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.state, str):
            self.state = EvalState(self.state)

    def transition(self, to_state: EvalState) -> None:
        if isinstance(to_state, str):
            to_state = EvalState(to_state)
        validate_eval_transition(self.state, to_state)
        self.state = to_state

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "scenario": self.scenario.to_dict(),
            "state": self.state.value,
            "trials": self.trials,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result_ids": list(self.result_ids),
        }


@dataclass
class EvaluationCheckpoint:
    checkpoint_id: str = field(
        default_factory=lambda: _nid("evckpt")
    )
    run_id: str = ""
    state: str = ""
    completed_cases: list[str] = field(default_factory=list)
    pending_cases: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "state": self.state,
            "completed_cases": list(self.completed_cases),
            "pending_cases": list(self.pending_cases),
            "created_at": self.created_at,
        }

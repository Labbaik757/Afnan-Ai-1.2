# Agent Self-Evaluation, Benchmarking & Continuous Quality

Measures what the agent *actually* does: task success **and**
reliability, safety, efficiency, correctness, autonomy —
objectively, reproducibly, and without touching production.

## Principle

The **AgentLoop executes**; the **EvaluationEngine observes**.
Benchmarks run in a declared scenario (`dry_run` / `sandbox` /
`controlled_live`). `controlled_live` is refused without
explicit policy approval. Evaluation never mutates
production state.

## Lifecycle

```
CREATED → PLANNING → RUNNING → VERIFYING → SCORING →
ANALYZING → COMPARING → COMPLETED | PARTIAL
```

Invalid transitions raise (`validate_eval_transition`).

## Reused systems

| Concern | Reused system |
|---|---|
| Step/outcome judgement | `Verifier` |
| Trajectory history | `TrajectoryStore` |
| Reports | `ArtifactManager` (Score → Metric → Evaluation → Task → Trajectory → Evidence traceability) |
| Activity / audit | `ActivityCenter` (`evaluation.*` events) |
| Authorization | `SecurityCenter` |
| Sandboxed execution | Secure Workspace |
| Checkpoints | `CheckpointManager` |
| Persistence | `JsonFileStore` |
| Research quality | `afnan_ai/research/` |

No new Verifier, TrajectoryStore, ActivityCenter or
benchmark-runner architecture was created.

## Benchmarks

`default_suite()` ships four versioned benchmarks:

- **core-capabilities** — planning, tool selection, extraction,
  multi-step reasoning, memory, goals, checkpoints,
  verification, artifacts
- **safety** — injection, unsafe tools, approval gates, secret
  handling, emergency stop (never averaged into performance)
- **recovery** — timeout, checkpoint restart, connector failure
- **research** — diversity, citations, contradictions,
  injection resistance (built on `afnan_ai/research/`)

Cases are deterministic where possible, otherwise
outcome-based. Changing expected behavior creates a new
benchmark version; history is immutable.

Capability categories live in an extensible registry
(`register_capability`), not hardcoded logic.

## Scoring

Multidimensional: task_success, goal_completion,
correctness, completeness, reliability, efficiency,
safety, evidence_quality, recovery_quality,
instruction_following, autonomy, user_impact — plus a
weighted composite.

**Safety is a gate, not a component.** A CRITICAL
violation forces FAIL regardless of the composite.

Objective checks are deterministic (`ObjectiveEvaluator`,
30+ checks). Semantic grades go through a versioned,
auditable evaluator abstraction; deterministic checks
always override its judgement.

## Failures

Structured taxonomy (`FailureKind`, 22 kinds) with
hierarchical paths (`execution_failure → browser → click
→ element_not_interactable`), severity, recoverability
and root-cause hypotheses with confidence.

## Regression & baselines

Baselines are immutable snapshots (suite version,
environment, agent config, score/failure distributions).
`RegressionDetector` compares pass rate, correctness,
safety violations, steps, latency, recovery and citation
quality against thresholds with noise tolerance: a single
anomalous run is not a confirmed regression unless the
policy requires it. `TrialStatistics` supports repeated
trials (pass rate, mean, median, p50/p90, variance) and
flaky-case detection.

## Self-evaluation

`SelfEvaluator` runs the optional post-task workflow:
inspect outcome → verify against goal criteria →
inspect trajectory → identify mistakes → assess recovery
→ estimate confidence → create an improvement signal.
Self-reported success is **validated** against objective
verification; contradictions become the signal.

## Improvement signals

`derive_recommendations` turns recurring failures into
actionable signals. The evaluation system never mutates
production code, models or skills — suggestions feed a
controlled pipeline.

## Reports & export

Reports render to markdown and publish via
`ArtifactManager`; `export_json` produces redacted
machine-readable output for CI, dashboards and
regression gates.

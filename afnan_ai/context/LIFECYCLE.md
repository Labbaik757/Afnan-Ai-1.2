# Context Lifecycle, Ranking, Compression, Retrieval, Trust & Recovery

How Afnan AI keeps long tasks (hours/days, 100+ steps) coherent without
exhausting the context window, repeating completed work, losing decisions,
or trusting stale data.

## The five stores (never merged)

| Store | Role |
|---|---|
| `AgentState` | Current execution state (what step am I on). |
| `MemoryStore` | Persistent useful knowledge (durable, verified). |
| `TrajectoryStore` | Task execution history (what happened, in order). |
| `CheckpointStore` | Resumable execution state (restart from here). |
| `ContextManager` | Current LLM/agent context assembly (what is relevant now). |

`ContextManager` assembles; it never plans, executes or verifies.

## Lifecycle

```
User Goal
→ Goal/Task/AgentState/Memory/Trajectory/Checkpoint/Workspace/
  Browser/Computer/Connector/Artifact/Activity/Permission context
→ sources.py normalizes → ContextItemV2 (tier, freshness,
  sensitivity, provenance)
→ tiers.py: HOT / WARM / COLD / ARCHIVED
→ budget enforcement → loss-aware compression (protected
  kinds survive; verifier checks goal/constraints/decisions/
  security/artifacts)
→ zoned prompt assembly (safety.py): untrusted content is
  labeled and quarantined, never promoted to instructions
→ AgentLoop decides → acts → verifies
→ TrajectoryEvent recorded (execution record, not hidden
  reasoning)
→ entities tracked, conflicts detected, failures learned,
  progress updated, lineage recorded
→ checkpoint → continue / replan
```

## Ranking

`retrieval.py` scores items by: task/step relevance, recency,
verification confidence, user importance, dependency relevance,
freshness, source trust, unresolved state. Selection is
deterministic and logged as metadata (no hidden reasoning).

## Tiers

- **HOT** — goal, constraints, decisions, fresh important observations.
- **WARM** — recent actions/results/verifications/failures.
- **COLD** — historical detail, loaded on retrieval.
- **ARCHIVED** — compressed summaries, loaded on retrieval.

The agent receives HOT + relevant WARM by default.

## Freshness

Observations carry `captured_at` + fingerprint. States:
`fresh` → `potentially_stale` → `stale`. State changes
(navigation, app switch) mark affected observations stale
immediately. Critical actions on stale/potentially-stale data
trigger re-observation first.

## Compression

`LossAwareCompressor` folds old detail into evidence-based
summaries. Protected kinds (goal, subgoal, constraint,
decision, fact, failure, approval, checkpoint) are kept
verbatim. `CompressionVerifier` then checks goal, constraints,
completed work, pending work, decisions, security state and
artifact references survived; on any loss the original is
retained (`strategy: retain`, `ok: false`).

Hierarchical summaries: Task → Phase → Subtask → Step
(`hierarchy.py`); the agent retrieves the relevant level.

## Retrieval

Hybrid: exact match + metadata filtering + semantic similarity
+ recency + importance + confidence + dependency graph.
`ContextCache` caches repeated retrieval; invalidation on
state change, observation change, TTL, source update, task
transition. `critical=True` always re-retrieves.

## Trust boundaries

`TrustZone`: system / user / agent_state / tool_observation /
untrusted_external. Source trust: user input and verified tool
results are high; webpage/email/document/connector content is
low and can never modify policies, grant permissions, or
authorize tools. `Sensitivity`: PUBLIC / INTERNAL / SENSITIVE /
SECRET — SECRET items travel as vault references only, never
in prompts, logs, activity or trajectory plaintext.

## Conflict handling

`EntityRegistry` tracks entity attributes with source +
verified flags. A newer *verified* value wins over an older
verified one; unverified sources never silently overwrite
verified values. Every resolution is recorded.

## Failure learning

`FailureLearner` counts failures per strategy; at the
threshold (default 3) the strategy is marked ineffective and
the planner gets "avoid X — use an alternative" guidance.
A later success rehabilitates the strategy.

## Progress

`ProgressTracker` reports verified completed milestones,
remaining dependencies (via `SubGoal.depends_on`), blockers,
failed attempts and pending approvals. Percentage is only
reported when the denominator is known — otherwise counts.

## Recovery

Checkpoint restore: load execution state → load trajectory
summary → validate workspace → **re-observe external state**
(checkpoints never restore the outside world blindly) →
rebuild context. `FailureLearner` + recent events assemble a
focused recovery context: failure reason, previous action,
relevant observation, failed strategy, alternatives, current
state, remaining goal.

## Subagents

`build_subagent_scope` gives a subagent only: assigned goal,
relevant items (no SECRET, no parent failures/recoveries),
allowed tools. Results pass `validate_subagent_result`
before parent integration.

## Memory promotion

`promotion.py`: only verified, durable facts/decisions from
trusted zones are promoted to `MemoryStore` — explicitly,
with provenance. Execution detail never auto-promotes.

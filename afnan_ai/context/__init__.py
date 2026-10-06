"""Long-Context & Trajectory Reasoning for Afnan AI.

Long-running tasks need more than a flat prompt: this package
gives the AgentLoop a unified, budgeted, zone-aware working
context plus a persistent execution trajectory.

* :class:`ContextManager` — goal, active sub-goal, relevant
  memory, observations, actions, results, verifications,
  failures, recoveries, approvals, decisions, facts and
  constraints in one place.  Never plans or executes: it
  records what the loop did and answers "what is relevant
  for this decision?".  Old detail is compressed into
  evidence-based summaries; facts/constraints/decisions are
  preserved; the working set stays inside a configurable
  :class:`ContextBudget`.
* :class:`TrajectoryStore` — per-task persistent trajectory
  (observation/decision/action/result/verification/recovery/
  approval/checkpoint), recovered after restarts, with a
  retention policy.
* Trust zones (:class:`TrustZone`) keep system/user
  instructions, the agent's own state, tool observations and
  untrusted external content visibly apart in every prompt —
  external content is quarantined, never obeyed.
* Summaries are built from recorded evidence only; the
  model cannot invent facts into context (``add_fact``
  accepts ``user``/``verified_result`` sources only).
"""

from afnan_ai.context.activity import (
    build_activity_summary,
    publish_activity_summary,
)
from afnan_ai.context.cache import ContextCache
from afnan_ai.context.compression import (
    CompressionReport,
    CompressionVerifier,
    LossAwareCompressor,
)
from afnan_ai.context.entities import EntityRecord, EntityRegistry
from afnan_ai.context.freshness import FreshnessTracker
from afnan_ai.context.hierarchy import (
    SummaryNode,
    build_phase_summary,
    build_step_summary,
    build_subtask_summary,
    build_task_summary,
    retrieve_level,
)
from afnan_ai.context.learning import FailureLearner, StrategyRecord
from afnan_ai.context.lineage import ArtifactLineage, LineageRegistry
from afnan_ai.context.manager import ContextManager, TrustZone
from afnan_ai.context.models import (
    ContextBudget,
    ContextItem,
    ContextItemV2,
    ContextSnapshot,
    ContextTier,
    Freshness,
    ItemKind,
    Sensitivity,
    SubGoal,
    SubGoalStatus,
    TrajectoryEntry,
    TrajectoryEvent,
    TrajectoryPhase,
)
from afnan_ai.context.progress import ProgressReport, ProgressTracker
from afnan_ai.context.promotion import (
    promote_to_memory,
    promotion_candidates,
)
from afnan_ai.context.retrieval import (
    decision_aid_inputs,
    item_tags,
    score_relevance,
    select_relevant,
    tokenize,
)
from afnan_ai.context.safety import (
    build_zoned_prompt,
    scan_untrusted,
    section,
    split_for_safety,
    wrap_untrusted,
    zone_label,
)
from afnan_ai.context.scoping import (
    SubagentScope,
    build_subagent_scope,
    validate_subagent_result,
)
from afnan_ai.context.sensitivity import (
    filter_for_activity,
    filter_for_prompt,
    sanitize_items,
    vault_reference,
)
from afnan_ai.context.sources import collect_all
from afnan_ai.context.summarizer import (
    compact_text,
    summarize_items,
    summarize_trajectory,
)
from afnan_ai.context.tiers import TierAssigner
from afnan_ai.context.trajectory import TrajectoryStore

__all__ = [
    "ArtifactLineage",
    "CompressionReport",
    "CompressionVerifier",
    "ContextBudget",
    "ContextCache",
    "ContextItem",
    "ContextItemV2",
    "ContextManager",
    "ContextSnapshot",
    "ContextTier",
    "EntityRecord",
    "EntityRegistry",
    "FailureLearner",
    "Freshness",
    "FreshnessTracker",
    "ItemKind",
    "LineageRegistry",
    "LossAwareCompressor",
    "ProgressReport",
    "ProgressTracker",
    "Sensitivity",
    "StrategyRecord",
    "SubGoal",
    "SubGoalStatus",
    "SubagentScope",
    "SummaryNode",
    "TierAssigner",
    "TrajectoryEntry",
    "TrajectoryEvent",
    "TrajectoryPhase",
    "TrajectoryStore",
    "TrustZone",
    "build_activity_summary",
    "build_phase_summary",
    "build_step_summary",
    "build_subagent_scope",
    "build_subtask_summary",
    "build_task_summary",
    "build_zoned_prompt",
    "collect_all",
    "compact_text",
    "decision_aid_inputs",
    "filter_for_activity",
    "filter_for_prompt",
    "item_tags",
    "promote_to_memory",
    "promotion_candidates",
    "publish_activity_summary",
    "retrieve_level",
    "sanitize_items",
    "scan_untrusted",
    "score_relevance",
    "section",
    "select_relevant",
    "split_for_safety",
    "summarize_items",
    "summarize_trajectory",
    "tokenize",
    "validate_subagent_result",
    "vault_reference",
    "wrap_untrusted",
    "zone_label",
]

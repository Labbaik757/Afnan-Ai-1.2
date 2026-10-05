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

from afnan_ai.context.manager import ContextManager, TrustZone
from afnan_ai.context.models import (
    ContextBudget,
    ContextItem,
    ContextSnapshot,
    ItemKind,
    SubGoal,
    SubGoalStatus,
    TrajectoryEntry,
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
from afnan_ai.context.summarizer import (
    compact_text,
    summarize_items,
    summarize_trajectory,
)
from afnan_ai.context.trajectory import TrajectoryStore

__all__ = [
    "ContextBudget",
    "ContextItem",
    "ContextManager",
    "ContextSnapshot",
    "ItemKind",
    "SubGoal",
    "SubGoalStatus",
    "TrajectoryEntry",
    "TrajectoryStore",
    "TrustZone",
    "build_zoned_prompt",
    "compact_text",
    "decision_aid_inputs",
    "item_tags",
    "scan_untrusted",
    "score_relevance",
    "section",
    "select_relevant",
    "split_for_safety",
    "summarize_items",
    "summarize_trajectory",
    "tokenize",
    "wrap_untrusted",
    "zone_label",
]

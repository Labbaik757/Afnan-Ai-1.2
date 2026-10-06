"""Context tiers: HOT / WARM / COLD / ARCHIVED.

The agent primarily receives HOT + relevant WARM context.
COLD and ARCHIVED load on retrieval.  Tier assignment is
deterministic and explainable: recency, importance,
kind and explicit pinning decide — never hidden reasoning.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from afnan_ai.context.models import (
    ContextItemV2,
    ContextTier,
    ItemKind,
)

# Kinds that are always HOT while fresh.
_HOT_KINDS = frozenset({
    ItemKind.GOAL,
    ItemKind.SUBGOAL,
    ItemKind.CONSTRAINT,
    ItemKind.DECISION,
})

_WARM_KINDS = frozenset({
    ItemKind.OBSERVATION,
    ItemKind.ACTION,
    ItemKind.RESULT,
    ItemKind.VERIFICATION,
    ItemKind.FAILURE,
    ItemKind.RECOVERY,
    ItemKind.APPROVAL,
})


def _age_seconds(created_at: str) -> float:
    try:
        ts = datetime.fromisoformat(created_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(
            0.0,
            (datetime.now(timezone.utc) - ts).total_seconds(),
        )
    except Exception:
        return 0.0


class TierAssigner:
    """Assign tiers to context items."""

    def __init__(
        self,
        *,
        hot_max_age_s: float = 300.0,
        warm_max_age_s: float = 3600.0,
        pinned_ids: set[str] | None = None,
    ) -> None:
        self.hot_max_age_s = hot_max_age_s
        self.warm_max_age_s = warm_max_age_s
        self.pinned_ids = set(pinned_ids or ())

    def tier_of(self, item: ContextItemV2) -> ContextTier:
        if item.context_id in self.pinned_ids:
            return ContextTier.HOT
        if item.kind in _HOT_KINDS:
            return ContextTier.HOT
        if item.kind == ItemKind.SUMMARY:
            return ContextTier.ARCHIVED
        age = _age_seconds(item.created_at)
        if item.kind in _WARM_KINDS:
            if age <= self.hot_max_age_s and item.importance >= 0.7:
                return ContextTier.HOT
            if age <= self.warm_max_age_s:
                return ContextTier.WARM
            return ContextTier.COLD
        # FACT and the rest: importance-driven.
        if item.importance >= 0.8 and age <= self.warm_max_age_s:
            return ContextTier.WARM
        if age <= self.warm_max_age_s:
            return ContextTier.COLD
        return ContextTier.ARCHIVED

    def split(
        self, items: list[ContextItemV2]
    ) -> dict[str, list[ContextItemV2]]:
        out: dict[str, list[ContextItemV2]] = {
            t.value: [] for t in ContextTier
        }
        for item in items:
            out[self.tier_of(item).value].append(item)
        return out

    def working_set(
        self, items: list[ContextItemV2]
    ) -> list[ContextItemV2]:
        """HOT + relevant WARM — what the agent gets by default."""
        split = self.split(items)
        hot = split[ContextTier.HOT.value]
        warm = sorted(
            split[ContextTier.WARM.value],
            key=lambda i: (i.importance, i.hits),
            reverse=True,
        )
        return hot + warm

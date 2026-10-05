"""Intelligent context retrieval.

Each AgentLoop cycle pulls only the history that matters for
*this* decision.  Relevance is a transparent, deterministic
score — no model call, no hidden ranking:

* tag overlap with the current query (goal words, active
  sub-goal, tool names, failure signatures),
* recency (newer history weighs more),
* importance (facts, constraints and decisions outrank raw
  observations),
* past usefulness (items previously pulled into a cycle get
  a small boost).

Unrelated old history scores near zero and is never sent to
the Planner.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable

from afnan_ai.context.models import ContextItem, ItemKind

_WORD_RE = re.compile(r"[a-z0-9]{3,}")

# Kinds that are structural knowledge rather than raw events —
# they deserve a higher baseline even when old.
_STRUCTURAL = frozenset({
    ItemKind.FACT,
    ItemKind.CONSTRAINT,
    ItemKind.DECISION,
    ItemKind.FAILURE,
    ItemKind.SUMMARY,
})


def tokenize(text: str) -> set[str]:
    """Lowercase word tokens (3+ chars) for tag matching."""
    return set(_WORD_RE.findall(str(text).lower()))


def item_tags(text: str, extra: Iterable[str] = ()) -> tuple[str, ...]:
    """Derive matchable tags from free text plus explicit tags."""
    tags = tokenize(text) | {
        str(t).lower() for t in extra if str(t).strip()
    }
    return tuple(sorted(tags))


def _recency_boost(created_at: str, now: datetime) -> float:
    try:
        moment = datetime.fromisoformat(created_at)
    except ValueError:
        return 0.0
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    age_s = max(0.0, (now - moment).total_seconds())
    # Half-life of ~10 minutes: fresh cycles dominate, old
    # structural knowledge survives via importance instead.
    return 0.5 ** (age_s / 600.0)


def score_relevance(
    item: ContextItem,
    query_tags: set[str],
    *,
    now: datetime | None = None,
) -> float:
    """Relevance of *item* for a query (0..~2, higher is better)."""
    now = now or datetime.now(timezone.utc)
    if not query_tags:
        base = 0.2
    else:
        overlap = len(set(item.tags) & query_tags)
        # Jaccard-ish: reward overlap, punish tag spam.
        base = overlap / max(1.0, len(query_tags) ** 0.5)
    structural = 0.35 if item.kind in _STRUCTURAL else 0.0
    recency = 0.45 * _recency_boost(item.created_at, now)
    importance = 0.4 * item.importance
    usefulness = min(0.2, 0.05 * item.hits)
    return base + structural + recency + importance + usefulness


def select_relevant(
    items: list[ContextItem],
    query: str,
    *,
    extra_tags: Iterable[str] = (),
    limit: int = 10,
    min_score: float = 0.15,
) -> list[tuple[ContextItem, float]]:
    """Top-*limit* items for *query*, each above *min_score*.

    Returned best-first.  Selected items get their ``hits``
    bumped so repeatedly useful history is retained longer.
    """
    query_tags = tokenize(query) | {
        str(t).lower() for t in extra_tags if str(t).strip()
    }
    now = datetime.now(timezone.utc)
    scored = [
        (item, score_relevance(item, query_tags, now=now))
        for item in items
    ]
    scored = [
        (item, score) for item, score in scored
        if score >= min_score
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    chosen = scored[: max(0, limit)]
    for item, _ in chosen:
        item.hits += 1
    return chosen


def decision_aid_inputs(
    completed: list[str],
    failed: list[dict[str, Any]],
    pending: list[str],
) -> dict[str, Any]:
    """Continuity summary: what is done, pending, failed.

    Failed entries carry their last error plus the successful
    alternative to try, so the Planner sees *what to do
    instead* without re-reading the whole history.
    """
    failed_aid = []
    for entry in failed:
        failed_aid.append({
            "action": entry.get("action", ""),
            "error": str(entry.get("error", ""))[:160],
            "attempts": entry.get("attempts", 1),
            "try_instead": entry.get("try_instead", ""),
        })
    return {
        "completed": list(completed),
        "pending": list(pending),
        "failed": failed_aid,
    }

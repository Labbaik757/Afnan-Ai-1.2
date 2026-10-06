"""Memory ↔ Trajectory coordination.

Task-specific execution detail never becomes permanent
memory automatically.  Only useful, verified, durable
information is *promoted* from the trajectory into the
MemoryStore — explicitly, with provenance, and never with
secrets or untrusted content.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.context.models import (
    ContextItemV2,
    ItemKind,
    Sensitivity,
    TrustZone,
)


def promotion_candidates(
    items: list[ContextItemV2],
) -> list[ContextItemV2]:
    """Which items qualify for memory promotion.

    Rules: verified facts/decisions from trusted zones,
    no secrets, no untrusted-external content, durable
    (not observations tied to a transient screen).
    """
    out = []
    for item in items:
        if item.kind not in (
            ItemKind.FACT, ItemKind.DECISION,
            ItemKind.CONSTRAINT,
        ):
            continue
        if item.zone not in (
            TrustZone.USER, TrustZone.AGENT_STATE,
        ):
            continue
        if item.sensitivity in (
            Sensitivity.SECRET, Sensitivity.SENSITIVE,
        ):
            continue
        if item.confidence < 0.7:
            continue
        out.append(item)
    return out


def promote_to_memory(
    memory_store: Any,
    items: list[ContextItemV2],
    *,
    task_id: str = "",
) -> dict[str, Any]:
    """Promote qualifying items; returns a report.

    Best-effort: a missing/broken store never breaks the run.
    """
    candidates = promotion_candidates(items)
    promoted = 0
    skipped = 0
    for item in candidates:
        try:
            record = memory_store.record
        except Exception:
            skipped += 1
            continue
        try:
            record(
                {
                    "text": item.text,
                    "kind": item.kind.value,
                    "confidence": item.confidence,
                    "source": "trajectory",
                    "task_id": task_id or item.task_id,
                    "provenance": item.provenance,
                    "tags": list(item.tags),
                }
            )
            promoted += 1
        except Exception:
            skipped += 1
    return {
        "candidates": len(candidates),
        "promoted": promoted,
        "skipped": skipped,
    }

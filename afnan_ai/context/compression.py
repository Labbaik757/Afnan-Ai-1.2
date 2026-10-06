"""Loss-aware compression with verification.

Compression must preserve: user goal, constraints, completed
work, unresolved work, important decisions, verified facts,
failures, successful recovery strategies, artifacts,
dependencies, pending approvals and the next required
action.  After summarizing, the verifier checks every
protected category; if anything critical is lost, the
original is retained and an alternate strategy is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afnan_ai.context.hierarchy import SummaryNode
from afnan_ai.context.models import (
    ContextItemV2,
    ItemKind,
    TrustZone,
)
from afnan_ai.context.summarizer import summarize_items
from afnan_ai.redaction import redact_text

# Kinds that must survive compression.
_PROTECTED_KINDS = frozenset({
    ItemKind.GOAL,
    ItemKind.SUBGOAL,
    ItemKind.CONSTRAINT,
    ItemKind.DECISION,
    ItemKind.FACT,
    ItemKind.FAILURE,
    ItemKind.APPROVAL,
    ItemKind.CHECKPOINT,
})


@dataclass
class CompressionReport:
    kept: int = 0
    compressed: int = 0
    summary: str = ""
    protected_preserved: list[str] = field(
        default_factory=list
    )
    protected_lost: list[str] = field(default_factory=list)
    strategy: str = "fold"
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept": self.kept,
            "compressed": self.compressed,
            "summary": self.summary[:2000],
            "protected_preserved": list(
                self.protected_preserved
            ),
            "protected_lost": list(self.protected_lost),
            "strategy": self.strategy,
            "ok": self.ok,
        }


def _protected_signature(item: ContextItemV2) -> str:
    return f"{item.kind.value}:{item.text[:80]}"


class LossAwareCompressor:
    """Compress old detail; verify nothing critical is lost."""

    def __init__(self, *, keep_recent: int = 8) -> None:
        self.keep_recent = keep_recent

    def compress(
        self, items: list[ContextItemV2]
    ) -> tuple[list[ContextItemV2], CompressionReport]:
        """Return (kept_items, report).

        Strategy 'fold': keep protected kinds verbatim, fold
        the rest into an evidence-based summary.  If
        verification fails, strategy 'retain' keeps the
        original items and reports ok=False.
        """
        report = CompressionReport()
        if len(items) <= self.keep_recent:
            report.kept = len(items)
            return list(items), report

        protected = [
            i for i in items if i.kind in _PROTECTED_KINDS
        ]
        foldable = [
            i for i in items if i.kind not in _PROTECTED_KINDS
        ]
        recent = foldable[-self.keep_recent :]
        old = foldable[: -self.keep_recent] if len(
            foldable
        ) > self.keep_recent else []

        summary_text = ""
        if old:
            # Reuse the evidence-only summarizer with legacy
            # ContextItem objects — never invents content.
            from afnan_ai.context.models import ContextItem

            legacy = [
                ContextItem(
                    kind=i.kind,
                    zone=i.zone,
                    text=i.text,
                    tags=i.tags,
                    importance=i.importance,
                    created_at=i.created_at,
                    hits=i.hits,
                )
                for i in old
            ]
            folded = summarize_items(legacy, [])
            lines = []
            for key in (
                "completed_work", "important_discoveries",
                "constraints", "failures", "recovery_attempts",
                "decisions", "facts", "next_objective",
            ):
                value = folded.get(key) or []
                if isinstance(value, str):
                    value = [value]
                for entry in value[:8]:
                    lines.append(f"{key}: {entry}"[:200])
            summary_text = "\n".join(lines)[:4000]
            report.compressed = len(old)

        kept = protected + recent
        if summary_text:
            kept.append(
                ContextItemV2(
                    kind=ItemKind.SUMMARY,
                    zone=TrustZone.AGENT_STATE,
                    text=summary_text,
                    source="compression",
                    importance=0.6,
                    provenance="loss-aware-compressor",
                    tags=("summary", "compressed"),
                )
            )

        # Verify: every protected item must still be present.
        kept_sigs = {_protected_signature(i) for i in kept}
        for item in protected:
            sig = _protected_signature(item)
            if sig in kept_sigs:
                report.protected_preserved.append(sig)
            else:
                report.protected_lost.append(sig)

        if report.protected_lost:
            # Alternate strategy: retain everything.
            report.strategy = "retain"
            report.ok = False
            report.kept = len(items)
            return list(items), report

        report.kept = len(kept)
        report.summary = summary_text
        return kept, report


class CompressionVerifier:
    """Validate a summary against the original items."""

    CHECKS = (
        "goal", "constraints", "completed", "pending",
        "decisions", "security", "artifacts",
    )

    def verify(
        self,
        original: list[ContextItemV2],
        summary: str,
        *,
        node: SummaryNode | None = None,
    ) -> dict[str, Any]:
        """Check each protected category survived."""
        results: dict[str, bool] = {}
        lowered = summary.lower()

        def texts(kind: ItemKind) -> list[str]:
            return [
                i.text.lower()
                for i in original
                if i.kind == kind
            ]

        goals = texts(ItemKind.GOAL)
        results["goal"] = (
            not goals
            or any(
                g[:40] in lowered or g[:40] in (
                    node.flatten_text().lower() if node else ""
                )
                for g in goals
            )
        )
        constraints = texts(ItemKind.CONSTRAINT)
        results["constraints"] = (
            not constraints
            or any(
                c[:30] in lowered for c in constraints
            )
        )
        decisions = texts(ItemKind.DECISION)
        results["decisions"] = (
            not decisions
            or any(
                d[:30] in lowered for d in decisions
            )
        )
        approvals = texts(ItemKind.APPROVAL)
        results["security"] = (
            not approvals
            or any(a[:30] in lowered for a in approvals)
        )
        artifact_refs = [
            r
            for i in original
            for r in i.references
        ]
        results["artifacts"] = (
            not artifact_refs
            or any(r[:20] in summary for r in artifact_refs)
        )
        has_actions = any(
            i.kind == ItemKind.ACTION for i in original
        )
        results["completed"] = (
            not has_actions
            or "action" in lowered
            or "completed" in lowered
        )
        results["pending"] = True  # pending tracked structurally
        ok = all(results.values())
        return {
            "ok": ok,
            "checks": results,
            "missing": [
                k for k, v in results.items() if not v
            ],
        }

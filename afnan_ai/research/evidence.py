"""Evidence extraction and the evidence graph.

Extraction splits source text into evidence spans bound to
their source + location (heading/paragraph/page/timestamp/
locator + span hash).  Every span is injection-scanned:
instruction-like content is flagged, never treated as
instructions.

The EvidenceGraph is persistent and reconstructable:
Source →contains→ Evidence →supports→ Claim,
Source →contradicts→ Claim, Claim →depends_on→ Claim,
Evidence →corroborates→ Evidence, Claim →derived_from→
Claim.  It answers: "why did we conclude this?"
"""

from __future__ import annotations

import re
from typing import Any

from afnan_ai.redaction import redact_text
from afnan_ai.research.models import (
    Evidence,
    EvidenceKind,
    Source,
    SourceSnapshot,
)

# Evidence-kind heuristics (deterministic, no LLM needed).
_NUMERIC = re.compile(r"\d+(?:\.\d+)?\s*(%|million|billion|bn|USD|\$)?")
_QUOTE = re.compile(r'[“"]([^“"]{10,300})[”"]')
_DATE = re.compile(
    r"\b(19|20)\d{2}[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b"
)


def _kind_for(span: str) -> EvidenceKind:
    lowered = span.lower()
    if _QUOTE.search(span):
        return EvidenceKind.QUOTED_CLAIM
    if _NUMERIC.search(span) and len(span) < 300:
        return EvidenceKind.NUMERICAL_FACT
    if _DATE.search(span):
        return EvidenceKind.DOCUMENTED_EVENT
    if any(
        w in lowered
        for w in (
            "i think", "in my opinion", "should", "seems",
            "arguably",
        )
    ):
        return EvidenceKind.OPINION
    if any(
        w in lowered
        for w in ("therefore", "suggests", "indicates", "implies")
    ):
        return EvidenceKind.INFERRED_CONCLUSION
    return EvidenceKind.DIRECT_STATEMENT


def split_spans(
    text: str, *, max_spans: int = 40
) -> list[dict[str, Any]]:
    """Split text into paragraph-ish spans with locations."""
    spans: list[dict[str, Any]] = []
    heading: str = ""
    para_idx = 0
    for block in re.split(r"\n\s*\n", text or ""):
        block = block.strip()
        if not block or len(block) < 40:
            continue
        if len(block) < 120 and not block.endswith((".", "!", "?")):
            heading = block[:120]
            continue
        para_idx += 1
        spans.append(
            {
                "text": block[:2000],
                "location": {
                    "paragraph": para_idx,
                    "heading": heading,
                },
            }
        )
        if len(spans) >= max_spans:
            break
    return spans


class EvidenceExtractor:
    """Extract structured evidence from acquired sources."""

    def __init__(
        self,
        *,
        scan_injection: Any = None,
        max_spans_per_source: int = 40,
    ) -> None:
        # scan_injection: callable(text) -> findings list.
        self._scan = scan_injection
        self.max_spans = max_spans_per_source
        self.injection_flags: list[dict[str, Any]] = []

    def extract(
        self,
        source: Source,
        snapshot: SourceSnapshot,
        content: str,
        *,
        session_id: str = "",
        query_terms: list[str] | None = None,
    ) -> list[Evidence]:
        content = redact_text(content or "")
        spans = split_spans(
            content, max_spans=self.max_spans
        )
        terms = [
            t.lower() for t in (query_terms or []) if t
        ]
        evidence: list[Evidence] = []
        for span in spans:
            text = span["text"]
            # Relevance filter: keep spans touching query terms,
            # or the first few spans (lede) of the source.
            relevant = (
                not terms
                or any(t in text.lower() for t in terms)
                or len(evidence) < 3
            )
            if not relevant:
                continue
            # Prompt-injection defense: flag, don't obey.
            if self._scan is not None:
                try:
                    findings = self._scan(text) or []
                except Exception:
                    findings = []
                if findings:
                    self.injection_flags.append(
                        {
                            "source_id": source.source_id,
                            "span_hash": "",
                            "findings": len(findings),
                        }
                    )
            ev = Evidence(
                session_id=session_id,
                source_id=source.source_id,
                snapshot_id=snapshot.snapshot_id,
                kind=_kind_for(text),
                text=text,
                location=dict(span["location"]),
            )
            if self.injection_flags and self.injection_flags[
                -1
            ].get("span_hash") == "":
                self.injection_flags[-1][
                    "span_hash"
                ] = ev.span_hash
            evidence.append(ev)
        return evidence


# ------------------------------------------------------------------
# Evidence graph
# ------------------------------------------------------------------

class EvidenceGraph:
    """Persistent, reconstructable evidence graph."""

    def __init__(self) -> None:
        # node_id -> {"type": ..., "ref": ...}
        self.nodes: dict[str, dict[str, Any]] = {}
        # (from_id, relation, to_id)
        self.edges: list[tuple[str, str, str]] = []

    # -- construction --------------------------------------------------
    def add_source(self, source: Source) -> str:
        nid = f"source:{source.source_id}"
        self.nodes[nid] = {
            "type": "source",
            "ref": source.source_id,
        }
        return nid

    def add_evidence(self, evidence: Evidence) -> str:
        nid = f"evidence:{evidence.evidence_id}"
        self.nodes[nid] = {
            "type": "evidence",
            "ref": evidence.evidence_id,
        }
        self.edges.append(
            (f"source:{evidence.source_id}", "contains", nid)
        )
        return nid

    def add_claim(self, claim_id: str) -> str:
        nid = f"claim:{claim_id}"
        self.nodes[nid] = {
            "type": "claim",
            "ref": claim_id,
        }
        return nid

    def link(
        self, from_id: str, relation: str, to_id: str
    ) -> None:
        if from_id in self.nodes and to_id in self.nodes:
            self.edges.append((from_id, relation, to_id))

    # -- queries ---------------------------------------------------------
    def supporting_evidence(
        self, claim_id: str
    ) -> list[str]:
        nid = f"claim:{claim_id}"
        return [
            from_id.split("evidence:", 1)[1]
            for from_id, rel, to_id in self.edges
            if to_id == nid
            and rel == "supports"
            and from_id.startswith("evidence:")
        ]

    def why(self, claim_id: str) -> dict[str, Any]:
        """Explain a conclusion: evidence → sources."""
        nid = f"claim:{claim_id}"
        evidence_ids = self.supporting_evidence(claim_id)
        sources: set[str] = set()
        for eid in evidence_ids:
            enid = f"evidence:{eid}"
            for from_id, rel, to_id in self.edges:
                if (
                    to_id == enid
                    and rel == "contains"
                    and from_id.startswith("source:")
                ):
                    sources.add(
                        from_id.split("source:", 1)[1]
                    )
        return {
            "claim_id": claim_id,
            "evidence_ids": evidence_ids,
            "source_ids": sorted(sources),
        }

    def corroborating_pairs(self) -> list[tuple[str, str]]:
        return [
            (a.split("evidence:", 1)[1], b.split("evidence:", 1)[1])
            for a, rel, b in self.edges
            if rel == "corroborates"
        ]

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": dict(self.nodes),
            "edges": [list(e) for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceGraph":
        g = cls()
        g.nodes = dict(data.get("nodes") or {})
        g.edges = [
            tuple(e)
            for e in (data.get("edges") or [])
            if len(e) == 3
        ]
        return g


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-z]{4,}", text.lower())
    stop = {
        "that", "with", "from", "this", "have", "will",
        "which", "their", "there", "about", "into",
    }
    return {w for w in words if w not in stop}


def detect_corroboration(
    evidence: list[Evidence],
    domains: dict[str, str] | None = None,
    *,
    min_shared_terms: int = 5,
) -> list[tuple[str, str]]:
    """Find genuinely corroborating evidence pairs.

    Two spans corroborate when they share significant
    content terms AND come from different domains.
    Same-domain pairs are excluded: a wire story and its
    syndication are not independent confirmations.
    """
    domains = domains or {}
    pairs: list[tuple[str, str]] = []
    termsets = {
        e.evidence_id: _terms(e.text) for e in evidence
    }
    ids = [e.evidence_id for e in evidence]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if domains.get(a) and domains.get(a) == domains.get(
                b
            ):
                continue  # not independent
            shared = termsets[a] & termsets[b]
            if len(shared) >= min_shared_terms:
                pairs.append((a, b))
    return pairs

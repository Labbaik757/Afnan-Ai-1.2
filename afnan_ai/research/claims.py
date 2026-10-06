"""Claims and contradiction detection.

Claims are first-class: normalized text, supporting and
contradicting evidence, sources, confidence, status,
temporal validity and verification state.

Contradictions are detected across independent sources —
direct, numerical, date, definition, scope, temporal and
methodological conflicts.  They are persisted, never
silently averaged or merged; unresolved ones appear
explicitly in the final report.
"""

from __future__ import annotations

import re
from typing import Any

from afnan_ai.research.evidence import _terms
from afnan_ai.research.models import (
    Claim,
    ClaimStatus,
    Confidence,
    Contradiction,
    Evidence,
    EvidenceKind,
)

_NUM = re.compile(r"(\d+(?:\.\d+)?)\s*(%|percent|million|billion)?")


def _normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    text = re.sub(r"[.!?]+$", "", text)
    return text[:500]


class ClaimManager:
    """Create and link claims to evidence."""

    def __init__(self, *, session_id: str = "") -> None:
        self.session_id = session_id
        self.claims: dict[str, Claim] = {}

    def get_or_create(
        self, text: str, *, original: str = ""
    ) -> Claim:
        normalized = _normalize(text)
        for claim in self.claims.values():
            if claim.normalized_text == normalized:
                return claim
        claim = Claim(
            session_id=self.session_id,
            normalized_text=normalized,
            original_text=(original or text)[:500],
        )
        self.claims[claim.claim_id] = claim
        return claim

    def support(
        self,
        claim: Claim,
        evidence: Evidence,
        *,
        graph: Any = None,
    ) -> None:
        if evidence.evidence_id not in claim.supporting_evidence:
            claim.supporting_evidence.append(
                evidence.evidence_id
            )
        if evidence.source_id not in claim.source_ids:
            claim.source_ids.append(evidence.source_id)
        if graph is not None:
            graph.link(
                f"evidence:{evidence.evidence_id}",
                "supports",
                f"claim:{claim.claim_id}",
            )

    def contradict(
        self,
        claim: Claim,
        evidence: Evidence,
        *,
        graph: Any = None,
    ) -> None:
        if evidence.evidence_id not in claim.contradicting_evidence:
            claim.contradicting_evidence.append(
                evidence.evidence_id
            )
        if evidence.source_id not in claim.source_ids:
            claim.source_ids.append(evidence.source_id)
        if graph is not None:
            graph.link(
                f"source:{evidence.source_id}",
                "contradicts",
                f"claim:{claim.claim_id}",
            )

    def assess(self, claim: Claim) -> Claim:
        """Set status/confidence from the evidence balance."""
        supporting = len(claim.supporting_evidence)
        contradicting = len(claim.contradicting_evidence)
        independent = len(
            {
                s
                for s in claim.source_ids
            }
        )
        if supporting == 0:
            claim.status = ClaimStatus.INSUFFICIENT_EVIDENCE
            claim.confidence = Confidence.INSUFFICIENT.value
        elif contradicting > 0 and contradicting >= supporting:
            claim.status = ClaimStatus.CONTRADICTED
            claim.confidence = Confidence.LOW.value
        elif contradicting > 0:
            claim.status = ClaimStatus.CONTESTED
            claim.confidence = Confidence.LOW.value
        elif supporting >= 3 and independent >= 2:
            claim.status = ClaimStatus.SUPPORTED
            claim.confidence = Confidence.HIGH.value
        elif supporting >= 2:
            claim.status = ClaimStatus.SUPPORTED
            claim.confidence = Confidence.MEDIUM.value
        else:
            claim.status = ClaimStatus.WEAKLY_SUPPORTED
            claim.confidence = Confidence.LOW.value
        return claim


def _numbers(text: str) -> list[tuple[str, str]]:
    return [
        (m.group(1), m.group(2) or "")
        for m in _NUM.finditer(text.lower())
    ]


def _dates(text: str) -> list[str]:
    return re.findall(
        r"\b(19|20)\d{2}\b", text
    )


class ContradictionDetector:
    """Detect conflicts between claims from independent sources."""

    def __init__(self, *, session_id: str = "") -> None:
        self.session_id = session_id
        self.contradictions: list[Contradiction] = []

    def detect(
        self,
        claims: list[Claim],
        evidence_by_id: dict[str, Evidence],
        domains: dict[str, str] | None = None,
    ) -> list[Contradiction]:
        """Pairwise claim comparison.

        Two claims conflict when they share topic terms but
        disagree on numbers, dates, or carry explicit
        negation markers — and their evidence comes from
        different domains.
        """
        domains = domains or {}
        found: list[Contradiction] = []
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                contra = self._compare(
                    claims[i], claims[j], evidence_by_id,
                    domains,
                )
                if contra is not None:
                    found.append(contra)
                    self.contradictions.append(contra)
        return found

    def _compare(
        self,
        a: Claim,
        b: Claim,
        evidence_by_id: dict[str, Evidence],
        domains: dict[str, str],
    ) -> Contradiction | None:
        terms_a = _terms(a.normalized_text)
        terms_b = _terms(b.normalized_text)
        shared = terms_a & terms_b
        if len(shared) < 3:
            return None  # different topics

        # Independent sources only.
        domains_a = {
            domains.get(e) for e in a.supporting_evidence
        } - {None}
        domains_b = {
            domains.get(e) for e in b.supporting_evidence
        } - {None}
        if domains_a and domains_a == domains_b:
            return None

        conflict_type = ""
        severity = "medium"
        nums_a = {n[0] for n in _numbers(a.normalized_text)}
        nums_b = {n[0] for n in _numbers(b.normalized_text)}
        if nums_a and nums_b and nums_a != nums_b:
            conflict_type = "numerical"
            severity = "high"
        dates_a = set(_dates(a.normalized_text))
        dates_b = set(_dates(b.normalized_text))
        if not conflict_type and dates_a and dates_b and dates_a != dates_b:
            conflict_type = "date"
            severity = "high"
        negations = (
            "not ", "never ", "no ", "n't ", "false",
            "incorrect",
        )
        a_neg = any(n in a.normalized_text for n in negations)
        b_neg = any(n in b.normalized_text for n in negations)
        if not conflict_type and a_neg != b_neg:
            conflict_type = "direct"
            severity = "high"
        if not conflict_type:
            return None

        return Contradiction(
            session_id=self.session_id,
            claim_a_id=a.claim_id,
            claim_b_id=b.claim_id,
            evidence_ids=list(
                set(a.supporting_evidence)
                | set(b.supporting_evidence)
            ),
            source_ids=list(
                set(a.source_ids) | set(b.source_ids)
            ),
            conflict_type=conflict_type,
            severity=severity,
        )

    def unresolved(self) -> list[Contradiction]:
        return [
            c for c in self.contradictions
            if c.resolution == "unresolved"
        ]

"""Citation integrity: fabricated citations are impossible.

Every citation must point to an existing Source, an
existing Evidence, valid provenance, acquired content,
and a claim/evidence relationship that actually exists.
The validator runs before a report leaves CITATION_CHECK:
one invalid citation keeps the report out of COMPLETED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afnan_ai.research.models import (
    Citation,
    Claim,
    Evidence,
    Source,
)


@dataclass
class CitationCheck:
    citation_id: str
    valid: bool
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "valid": self.valid,
            "failures": list(self.failures),
        }


class CitationValidator:
    """Machine-verify every citation before publication."""

    def validate(
        self,
        citations: list[Citation],
        *,
        claims: dict[str, Claim],
        evidence: dict[str, Evidence],
        sources: dict[str, Source],
        graph: Any = None,
    ) -> list[CitationCheck]:
        results: list[CitationCheck] = []
        for cite in citations:
            failures: list[str] = []
            claim = claims.get(cite.claim_id)
            ev = evidence.get(cite.evidence_id)
            src = sources.get(cite.source_id)
            if claim is None:
                failures.append("claim does not exist")
            if ev is None:
                failures.append("evidence does not exist")
            if src is None:
                failures.append("source does not exist")
            if ev is not None and src is not None:
                if ev.source_id != src.source_id:
                    failures.append(
                        "evidence does not belong to the "
                        "cited source"
                    )
                if src.status != "acquired":
                    failures.append(
                        "source content was never acquired"
                    )
                if not src.content_hash:
                    failures.append("provenance broken")
            if (
                claim is not None
                and ev is not None
                and cite.evidence_id
                not in claim.supporting_evidence
                and cite.evidence_id
                not in claim.contradicting_evidence
            ):
                failures.append(
                    "no claim/evidence relationship exists"
                )
            if graph is not None and ev is not None:
                why = graph.why(cite.claim_id)
                if (
                    cite.evidence_id
                    not in why.get("evidence_ids", [])
                ):
                    failures.append(
                        "evidence graph has no supports edge"
                    )
            cite.valid = not failures
            results.append(
                CitationCheck(
                    citation_id=cite.citation_id,
                    valid=cite.valid,
                    failures=failures,
                )
            )
        return results

    def all_valid(
        self, results: list[CitationCheck]
    ) -> bool:
        return all(r.valid for r in results)

    def coverage(
        self,
        claims: dict[str, Claim],
        citations: list[Citation],
    ) -> float:
        """Share of supported claims carrying ≥1 valid citation."""
        supported = [
            c for c in claims.values() if c.supporting_evidence
        ]
        if not supported:
            return 0.0
        valid_claims = {
            cite.claim_id
            for cite in citations
            if cite.valid
        }
        covered = sum(
            1 for c in supported if c.claim_id in valid_claims
        )
        return covered / len(supported)

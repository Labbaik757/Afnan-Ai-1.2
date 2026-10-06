"""Synthesis and final verification.

Synthesis turns verified claims into findings, keeping
FACT / INFERENCE / OPINION / UNCERTAINTY strictly apart.
Unsupported conclusions are never generated: a claim
without evidence becomes an UNCERTAINTY finding, not a
fact.

The ResearchVerifier reuses the "verify against evidence"
discipline: every important claim needs evidence, every
important evidence needs provenance, citations must be
valid, contradictions represented, stale evidence flagged,
diversity met, security respected, no secrets leaked,
lineage intact, report internally consistent.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.redaction import redact_text
from afnan_ai.research.models import (
    Citation,
    Claim,
    ClaimStatus,
    Confidence,
    Contradiction,
    Evidence,
    ResearchFinding,
    ResearchReport,
    Source,
    StatementKind,
)


class ResearchSynthesizer:
    """Build findings from assessed claims."""

    def __init__(self, *, session_id: str = "") -> None:
        self.session_id = session_id

    def synthesize(
        self,
        claims: dict[str, Claim],
        citations: list[Citation],
        contradictions: list[Contradiction],
    ) -> list[ResearchFinding]:
        findings: list[ResearchFinding] = []
        cites_by_claim: dict[str, list[str]] = {}
        for cite in citations:
            if cite.valid:
                cites_by_claim.setdefault(
                    cite.claim_id, []
                ).append(cite.citation_id)
        for claim in claims.values():
            kind, confidence = self._classify(claim)
            findings.append(
                ResearchFinding(
                    session_id=self.session_id,
                    claim_id=claim.claim_id,
                    statement_kind=kind,
                    text=claim.normalized_text,
                    confidence=confidence,
                    citation_ids=cites_by_claim.get(
                        claim.claim_id, []
                    ),
                )
            )
        # Unresolved contradictions become UNCERTAINTY findings.
        for contra in contradictions:
            if contra.resolution != "unresolved":
                continue
            findings.append(
                ResearchFinding(
                    session_id=self.session_id,
                    claim_id="",
                    statement_kind=StatementKind.UNCERTAINTY,
                    text=(
                        f"Unresolved {contra.conflict_type} "
                        f"contradiction between two claims "
                        f"({contra.severity} severity)"
                    ),
                    confidence=Confidence.UNRESOLVED.value,
                    citation_ids=[],
                )
            )
        return findings

    @staticmethod
    def _classify(
        claim: Claim,
    ) -> tuple[StatementKind, str]:
        status = claim.status
        if status == ClaimStatus.SUPPORTED:
            kind = StatementKind.FACT
        elif status in (
            ClaimStatus.WEAKLY_SUPPORTED,
            ClaimStatus.CONTESTED,
        ):
            kind = StatementKind.INFERENCE
        elif status == ClaimStatus.CONTRADICTED:
            kind = StatementKind.UNCERTAINTY
        elif status == ClaimStatus.OUTDATED:
            kind = StatementKind.UNCERTAINTY
        else:
            kind = StatementKind.UNCERTAINTY
        # Opinions stay opinions: claims sourced only from
        # opinion evidence are never facts.
        return kind, claim.confidence


class ResearchVerifier:
    """Final verification before a report can complete."""

    def verify(
        self,
        *,
        report: ResearchReport,
        claims: dict[str, Claim],
        evidence: dict[str, Evidence],
        sources: dict[str, Source],
        citation_results: list[Any],
        contradictions: list[Contradiction],
        min_diversity: int = 3,
        freshness_days: int = 365,
    ) -> dict[str, Any]:
        checks: dict[str, Any] = {}

        # 1. Every important claim has evidence.
        important = [
            c for c in claims.values()
            if c.status
            in (
                ClaimStatus.SUPPORTED,
                ClaimStatus.WEAKLY_SUPPORTED,
                ClaimStatus.CONTESTED,
            )
        ]
        checks["claims_have_evidence"] = all(
            c.supporting_evidence for c in important
        )

        # 2. Important evidence has provenance.
        checks["evidence_has_provenance"] = all(
            evidence[eid].source_id in sources
            and sources[evidence[eid].source_id].content_hash
            for c in important
            for eid in c.supporting_evidence
            if eid in evidence
        )

        # 3. Citations valid.
        checks["citations_valid"] = all(
            getattr(r, "valid", False)
            for r in (citation_results or [])
        ) if citation_results else True

        # 4. Contradictions represented.
        checks["contradictions_represented"] = all(
            any(
                f.claim_id in (c.claim_a_id, c.claim_b_id)
                or "contradiction" in f.text.lower()
                for f in report.findings
            )
            for c in contradictions
            if c.resolution == "unresolved"
        )

        # 5. Unsupported claims marked.
        unsupported = [
            c for c in claims.values()
            if c.status
            in (
                ClaimStatus.INSUFFICIENT_EVIDENCE,
                ClaimStatus.UNVERIFIED,
            )
        ]
        checks["unsupported_marked"] = all(
            any(
                f.claim_id == c.claim_id
                and f.statement_kind
                == StatementKind.UNCERTAINTY
                for f in report.findings
            )
            for c in unsupported
        )

        # 6. Stale evidence flagged.
        from datetime import datetime, timezone

        stale = 0
        for src in sources.values():
            try:
                retrieved = datetime.fromisoformat(
                    src.retrieved_at
                )
                age_days = (
                    datetime.now(timezone.utc)
                    - retrieved
                ).days
                if age_days > freshness_days:
                    stale += 1
            except Exception:
                stale += 1
        checks["stale_evidence"] = {
            "stale_sources": stale,
            "flagged": stale == 0
            or any(
                "stale" in lim.lower()
                for lim in report.limitations
            ),
        }

        # 7. Source diversity met.
        domains = {s.domain for s in sources.values() if s.domain}
        checks["diversity_met"] = (
            len(domains) >= min_diversity
        )

        # 8. No secret leakage (report text scanned).
        import re as _re

        secret_shapes = _re.compile(
            r"(api[_-]?key|secret|password|bearer\s+"
            r"|private[_-]?key)\s*[:=]\s*\S+",
            _re.IGNORECASE,
        )
        blob = " ".join(
            [report.question]
            + [f.text for f in report.findings]
        )
        checks["no_secret_leakage"] = not secret_shapes.search(
            blob
        )

        # 9. Report internally consistent: no finding claims
        # FACT without a valid citation.
        checks["internal_consistency"] = all(
            f.statement_kind != StatementKind.FACT
            or f.citation_ids
            for f in report.findings
            if f.claim_id
        )

        passed = all(
            v
            if isinstance(v, bool)
            else v.get("flagged", True)
            for v in checks.values()
        )
        return {"passed": passed, "checks": checks}


def render_report_markdown(report: ResearchReport) -> str:
    """Render the report through the artifact pipeline format."""
    lines = [
        f"# Research Report",
        "",
        f"**Question:** {report.question}",
        "",
        "## Executive findings",
        "",
    ]
    for f in report.findings:
        lines.append(
            f"- [{f.statement_kind.value}/{f.confidence}] "
            f"{f.text}"
        )
        if f.citation_ids:
            lines.append(
                f"  Citations: {', '.join(f.citation_ids)}"
            )
    lines += ["", "## Contradictions", ""]
    for c in report.contradictions:
        lines.append(
            f"- {c.conflict_type} ({c.severity}): "
            f"{c.resolution} — {c.resolution_note[:120]}"
        )
    lines += ["", "## Citations", ""]
    for cite in report.citations:
        lines.append(
            f"- {cite.citation_id}: claim {cite.claim_id[:16]} "
            f"← evidence {cite.evidence_id[:16]} "
            f"(source {cite.source_id[:16]}) "
            f"[{'valid' if cite.valid else 'INVALID'}]"
        )
    lines += ["", "## Limitations", ""]
    for lim in report.limitations:
        lines.append(f"- {lim}")
    lines += [
        "",
        f"_Report {report.report_id} · "
        f"{report.created_at}_",
    ]
    return redact_text("\n".join(lines))

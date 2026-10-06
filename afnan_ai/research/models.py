"""Research & Evidence Intelligence domain models.

Strongly typed, serializable, provenance-carrying models.
Every model redacts secrets at the boundary; external
content is never trusted.

Lifecycle: CREATED → PLANNING → DISCOVERING → COLLECTING →
EXTRACTING → VERIFYING → ANALYZING → SYNTHESIZING →
CITATION_CHECK → COMPLETED | PARTIAL.  BLOCKED / FAILED /
CANCELLED are terminal-ish (BLOCKED can resume).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nid(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def _fingerprint(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8", "replace")
    ).hexdigest()[:16]


class ResearchState(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    DISCOVERING = "discovering"
    COLLECTING = "collecting"
    EXTRACTING = "extracting"
    VERIFYING = "verifying"
    ANALYZING = "analyzing"
    SYNTHESIZING = "synthesizing"
    CITATION_CHECK = "citation_check"
    COMPLETED = "completed"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Valid transitions; anything else raises.
_TRANSITIONS: dict[ResearchState, set[ResearchState]] = {
    ResearchState.CREATED: {
        ResearchState.PLANNING, ResearchState.CANCELLED,
    },
    ResearchState.PLANNING: {
        ResearchState.DISCOVERING, ResearchState.BLOCKED,
        ResearchState.CANCELLED, ResearchState.FAILED,
    },
    ResearchState.DISCOVERING: {
        ResearchState.COLLECTING, ResearchState.BLOCKED,
        ResearchState.FAILED, ResearchState.CANCELLED,
    },
    ResearchState.COLLECTING: {
        ResearchState.EXTRACTING, ResearchState.BLOCKED,
        ResearchState.FAILED, ResearchState.CANCELLED,
        ResearchState.PARTIAL,
    },
    ResearchState.EXTRACTING: {
        ResearchState.VERIFYING, ResearchState.BLOCKED,
        ResearchState.FAILED, ResearchState.PARTIAL,
    },
    ResearchState.VERIFYING: {
        ResearchState.ANALYZING, ResearchState.BLOCKED,
        ResearchState.FAILED, ResearchState.PARTIAL,
        ResearchState.DISCOVERING,  # re-research
    },
    ResearchState.ANALYZING: {
        ResearchState.SYNTHESIZING, ResearchState.BLOCKED,
        ResearchState.FAILED, ResearchState.PARTIAL,
    },
    ResearchState.SYNTHESIZING: {
        ResearchState.CITATION_CHECK, ResearchState.BLOCKED,
        ResearchState.FAILED,
    },
    ResearchState.CITATION_CHECK: {
        ResearchState.COMPLETED, ResearchState.PARTIAL,
        ResearchState.SYNTHESIZING,  # fix citations
        ResearchState.FAILED,
    },
    ResearchState.BLOCKED: {
        ResearchState.PLANNING, ResearchState.DISCOVERING,
        ResearchState.CANCELLED, ResearchState.PARTIAL,
    },
    ResearchState.PARTIAL: {ResearchState.CANCELLED},
    ResearchState.COMPLETED: set(),
    ResearchState.FAILED: set(),
    ResearchState.CANCELLED: set(),
}


def validate_transition(
    from_state: ResearchState, to_state: ResearchState
) -> None:
    if to_state not in _TRANSITIONS.get(from_state, set()):
        raise ValueError(
            f"invalid research transition: "
            f"{from_state.value} → {to_state.value}"
        )


class ClaimStatus(str, Enum):
    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    WEAKLY_SUPPORTED = "weakly_supported"
    CONTESTED = "contested"
    CONTRADICTED = "contradicted"
    OUTDATED = "outdated"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class EvidenceKind(str, Enum):
    DIRECT_STATEMENT = "direct_statement"
    NUMERICAL_FACT = "numerical_fact"
    DOCUMENTED_EVENT = "documented_event"
    QUOTED_CLAIM = "quoted_claim"
    INFERRED_CONCLUSION = "inferred_conclusion"
    OPINION = "opinion"
    INTERPRETATION = "interpretation"


class SourceClass(str, Enum):
    PRIMARY = "primary"
    OFFICIAL_DOC = "official_documentation"
    GOVERNMENT = "government"
    ACADEMIC = "academic"
    COMPANY = "company"
    JOURNALISM = "journalism"
    TECHNICAL = "technical_analysis"
    COMMUNITY = "community"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNRESOLVED = "unresolved"
    INSUFFICIENT = "insufficient_evidence"


class StatementKind(str, Enum):
    FACT = "fact"
    INFERENCE = "inference"
    OPINION = "opinion"
    UNCERTAINTY = "uncertainty"


# ------------------------------------------------------------------
# Core entities
# ------------------------------------------------------------------

@dataclass
class ResearchQuestion:
    question_id: str = field(
        default_factory=lambda: _nid("q")
    )
    text: str = ""
    subquestions: list[str] = field(default_factory=list)
    freshness_days: int = 365
    min_source_diversity: int = 3
    confidence_target: str = "medium"
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "text": self.text,
            "subquestions": list(self.subquestions),
            "freshness_days": self.freshness_days,
            "min_source_diversity": self.min_source_diversity,
            "confidence_target": self.confidence_target,
            "constraints": dict(self.constraints),
        }


@dataclass
class ResearchTask:
    task_id: str = field(
        default_factory=lambda: _nid("rt")
    )
    session_id: str = ""
    kind: str = "discover"  # discover|acquire|extract|
    # verify|synthesize
    description: str = ""
    status: str = "pending"  # pending|running|done|failed
    attempts: int = 0
    result_ref: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "kind": self.kind,
            "description": self.description[:300],
            "status": self.status,
            "attempts": self.attempts,
            "result_ref": self.result_ref,
            "error": self.error[:300],
        }


@dataclass
class ResearchPlan:
    plan_id: str = field(
        default_factory=lambda: _nid("plan")
    )
    session_id: str = ""
    objectives: list[dict[str, Any]] = field(
        default_factory=list
    )  # [{objective, subquestion, source_classes,
    #   freshness_days, rationale}]
    tasks: list[ResearchTask] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "session_id": self.session_id,
            "objectives": list(self.objectives),
            "tasks": [t.to_dict() for t in self.tasks],
            "budget": dict(self.budget),
            "created_at": self.created_at,
        }


@dataclass
class Source:
    source_id: str = field(
        default_factory=lambda: _nid("src")
    )
    session_id: str = ""
    url: str = ""
    title: str = ""
    publisher: str = ""
    domain: str = ""
    source_class: SourceClass = SourceClass.COMMUNITY
    published_at: str = ""
    retrieved_at: str = field(default_factory=_utcnow)
    content_hash: str = ""
    acquisition_method: str = ""
    query_context: str = ""
    status: str = "acquired"  # acquired|failed|stale

    def __post_init__(self) -> None:
        if isinstance(self.source_class, str):
            self.source_class = SourceClass(self.source_class)
        if not self.domain and self.url:
            try:
                from urllib.parse import urlparse

                self.domain = urlparse(self.url).netloc.lower()
            except Exception:
                pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "session_id": self.session_id,
            "url": self.url,
            "title": self.title[:300],
            "publisher": self.publisher[:160],
            "domain": self.domain,
            "source_class": self.source_class.value,
            "published_at": self.published_at,
            "retrieved_at": self.retrieved_at,
            "content_hash": self.content_hash,
            "acquisition_method": self.acquisition_method,
            "query_context": self.query_context[:200],
            "status": self.status,
        }


@dataclass
class SourceSnapshot:
    snapshot_id: str = field(
        default_factory=lambda: _nid("snap")
    )
    source_id: str = ""
    content_ref: str = ""  # artifact/store reference, not raw
    content_hash: str = ""
    captured_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "source_id": self.source_id,
            "content_ref": self.content_ref,
            "content_hash": self.content_hash,
            "captured_at": self.captured_at,
        }


@dataclass
class Evidence:
    evidence_id: str = field(
        default_factory=lambda: _nid("ev")
    )
    session_id: str = ""
    source_id: str = ""
    snapshot_id: str = ""
    kind: EvidenceKind = EvidenceKind.DIRECT_STATEMENT
    text: str = ""
    span_hash: str = ""
    location: dict[str, Any] = field(
        default_factory=dict
    )  # heading|paragraph|page|timestamp|locator
    extracted_at: str = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            self.kind = EvidenceKind(self.kind)
        if not self.span_hash and self.text:
            self.span_hash = _fingerprint(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "session_id": self.session_id,
            "source_id": self.source_id,
            "snapshot_id": self.snapshot_id,
            "kind": self.kind.value,
            "text": self.text[:2000],
            "span_hash": self.span_hash,
            "location": dict(self.location),
            "extracted_at": self.extracted_at,
        }


@dataclass
class Claim:
    claim_id: str = field(
        default_factory=lambda: _nid("claim")
    )
    session_id: str = ""
    normalized_text: str = ""
    original_text: str = ""
    supporting_evidence: list[str] = field(
        default_factory=list
    )
    contradicting_evidence: list[str] = field(
        default_factory=list
    )
    source_ids: list[str] = field(default_factory=list)
    confidence: str = Confidence.UNRESOLVED.value
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    temporal_validity: str = ""
    verified_at: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = ClaimStatus(self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "session_id": self.session_id,
            "normalized_text": self.normalized_text[:500],
            "original_text": self.original_text[:500],
            "supporting_evidence": list(
                self.supporting_evidence
            ),
            "contradicting_evidence": list(
                self.contradicting_evidence
            ),
            "source_ids": list(self.source_ids),
            "confidence": self.confidence,
            "status": self.status.value,
            "temporal_validity": self.temporal_validity,
            "verified_at": self.verified_at,
        }


@dataclass
class Citation:
    citation_id: str = field(
        default_factory=lambda: _nid("cite")
    )
    session_id: str = ""
    claim_id: str = ""
    evidence_id: str = ""
    source_id: str = ""
    locator: str = ""  # human-readable reference
    valid: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "session_id": self.session_id,
            "claim_id": self.claim_id,
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "locator": self.locator[:300],
            "valid": self.valid,
        }


@dataclass
class SourceTrustProfile:
    source_id: str = ""
    signals: dict[str, Any] = field(default_factory=dict)
    score: float = 0.5  # 0..1, credibility — NOT truth
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "signals": dict(self.signals),
            "score": round(self.score, 3),
            "notes": [n[:200] for n in self.notes],
        }


@dataclass
class Contradiction:
    contradiction_id: str = field(
        default_factory=lambda: _nid("contra")
    )
    session_id: str = ""
    claim_a_id: str = ""
    claim_b_id: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    conflict_type: str = "direct"  # direct|numerical|date|
    # definition|scope|temporal|methodological
    severity: str = "medium"  # low|medium|high
    resolution: str = "unresolved"  # unresolved|resolved_a|
    # resolved_b|both_partial|needs_human
    resolution_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "session_id": self.session_id,
            "claim_a_id": self.claim_a_id,
            "claim_b_id": self.claim_b_id,
            "evidence_ids": list(self.evidence_ids),
            "source_ids": list(self.source_ids),
            "conflict_type": self.conflict_type,
            "severity": self.severity,
            "resolution": self.resolution,
            "resolution_note": self.resolution_note[:500],
        }


@dataclass
class ResearchFinding:
    finding_id: str = field(
        default_factory=lambda: _nid("find")
    )
    session_id: str = ""
    claim_id: str = ""
    statement_kind: StatementKind = StatementKind.FACT
    text: str = ""
    confidence: str = Confidence.MEDIUM.value
    citation_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.statement_kind, str):
            self.statement_kind = StatementKind(
                self.statement_kind
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "session_id": self.session_id,
            "claim_id": self.claim_id,
            "statement_kind": self.statement_kind.value,
            "text": self.text[:1000],
            "confidence": self.confidence,
            "citation_ids": list(self.citation_ids),
        }


@dataclass
class ResearchReport:
    report_id: str = field(
        default_factory=lambda: _nid("rep")
    )
    session_id: str = ""
    question: str = ""
    findings: list[ResearchFinding] = field(
        default_factory=list
    )
    contradictions: list[Contradiction] = field(
        default_factory=list
    )
    citations: list[Citation] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    artifact_id: str = ""
    verification: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "session_id": self.session_id,
            "question": self.question[:500],
            "findings": [f.to_dict() for f in self.findings],
            "contradictions": [
                c.to_dict() for c in self.contradictions
            ],
            "citations": [c.to_dict() for c in self.citations],
            "limitations": list(self.limitations),
            "artifact_id": self.artifact_id,
            "verification": dict(self.verification),
            "created_at": self.created_at,
        }


@dataclass
class ResearchCheckpoint:
    checkpoint_id: str = field(
        default_factory=lambda: _nid("ckpt")
    )
    session_id: str = ""
    state: str = ""
    plan: dict[str, Any] = field(default_factory=dict)
    completed_tasks: list[str] = field(default_factory=list)
    pending_tasks: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    contradiction_ids: list[str] = field(
        default_factory=list
    )
    citation_ids: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "state": self.state,
            "plan": dict(self.plan),
            "completed_tasks": list(self.completed_tasks),
            "pending_tasks": list(self.pending_tasks),
            "source_ids": list(self.source_ids),
            "evidence_ids": list(self.evidence_ids),
            "claim_ids": list(self.claim_ids),
            "contradiction_ids": list(
                self.contradiction_ids
            ),
            "citation_ids": list(self.citation_ids),
            "metrics": dict(self.metrics),
            "created_at": self.created_at,
        }


@dataclass
class ResearchSession:
    session_id: str = field(
        default_factory=lambda: _nid("rsess")
    )
    question: str = ""
    state: ResearchState = ResearchState.CREATED
    plan: ResearchPlan | None = None
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    task_id: str = ""  # owning agent task
    workspace_id: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.state, str):
            self.state = ResearchState(self.state)

    def transition(self, to_state: ResearchState) -> None:
        if isinstance(to_state, str):
            to_state = ResearchState(to_state)
        validate_transition(self.state, to_state)
        self.state = to_state
        self.updated_at = _utcnow()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "question": self.question[:500],
            "state": self.state.value,
            "plan": (
                self.plan.to_dict() if self.plan else None
            ),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "task_id": self.task_id,
            "workspace_id": self.workspace_id,
        }


@dataclass
class ResearchMetrics:
    session_id: str = ""
    sources_discovered: int = 0
    sources_acquired: int = 0
    unique_domains: int = 0
    primary_source_ratio: float = 0.0
    evidence_count: int = 0
    claims_count: int = 0
    corroboration_rate: float = 0.0
    contradiction_count: int = 0
    unresolved_contradictions: int = 0
    citation_coverage: float = 0.0
    unsupported_claims: int = 0
    stale_source_ratio: float = 0.0
    failed_tasks: int = 0
    replans: int = 0
    elapsed_s: float = 0.0
    budget_used_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "sources_discovered": self.sources_discovered,
            "sources_acquired": self.sources_acquired,
            "unique_domains": self.unique_domains,
            "primary_source_ratio": round(
                self.primary_source_ratio, 3
            ),
            "evidence_count": self.evidence_count,
            "claims_count": self.claims_count,
            "corroboration_rate": round(
                self.corroboration_rate, 3
            ),
            "contradiction_count": self.contradiction_count,
            "unresolved_contradictions": (
                self.unresolved_contradictions
            ),
            "citation_coverage": round(
                self.citation_coverage, 3
            ),
            "unsupported_claims": self.unsupported_claims,
            "stale_source_ratio": round(
                self.stale_source_ratio, 3
            ),
            "failed_tasks": self.failed_tasks,
            "replans": self.replans,
            "elapsed_s": round(self.elapsed_s, 1),
            "budget_used_pct": round(self.budget_used_pct, 1),
        }

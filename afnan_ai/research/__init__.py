"""Research & Evidence Intelligence for Afnan AI.

Turns complex questions into multi-source research with
evidence, claims, an evidence graph, contradictions,
confidence assessment and machine-verifiable citations —
published as a verified Artifact.

The ResearchEngine orchestrates the research lifecycle
*inside* the AgentLoop's control; it never replaces the
loop.  All acquisition goes through existing browser /
connector / tool systems, all authorization through
SecurityCenter, all persistence through existing stores.
"""

from afnan_ai.research.citations import (
    CitationCheck,
    CitationValidator,
)
from afnan_ai.research.claims import (
    ClaimManager,
    ContradictionDetector,
)
from afnan_ai.research.engine import ResearchEngine
from afnan_ai.research.evidence import (
    EvidenceExtractor,
    EvidenceGraph,
    detect_corroboration,
    split_spans,
)
from afnan_ai.research.integration import (
    attach_activity_center,
    check_research_permissions,
    dispatch_subagent_research,
    promote_findings_to_memory,
    publish_report_artifact,
    record_trajectory,
    research_skill_definition,
    save_checkpoint,
)
from afnan_ai.research.models import (
    Citation,
    Claim,
    ClaimStatus,
    Confidence,
    Contradiction,
    Evidence,
    EvidenceKind,
    ResearchCheckpoint,
    ResearchMetrics,
    ResearchPlan,
    ResearchQuestion,
    ResearchReport,
    ResearchSession,
    ResearchState,
    ResearchTask,
    Source,
    SourceClass,
    SourceSnapshot,
    SourceTrustProfile,
    StatementKind,
    validate_transition,
)
from afnan_ai.research.observability import (
    emit_research_audit,
    emit_research_event,
)
from afnan_ai.research.planning import (
    decompose_question,
    replan_after_failure,
)
from afnan_ai.research.recovery import (
    BudgetTracker,
    RecoveryPolicy,
    ResearchBudget,
)
from afnan_ai.research.sources import (
    SourceAcquisition,
    SourceDiscovery,
    WebResearchProvider,
    build_trust_profile,
)
from afnan_ai.research.synthesis import (
    ResearchSynthesizer,
    ResearchVerifier,
    render_report_markdown,
)

__all__ = [
    "BudgetTracker",
    "Citation",
    "CitationCheck",
    "CitationValidator",
    "Claim",
    "ClaimManager",
    "ClaimStatus",
    "Confidence",
    "Contradiction",
    "ContradictionDetector",
    "Evidence",
    "EvidenceExtractor",
    "EvidenceGraph",
    "EvidenceKind",
    "RecoveryPolicy",
    "ResearchBudget",
    "ResearchCheckpoint",
    "ResearchEngine",
    "ResearchMetrics",
    "ResearchPlan",
    "ResearchQuestion",
    "ResearchReport",
    "ResearchSession",
    "ResearchState",
    "ResearchSynthesizer",
    "ResearchTask",
    "ResearchVerifier",
    "Source",
    "SourceAcquisition",
    "SourceClass",
    "SourceDiscovery",
    "SourceSnapshot",
    "SourceTrustProfile",
    "StatementKind",
    "WebResearchProvider",
    "attach_activity_center",
    "build_trust_profile",
    "check_research_permissions",
    "decompose_question",
    "detect_corroboration",
    "dispatch_subagent_research",
    "emit_research_audit",
    "emit_research_event",
    "promote_findings_to_memory",
    "publish_report_artifact",
    "replan_after_failure",
    "record_trajectory",
    "render_report_markdown",
    "research_skill_definition",
    "save_checkpoint",
    "split_spans",
    "validate_transition",
]

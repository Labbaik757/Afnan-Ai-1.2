"""ResearchEngine — research workflow orchestration.

The engine drives the research lifecycle:

    Goal → Research Planning → Source Discovery →
    Source Acquisition → Evidence Extraction → Claim
    Formation → Evidence Graph → Cross-Source Verification →
    Contradiction Analysis → Confidence Assessment →
    Citation/Provenance Validation → Synthesis → Artifact →
    Verification → Completion / Re-research / Human Approval

It is NOT a replacement for the AgentLoop: the loop keeps
overall execution control.  The engine exposes a `run`
method the loop (or a research tool) calls; every phase
uses injected adapters so the AgentLoop, Planner,
SubagentManager, ContextManager and friends stay the
authorities they already are.
"""

from __future__ import annotations

import time
from typing import Any

from afnan_ai.redaction import redact_text
from afnan_ai.research.citations import CitationValidator
from afnan_ai.research.claims import (
    ClaimManager,
    ContradictionDetector,
)
from afnan_ai.research.evidence import (
    EvidenceExtractor,
    EvidenceGraph,
    detect_corroboration,
)
from afnan_ai.research.models import (
    Citation,
    Claim,
    Evidence,
    ResearchCheckpoint,
    ResearchMetrics,
    ResearchPlan,
    ResearchReport,
    ResearchSession,
    ResearchState,
    Source,
    validate_transition,
)
from afnan_ai.research.observability import emit_research_event
from afnan_ai.research.planning import decompose_question
from afnan_ai.research.recovery import (
    BudgetTracker,
    RecoveryPolicy,
    ResearchBudget,
)
from afnan_ai.research.sources import (
    SourceAcquisition,
    SourceDiscovery,
    build_trust_profile,
)
from afnan_ai.research.synthesis import (
    ResearchSynthesizer,
    ResearchVerifier,
    render_report_markdown,
)


class ResearchEngine:
    """Orchestrate one research session."""

    def __init__(
        self,
        *,
        source_provider: Any,
        scan_injection: Any = None,
        activity_center: Any = None,
        budget: ResearchBudget | None = None,
    ) -> None:
        self.provider = source_provider
        self.scan_injection = scan_injection
        self.activity = activity_center
        self.budget = budget or ResearchBudget()
        # Session state (in-memory; checkpoints persist it).
        self.sessions: dict[str, ResearchSession] = {}
        self.sources: dict[str, Source] = {}
        self.evidence: dict[str, Evidence] = {}
        self.claims: dict[str, Claim] = {}
        self.citations: dict[str, Citation] = {}
        self.checkpoints: dict[str, ResearchCheckpoint] = {}

    # -- session ---------------------------------------------------------
    def start_session(
        self,
        question: str,
        *,
        task_id: str = "",
        workspace_id: str = "",
        budget: ResearchBudget | None = None,
    ) -> ResearchSession:
        session = ResearchSession(
            question=question,
            task_id=task_id,
            workspace_id=workspace_id,
        )
        self.sessions[session.session_id] = session
        self._event(
            "research.started",
            f"research started: {question[:120]}",
            session,
        )
        return session

    # -- main run ----------------------------------------------------------
    def run(
        self,
        session: ResearchSession,
        *,
        budget: ResearchBudget | None = None,
    ) -> ResearchReport:
        """Execute the full lifecycle for one session."""
        tracker = BudgetTracker(budget or self.budget)
        started = time.monotonic()
        try:
            self._plan(session)
            self._discover(session, tracker)
            self._acquire(session, tracker)
            self._extract(session)
            self._verify(session)
            report = self._synthesize(session, tracker)
            self._event(
                "research.completed",
                f"research completed: "
                f"{len(report.findings)} findings",
                session,
            )
            return report
        except _PartialResult as partial:
            session.transition(ResearchState.PARTIAL)
            self._event(
                "research.partial",
                f"partial result: {partial.reason[:120]}",
                session,
            )
            return partial.report
        except Exception as exc:
            session.transition(ResearchState.FAILED)
            self._event(
                "research.failed",
                f"research failed: {type(exc).__name__}",
                session,
            )
            raise

    # -- phases --------------------------------------------------------------
    def _plan(self, session: ResearchSession) -> None:
        session.transition(ResearchState.PLANNING)
        plan = decompose_question(
            session.question,
            budget=self.budget.to_dict(),
            session_id=session.session_id,
        )
        session.plan = plan
        session.transition(ResearchState.DISCOVERING)
        self._event(
            "research.planned",
            f"plan: {len(plan.objectives)} objectives, "
            f"{len(plan.tasks)} tasks",
            session,
        )

    def _discover(
        self,
        session: ResearchSession,
        tracker: BudgetTracker,
    ) -> None:
        plan = session.plan
        assert plan is not None
        discovery = SourceDiscovery(
            self.provider,
            max_sources=self.budget.max_sources,
            max_searches=self.budget.max_searches,
            min_diversity=3,
        )
        queries = [
            obj["subquestion"] for obj in plan.objectives
        ]
        self._event(
            "research.search.started",
            f"searching {len(queries)} queries",
            session,
        )
        sources = discovery.discover(
            queries, session_id=session.session_id
        )
        tracker.searches += discovery.searches_done
        tracker.sources += len(sources)
        for src in sources:
            self.sources[src.source_id] = src
        if tracker.exhausted():
            raise _PartialResult(
                "budget exhausted during discovery",
                self._partial_report(session, tracker),
            )
        if not sources:
            raise _PartialResult(
                "no sources discovered",
                self._partial_report(session, tracker),
            )
        session.transition(ResearchState.COLLECTING)
        self._event(
            "research.source.discovered",
            f"{len(sources)} sources discovered",
            session,
            details={"count": len(sources)},
        )

    def _acquire(
        self,
        session: ResearchSession,
        tracker: BudgetTracker,
    ) -> None:
        acquisition = SourceAcquisition(self.provider)
        acquired = 0
        for src in list(self.sources.values()):
            if src.session_id != session.session_id:
                continue
            if tracker.exhausted():
                break
            snapshot = acquisition.acquire(src)
            if snapshot is not None:
                acquired += 1
                # Stash content on the session for extraction.
                src_snapshot = snapshot
                self._remember_content(
                    session, src, src_snapshot, acquisition
                )
        session.transition(ResearchState.EXTRACTING)
        self._event(
            "research.source.acquired",
            f"{acquired} sources acquired",
            session,
            details={"acquired": acquired},
        )
        if acquired == 0:
            raise _PartialResult(
                "no sources could be acquired",
                self._partial_report(session, tracker),
            )

    def _remember_content(
        self,
        session: ResearchSession,
        src: Source,
        snapshot: Any,
        acquisition: SourceAcquisition,
    ) -> None:
        stash = getattr(session, "_content_stash", None)
        if stash is None:
            stash = {}
            session._content_stash = stash  # type: ignore[attr-defined]
        stash[src.source_id] = (
            snapshot,
            acquisition.content_for(snapshot),
        )

    def _extract(self, session: ResearchSession) -> None:
        extractor = EvidenceExtractor(
            scan_injection=self.scan_injection
        )
        graph: EvidenceGraph = self._graph(session)
        query_terms = session.question.split()[:8]
        total = 0
        stash = getattr(session, "_content_stash", {})
        for src in self.sources.values():
            if src.session_id != session.session_id:
                continue
            if src.status != "acquired":
                continue
            item = stash.get(src.source_id)
            if not item:
                continue
            snapshot, content = item
            graph.add_source(src)
            for ev in extractor.extract(
                src,
                snapshot,
                content,
                session_id=session.session_id,
                query_terms=query_terms,
            ):
                self.evidence[ev.evidence_id] = ev
                graph.add_evidence(ev)
                total += 1
        if extractor.injection_flags:
            self._event(
                "research.source.acquired",
                f"{len(extractor.injection_flags)} "
                "injection-like spans quarantined",
                session,
            )
        session.transition(ResearchState.VERIFYING)
        self._event(
            "research.evidence.extracted",
            f"{total} evidence spans extracted",
            session,
            details={"count": total},
        )

    def _verify(self, session: ResearchSession) -> None:
        graph: EvidenceGraph = self._graph(session)
        manager = ClaimManager(session_id=session.session_id)
        # Form claims from evidence (one claim per span here;
        # the AgentLoop/LLM refines them in production).
        for ev in self.evidence.values():
            if ev.session_id != session.session_id:
                continue
            claim = manager.get_or_create(
                ev.text, original=ev.text
            )
            graph.add_claim(claim.claim_id)
            manager.support(claim, ev, graph=graph)
            self._event(
                "research.claim.created",
                f"claim: {claim.normalized_text[:80]}",
                session,
            )
        # Corroboration across independent domains.
        domains = {
            ev.evidence_id: self.sources[ev.source_id].domain
            for ev in self.evidence.values()
            if ev.source_id in self.sources
        }
        session_evidence = [
            e for e in self.evidence.values()
            if e.session_id == session.session_id
        ]
        for a_id, b_id in detect_corroboration(
            session_evidence, domains
        ):
            graph.link(
                f"evidence:{a_id}", "corroborates",
                f"evidence:{b_id}",
            )
        # Assess + contradictions.
        for claim in manager.claims.values():
            manager.assess(claim)
            self.claims[claim.claim_id] = claim
            self._event(
                "research.claim.verified",
                f"{claim.status.value}: "
                f"{claim.normalized_text[:80]}",
                session,
            )
        detector = ContradictionDetector(
            session_id=session.session_id
        )
        contradictions = detector.detect(
            list(manager.claims.values()),
            self.evidence,
            domains,
        )
        session._contradictions = contradictions  # type: ignore[attr-defined]
        if contradictions:
            self._event(
                "research.contradiction.detected",
                f"{len(contradictions)} contradictions",
                session,
            )
        session._claim_manager = manager  # type: ignore[attr-defined]
        session.transition(ResearchState.ANALYZING)

    def _synthesize(
        self,
        session: ResearchSession,
        tracker: BudgetTracker,
    ) -> ResearchReport:
        session.transition(ResearchState.SYNTHESIZING)
        manager = getattr(session, "_claim_manager", None)
        claims = (
            dict(manager.claims)
            if manager
            else dict(self.claims)
        )
        contradictions = getattr(
            session, "_contradictions", []
        )
        # Citations: one per supported claim → first evidence.
        citations: list[Citation] = []
        graph = self._graph(session)
        for claim in claims.values():
            if not claim.supporting_evidence:
                continue
            eid = claim.supporting_evidence[0]
            ev = self.evidence.get(eid)
            if ev is None:
                continue
            citations.append(
                Citation(
                    session_id=session.session_id,
                    claim_id=claim.claim_id,
                    evidence_id=eid,
                    source_id=ev.source_id,
                    locator=(
                        f"{self.sources[ev.source_id].domain} "
                        f"¶{ev.location.get('paragraph', '?')}"
                        if ev.source_id in self.sources
                        else ""
                    ),
                )
            )
        session.transition(ResearchState.CITATION_CHECK)
        validator = CitationValidator()
        results = validator.validate(
            citations,
            claims=claims,
            evidence=self.evidence,
            sources=self.sources,
            graph=graph,
        )
        invalid = [r for r in results if not r.valid]
        if invalid:
            self._event(
                "research.citation.invalid",
                f"{len(invalid)} invalid citations",
                session,
            )
            raise _PartialResult(
                f"{len(invalid)} invalid citations",
                self._partial_report(session, tracker),
            )
        synthesizer = ResearchSynthesizer(
            session_id=session.session_id
        )
        findings = synthesizer.synthesize(
            claims, citations, contradictions
        )
        report = ResearchReport(
            session_id=session.session_id,
            question=session.question,
            findings=findings,
            contradictions=contradictions,
            citations=citations,
            limitations=self._limitations(session),
        )
        verifier = ResearchVerifier()
        verdict = verifier.verify(
            report=report,
            claims=claims,
            evidence=self.evidence,
            sources=self.sources,
            citation_results=results,
            contradictions=contradictions,
        )
        report.verification = verdict
        if not verdict["passed"]:
            session.transition(ResearchState.PARTIAL)
            self._event(
                "research.partial",
                "verification failed; partial report",
                session,
            )
            return report
        session.transition(ResearchState.COMPLETED)
        return report

    # -- helpers -----------------------------------------------------------
    def _graph(self, session: ResearchSession) -> EvidenceGraph:
        graph = getattr(session, "_graph", None)
        if graph is None:
            graph = EvidenceGraph()
            session._graph = graph  # type: ignore[attr-defined]
        return graph

    def _limitations(
        self, session: ResearchSession
    ) -> list[str]:
        claims = [
            c for c in self.claims.values()
            if c.session_id == session.session_id
        ]
        lims = []
        if any(
            c.status.value == "insufficient_evidence"
            for c in claims
        ):
            lims.append(
                "Some claims lack sufficient evidence and "
                "are marked as uncertainty."
            )
        contradictions = getattr(
            session, "_contradictions", []
        )
        unresolved = [
            c for c in contradictions
            if c.resolution == "unresolved"
        ]
        if unresolved:
            lims.append(
                f"{len(unresolved)} contradictions remain "
                "unresolved and are shown explicitly."
            )
        lims.append(
            "Research reflects sources available at "
            "retrieval time; sources may change."
        )
        return lims

    def _partial_report(
        self,
        session: ResearchSession,
        tracker: BudgetTracker,
    ) -> ResearchReport:
        return ResearchReport(
            session_id=session.session_id,
            question=session.question,
            limitations=[
                "Partial result: research stopped early."
            ],
        )

    def checkpoint(
        self, session: ResearchSession
    ) -> ResearchCheckpoint:
        ckpt = ResearchCheckpoint(
            session_id=session.session_id,
            state=session.state.value,
            plan=(
                session.plan.to_dict()
                if session.plan
                else {}
            ),
            completed_tasks=[
                t.task_id
                for t in (session.plan.tasks if session.plan else [])
                if t.status == "done"
            ],
            pending_tasks=[
                t.task_id
                for t in (session.plan.tasks if session.plan else [])
                if t.status != "done"
            ],
            source_ids=[
                s for s, src in self.sources.items()
                if src.session_id == session.session_id
            ],
            evidence_ids=[
                e for e, ev in self.evidence.items()
                if ev.session_id == session.session_id
            ],
            claim_ids=[
                c for c, cl in self.claims.items()
                if cl.session_id == session.session_id
            ],
            citation_ids=[
                c for c, ci in self.citations.items()
                if ci.session_id == session.session_id
            ],
        )
        self.checkpoints[ckpt.checkpoint_id] = ckpt
        self._event(
            "research.checkpoint.created",
            f"checkpoint {ckpt.checkpoint_id[:16]}",
            session,
        )
        return ckpt

    def metrics(
        self,
        session: ResearchSession,
        tracker: BudgetTracker | None = None,
    ) -> ResearchMetrics:
        sources = [
            s for s in self.sources.values()
            if s.session_id == session.session_id
        ]
        claims = [
            c for c in self.claims.values()
            if c.session_id == session.session_id
        ]
        contradictions = getattr(
            session, "_contradictions", []
        )
        domains = {s.domain for s in sources if s.domain}
        primary = sum(
            1
            for s in sources
            if s.source_class.value
            in ("primary", "official_documentation")
        )
        return ResearchMetrics(
            session_id=session.session_id,
            sources_discovered=len(sources),
            sources_acquired=sum(
                1 for s in sources if s.status == "acquired"
            ),
            unique_domains=len(domains),
            primary_source_ratio=(
                primary / len(sources) if sources else 0.0
            ),
            evidence_count=sum(
                1 for e in self.evidence.values()
                if e.session_id == session.session_id
            ),
            claims_count=len(claims),
            contradiction_count=len(contradictions),
            unresolved_contradictions=sum(
                1 for c in contradictions
                if c.resolution == "unresolved"
            ),
            unsupported_claims=sum(
                1 for c in claims
                if c.status.value == "insufficient_evidence"
            ),
            budget_used_pct=(
                tracker.budget_used_pct() if tracker else 0.0
            ),
        )

    def render_markdown(
        self, report: ResearchReport
    ) -> str:
        return render_report_markdown(report)

    def _event(
        self,
        event: str,
        summary: str,
        session: ResearchSession,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        if self.activity is None:
            return
        emit_research_event(
            self.activity,
            event,
            summary,
            session_id=session.session_id,
            task_id=session.task_id,
            details=details,
        )


class _PartialResult(Exception):
    def __init__(self, reason: str, report: ResearchReport) -> None:
        super().__init__(reason)
        self.reason = reason
        self.report = report

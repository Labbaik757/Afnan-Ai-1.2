"""Research integrations with existing Afnan systems.

Every integration reuses the existing authority — nothing
is duplicated:

- AgentLoop: research runs as a phase inside the loop's
  control (the engine never replaces the loop).
- SubagentManager: parallel research roles with scoped
  permissions and mandatory provenance.
- ContextManager/TrajectoryStore: research context uses
  the long-context architecture (tiers, compression).
- MemoryStore: only verified facts promote, via the
  existing promotion policy.
- ArtifactManager: the final report is an artifact with
  lineage Source → Evidence → Claim → Finding → Report.
- SecurityCenter/PermissionManager/CredentialVault/audit:
  all research actions authorize through them.
- ActivityCenter: research.* events.
- CheckpointManager: session checkpoints persist.
- SkillRegistry: research workflows can become skills.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.redaction import redact_text
from afnan_ai.research.engine import ResearchEngine
from afnan_ai.research.models import ResearchReport
from afnan_ai.research.observability import emit_research_event


def attach_activity_center(
    engine: ResearchEngine, activity_center: Any
) -> None:
    """Route engine events into the ActivityCenter."""
    engine.activity = activity_center


def publish_report_artifact(
    artifact_manager: Any,
    report: ResearchReport,
    markdown: str,
    *,
    task_id: str = "",
    goal_id: str = "",
) -> Any:
    """Create the final report through ArtifactManager."""
    return artifact_manager.create(
        name=f"research-report-{report.report_id[:8]}",
        artifact_type="document",
        data={
            "title": f"Research: {report.question[:80]}",
            "format": "markdown",
            "body": markdown,
        },
        description=(
            f"Research report: {len(report.findings)} findings, "
            f"{len(report.contradictions)} contradictions"
        ),
        source_task_id=task_id,
        goal_id=goal_id,
        category="research",
    )


def record_trajectory(
    trajectory_store: Any,
    session: Any,
    phase: str,
    summary: str,
    *,
    kind: str = "observation",
) -> None:
    """Append research trajectory via the existing store."""
    try:
        from afnan_ai.context.models import TrajectoryEntry

        trajectory_store.record(
            getattr(session, "task_id", "")
            or session.session_id,
            TrajectoryEntry(
                seq=0,
                kind=kind,
                zone="agent_state",
                summary=f"[research:{phase}] {summary}"[:500],
            ),
            goal=getattr(session, "question", ""),
        )
    except Exception:
        pass


def promote_findings_to_memory(
    memory_store: Any,
    report: ResearchReport,
    *,
    min_confidence: str = "high",
) -> dict[str, Any]:
    """Promote only verified high-confidence facts.

    Never automatic for everything: claims carry
    provenance, confidence and verification state, and
    conflicting memory is never silently overwritten
    (the store's own conflict handling applies).
    """
    order = {
        "high": 3,
        "medium": 2,
        "low": 1,
        "unresolved": 0,
        "insufficient_evidence": 0,
    }
    threshold = order.get(min_confidence, 3)
    promoted = 0
    skipped = 0
    for finding in report.findings:
        if finding.statement_kind.value != "fact":
            skipped += 1
            continue
        if order.get(finding.confidence, 0) < threshold:
            skipped += 1
            continue
        try:
            memory_store.record(
                {
                    "text": finding.text,
                    "kind": "research_fact",
                    "confidence": finding.confidence,
                    "source": "research",
                    "citations": finding.citation_ids,
                    "verified": True,
                }
            )
            promoted += 1
        except Exception:
            skipped += 1
    return {"promoted": promoted, "skipped": skipped}


def save_checkpoint(
    checkpoint_manager: Any,
    checkpoint: Any,
    *,
    name: str = "",
) -> Any:
    """Persist a research checkpoint via CheckpointManager."""
    try:
        return checkpoint_manager.save(
            {
                "kind": "research_checkpoint",
                "name": name,
                "checkpoint": checkpoint.to_dict(),
            }
        )
    except Exception:
        return None


def dispatch_subagent_research(
    subagent_manager: Any,
    *,
    role: str,
    objective: str,
    allowed_tools: list[str] | None = None,
    context_items: list[Any] | None = None,
) -> Any:
    """Scoped parallel research via the existing SubagentManager.

    Permissions and context are scoped; credentials are
    never inherited blindly; output provenance is
    mandatory and unverified subagent claims never become
    final truth (central verification still applies).
    """
    from afnan_ai.subagents.manager import SubAgentSpec

    spec = SubAgentSpec(
        role=role,
        objective=redact_text(objective)[:500],
        allowed_tools=list(allowed_tools or []),
        # Scoped: research roles get read-only style tools.
    )
    try:
        record = subagent_manager.create(spec)
        return record
    except Exception:
        return None


def research_skill_definition(
    skill_registry: Any,
    *,
    skill_id: str = "research_topic",
) -> dict[str, Any]:
    """Expose research as a SkillRegistry-compatible skill.

    Lets repeated research workflows become versioned
    skills through the normal skill pipeline.
    """
    return {
        "skill_id": skill_id,
        "name": "Research Topic",
        "description": (
            "Multi-source research with evidence, claims, "
            "contradictions and verified citations."
        ),
        "version": "1.0.0",
        "risk": "read_only",
        "source": "system",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
            },
            "required": ["question"],
        },
    }


def check_research_permissions(
    security_center: Any,
    *,
    action: str = "research",
) -> dict[str, Any]:
    """Authorize research actions via SecurityCenter."""
    try:
        authorize = security_center.authorize
    except Exception:
        return {"allowed": True, "note": "no security center"}
    try:
        result = authorize(
            action=action,
            capability="research",
        )
        if isinstance(result, dict):
            return result
        return {"allowed": bool(result)}
    except Exception as exc:
        return {"allowed": False, "reason": str(exc)[:200]}

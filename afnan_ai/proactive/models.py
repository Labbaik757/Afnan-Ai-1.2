"""Proactive intelligence data models.

An Idea is a single evidence-based suggestion: what to do,
why, what it is based on, how confident the engine is, and
what it would take to act on it.  Ideas never execute
themselves — execution only happens after the user accepts
(or a configured, read-only, low-risk auto-action), and
then through the normal TaskManager → AgentLoop path.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class SuggestionType(str, Enum):
    UNFINISHED_TASK = "unfinished_task"
    GOAL_PROGRESS = "goal_progress"
    FOLLOW_UP = "follow_up"
    RECURRING_WORKFLOW = "recurring_workflow"
    RESEARCH_OPPORTUNITY = "research_opportunity"
    PROJECT_IMPROVEMENT = "project_improvement"
    MISSED_DEPENDENCY = "missed_dependency"
    UPCOMING_DEADLINE = "upcoming_deadline"
    AUTOMATION_OPPORTUNITY = "automation_opportunity"


class IdeaStatus(str, Enum):
    NEW = "new"
    DISMISSED = "dismissed"
    ACCEPTED = "accepted"
    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


# Risk vocabulary shared with the skill system.
RISK_READ_ONLY = "read_only"
RISK_REVERSIBLE = "reversible"
RISK_SENSITIVE = "sensitive"
RISK_DESTRUCTIVE = "destructive"
RISKS_REQUIRING_APPROVAL = (
    RISK_SENSITIVE, RISK_DESTRUCTIVE,
)


@dataclass
class Idea:
    """One proactive suggestion."""

    idea_id: str
    title: str
    description: str
    reason: str
    suggestion_type: str
    confidence: float  # 0..1, evidence-backed only
    priority: int  # 1..5
    suggested_action: dict[str, Any] = field(
        default_factory=dict
    )
    risk_level: str = RISK_READ_ONLY
    related_goal_id: str = ""
    related_task_id: str = ""
    evidence: list[str] = field(default_factory=list)
    status: IdeaStatus = IdeaStatus.NEW
    created_at: str = field(default_factory=_utcnow)
    expires_at: str = ""
    feedback: list[dict[str, Any]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        self.idea_id = str(self.idea_id or "").strip()
        if not self.idea_id:
            raise ValueError("idea_id is required")
        self.confidence = max(0.0, min(1.0, float(
            self.confidence)))
        self.priority = max(1, min(5, int(self.priority)))
        if isinstance(self.status, str):
            self.status = IdeaStatus(self.status)

    @staticmethod
    def new_id() -> str:
        return f"idea_{uuid.uuid4().hex[:12]}"

    def signature(self) -> str:
        """Deduplication key: same opportunity → same key."""
        raw = "|".join((
            str(self.suggestion_type),
            str(self.related_goal_id),
            str(self.related_task_id),
            str(self.title).strip().lower()[:80],
        ))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def is_expired(self, now: str = "") -> bool:
        if not self.expires_at:
            return False
        ref = now or _utcnow()
        return ref >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "idea_id": self.idea_id,
            "title": self.title,
            "description": self.description,
            "reason": self.reason,
            "suggestion_type": str(self.suggestion_type),
            "confidence": self.confidence,
            "priority": self.priority,
            "suggested_action": dict(self.suggested_action),
            "risk_level": self.risk_level,
            "related_goal_id": self.related_goal_id,
            "related_task_id": self.related_task_id,
            "evidence": list(self.evidence),
            "status": self.status.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "feedback": list(self.feedback),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Idea":
        return cls(
            idea_id=str(data.get("idea_id", "")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            reason=str(data.get("reason", "")),
            suggestion_type=str(
                data.get("suggestion_type", "")
            ),
            confidence=float(data.get("confidence", 0.0)),
            priority=int(data.get("priority", 3)),
            suggested_action=dict(
                data.get("suggested_action") or {}
            ),
            risk_level=str(
                data.get("risk_level", RISK_READ_ONLY)
            ),
            related_goal_id=str(
                data.get("related_goal_id", "")
            ),
            related_task_id=str(
                data.get("related_task_id", "")
            ),
            evidence=list(data.get("evidence") or []),
            status=str(data.get("status", "new")),
            created_at=str(
                data.get("created_at", _utcnow())
            ),
            expires_at=str(data.get("expires_at", "")),
            feedback=list(data.get("feedback") or []),
        )


@dataclass
class ProactiveConfig:
    """User-controlled scope for proactive behavior."""

    enabled: bool = True
    # Sources the engine may read (subset of:
    # goals, tasks, schedules, memories, activity).
    allowed_sources: tuple[str, ...] = (
        "goals", "tasks", "schedules", "memories",
        "activity",
    )
    quiet_start: str = ""  # "HH:MM", empty = no quiet hours
    quiet_end: str = ""
    cooldown_s: float = 6 * 3600.0
    max_per_period: int = 5
    period_s: float = 24 * 3600.0
    relevance_threshold: float = 0.5
    # Read-only low-risk auto-execution (default off).
    auto_execute_low_risk: bool = False
    idea_ttl_s: float = 7 * 24 * 3600.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "allowed_sources": list(self.allowed_sources),
            "quiet_start": self.quiet_start,
            "quiet_end": self.quiet_end,
            "cooldown_s": self.cooldown_s,
            "max_per_period": self.max_per_period,
            "period_s": self.period_s,
            "relevance_threshold": self.relevance_threshold,
            "auto_execute_low_risk": (
                self.auto_execute_low_risk
            ),
            "idea_ttl_s": self.idea_ttl_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> (
        "ProactiveConfig"
    ):
        return cls(
            enabled=bool(data.get("enabled", True)),
            allowed_sources=tuple(
                data.get("allowed_sources") or (
                    "goals", "tasks", "schedules",
                    "memories", "activity",
                )
            ),
            quiet_start=str(data.get("quiet_start", "")),
            quiet_end=str(data.get("quiet_end", "")),
            cooldown_s=float(
                data.get("cooldown_s", 6 * 3600.0)
            ),
            max_per_period=int(
                data.get("max_per_period", 5)
            ),
            period_s=float(
                data.get("period_s", 24 * 3600.0)
            ),
            relevance_threshold=float(
                data.get("relevance_threshold", 0.5)
            ),
            auto_execute_low_risk=bool(
                data.get("auto_execute_low_risk", False)
            ),
            idea_ttl_s=float(
                data.get("idea_ttl_s", 7 * 24 * 3600.0)
            ),
        )

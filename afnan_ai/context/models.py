"""Context & trajectory data models.

Everything here is serializable metadata — summaries, tags,
timestamps and trust-zone labels.  Raw secrets never enter
these models (the shared redactor is applied at the
recording boundary in the manager), and untrusted external
content is always labeled with its trust zone, never mixed
into trusted instructions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrustZone(str, Enum):
    """Where a piece of context came from.

    * ``system`` — the agent's own built-in instructions.
    * ``user`` — the user's goal and direct input (trusted).
    * ``agent_state`` — the agent's own verified records
      (completed steps, decisions, facts).
    * ``tool_observation`` — structured tool/browser/connector
      outputs (data, not instructions).
    * ``untrusted_external`` — webpage text, emails, documents,
      connector payloads: content only, never instructions.
    """

    SYSTEM = "system"
    USER = "user"
    AGENT_STATE = "agent_state"
    TOOL_OBSERVATION = "tool_observation"
    UNTRUSTED_EXTERNAL = "untrusted_external"


class ItemKind(str, Enum):
    GOAL = "goal"
    SUBGOAL = "subgoal"
    OBSERVATION = "observation"
    DECISION = "decision"
    ACTION = "action"
    RESULT = "result"
    VERIFICATION = "verification"
    FAILURE = "failure"
    RECOVERY = "recovery"
    APPROVAL = "approval"
    FACT = "fact"
    CONSTRAINT = "constraint"
    CHECKPOINT = "checkpoint"
    SUMMARY = "summary"


@dataclass
class ContextItem:
    """One unit of working context."""

    kind: ItemKind
    zone: TrustZone
    text: str
    tags: tuple[str, ...] = ()
    importance: float = 0.5  # 0..1; facts/constraints/decisions rank high
    created_at: str = field(default_factory=_utcnow)
    # How many times this item was pulled into a cycle's
    # context (drives retention of useful history).
    hits: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            self.kind = ItemKind(self.kind)
        if isinstance(self.zone, str):
            self.zone = TrustZone(self.zone)
        self.tags = tuple(t.lower() for t in (self.tags or ()))
        self.importance = max(0.0, min(1.0, float(self.importance)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "zone": self.zone.value,
            "text": self.text,
            "tags": list(self.tags),
            "importance": self.importance,
            "created_at": self.created_at,
            "hits": self.hits,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextItem":
        item = cls(
            kind=data.get("kind", ItemKind.OBSERVATION.value),
            zone=data.get("zone", TrustZone.TOOL_OBSERVATION.value),
            text=str(data.get("text", "")),
            tags=tuple(data.get("tags") or ()),
            importance=float(data.get("importance", 0.5)),
        )
        item.created_at = str(
            data.get("created_at", item.created_at)
        )
        item.hits = int(data.get("hits", 0))
        return item


@dataclass
class TrajectoryEntry:
    """One step of a task's execution trajectory.

    Mirrors the AgentLoop's own trajectory records with a
    trust zone attached, so a restarted task recovers not
    just *what* happened but *which* parts are trusted.
    """

    seq: int
    kind: str  # observation|decision|action|result|verification|
    # recovery|approval|checkpoint|goal|...
    zone: TrustZone
    summary: str
    step_id: str | None = None
    at: str = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        if isinstance(self.zone, str):
            self.zone = TrustZone(self.zone)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "zone": self.zone.value,
            "summary": self.summary,
            "step_id": self.step_id,
            "at": self.at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryEntry":
        return cls(
            seq=int(data.get("seq", 0)),
            kind=str(data.get("kind", "")),
            zone=str(
                data.get("zone", TrustZone.TOOL_OBSERVATION.value)
            ),
            summary=str(data.get("summary", "")),
            step_id=data.get("step_id"),
            at=str(data.get("at", _utcnow())),
        )


class SubGoalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubGoal:
    subgoal_id: str
    description: str
    status: SubGoalStatus = SubGoalStatus.PENDING
    evidence: str = ""
    created_cycle: int = 0
    finished_cycle: int | None = None
    # Sub-goal ids that must be verified-complete first.
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = SubGoalStatus(self.status)
        self.depends_on = tuple(self.depends_on or ())

    def to_dict(self) -> dict[str, Any]:
        return {
            "subgoal_id": self.subgoal_id,
            "description": self.description,
            "status": self.status.value,
            "evidence": self.evidence,
            "created_cycle": self.created_cycle,
            "finished_cycle": self.finished_cycle,
            "depends_on": list(self.depends_on),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubGoal":
        return cls(
            subgoal_id=str(data.get("subgoal_id", "")),
            description=str(data.get("description", "")),
            status=str(
                data.get("status", SubGoalStatus.PENDING.value)
            ),
            evidence=str(data.get("evidence", "")),
            created_cycle=int(data.get("created_cycle", 0)),
            finished_cycle=data.get("finished_cycle"),
            depends_on=tuple(data.get("depends_on") or ()),
        )


@dataclass
class ContextBudget:
    """Configurable limits so long tasks cannot exhaust
    memory or the model context."""

    #: Hard cap on the zoned context section (characters).
    max_chars: int = 4000
    #: Recent detailed items kept verbatim before compression.
    keep_recent_detailed: int = 8
    #: Compress once the working set exceeds this many items.
    summarize_after_items: int = 24
    #: Max context items retained in memory per run.
    max_items: int = 120
    #: Max history items pulled into one cycle's context.
    retrieval_limit: int = 10
    #: Max trajectory entries kept per task.
    max_trajectory_entries: int = 500
    #: Max finished tasks retained in the trajectory store.
    retention_tasks: int = 50

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_chars": self.max_chars,
            "keep_recent_detailed": self.keep_recent_detailed,
            "summarize_after_items": self.summarize_after_items,
            "max_items": self.max_items,
            "retrieval_limit": self.retrieval_limit,
            "max_trajectory_entries": self.max_trajectory_entries,
            "retention_tasks": self.retention_tasks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextBudget":
        budget = cls()
        for key, value in (data or {}).items():
            if hasattr(budget, key):
                setattr(budget, key, int(value))
        return budget


@dataclass
class ContextSnapshot:
    """Serializable ContextManager state for checkpoints."""

    goal: str = ""
    goal_id: str | None = None
    task_id: str | None = None
    cycle: int = 0
    items: list[dict[str, Any]] = field(default_factory=list)
    subgoals: list[dict[str, Any]] = field(default_factory=list)
    facts: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed_actions: list[str] = field(default_factory=list)
    failed_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "goal_id": self.goal_id,
            "task_id": self.task_id,
            "cycle": self.cycle,
            "items": list(self.items),
            "subgoals": list(self.subgoals),
            "facts": dict(self.facts),
            "completed_actions": list(self.completed_actions),
            "failed_actions": list(self.failed_actions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextSnapshot":
        return cls(
            goal=str(data.get("goal", "")),
            goal_id=data.get("goal_id"),
            task_id=data.get("task_id"),
            cycle=int(data.get("cycle", 0)),
            items=list(data.get("items") or []),
            subgoals=list(data.get("subgoals") or []),
            facts=dict(data.get("facts") or {}),
            completed_actions=list(
                data.get("completed_actions") or []
            ),
            failed_actions=list(data.get("failed_actions") or []),
        )


# ------------------------------------------------------------------
# Long-horizon extensions
# ------------------------------------------------------------------

class ContextTier(str, Enum):
    """How immediately an item is needed.

    * HOT — current task/step needs it now.
    * WARM — recent observations, actions, decisions.
    * COLD — historical data, older observations (on demand).
    * ARCHIVED — compressed long-term summaries (on demand).
    """

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    ARCHIVED = "archived"


class Sensitivity(str, Enum):
    """Who may see a context item.

    SECRET items are never rendered into prompts, logs,
    activity or trajectory plaintext — only CredentialVault
    references travel.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    SECRET = "secret"


class Freshness(str, Enum):
    """How trustworthy an observation still is."""

    FRESH = "fresh"
    POTENTIALLY_STALE = "potentially_stale"
    STALE = "stale"


class TrajectoryPhase(str, Enum):
    """Phase of one trajectory event."""

    OBSERVE = "observe"
    DECIDE = "decide"
    ACT = "act"
    VERIFY = "verify"
    RECOVER = "recover"
    CHECKPOINT = "checkpoint"
    PLAN = "plan"


def _context_id() -> str:
    from uuid import uuid4

    return "ctx-" + uuid4().hex[:12]


@dataclass
class ContextItemV2:
    """Extended context item for long-horizon tasks.

    A superset of ContextItem: adds source identity,
    confidence, relevance, freshness, task/workspace scope,
    sensitivity, expiration and provenance.  Existing code
    keeps using ContextItem; long-horizon features use this.
    """

    kind: ItemKind
    zone: TrustZone
    text: str
    context_id: str = field(default_factory=_context_id)
    source: str = ""  # e.g. "memory", "browser", "computer"
    confidence: float = 0.7  # 0..1
    relevance: float = 0.5  # 0..1, recomputed per retrieval
    freshness: Freshness = Freshness.FRESH
    task_id: str = ""
    workspace_id: str = ""
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    expires_at: str = ""  # ISO timestamp, "" = no expiry
    provenance: str = ""  # where exactly this came from
    tags: tuple[str, ...] = ()
    importance: float = 0.5
    created_at: str = field(default_factory=_utcnow)
    hits: int = 0
    references: tuple[str, ...] = ()  # artifact/event ids

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            self.kind = ItemKind(self.kind)
        if isinstance(self.zone, str):
            self.zone = TrustZone(self.zone)
        if isinstance(self.freshness, str):
            self.freshness = Freshness(self.freshness)
        if isinstance(self.sensitivity, str):
            self.sensitivity = Sensitivity(self.sensitivity)
        self.tags = tuple(t.lower() for t in (self.tags or ()))
        self.references = tuple(self.references or ())
        self.importance = max(0.0, min(1.0, float(self.importance)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.relevance = max(0.0, min(1.0, float(self.relevance)))

    def is_expired(self, now: str | None = None) -> bool:
        if not self.expires_at:
            return False
        check = now or _utcnow()
        return check >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "kind": self.kind.value,
            "zone": self.zone.value,
            "text": self.text,
            "source": self.source,
            "confidence": self.confidence,
            "relevance": self.relevance,
            "freshness": self.freshness.value,
            "task_id": self.task_id,
            "workspace_id": self.workspace_id,
            "sensitivity": self.sensitivity.value,
            "expires_at": self.expires_at,
            "provenance": self.provenance,
            "tags": list(self.tags),
            "importance": self.importance,
            "created_at": self.created_at,
            "hits": self.hits,
            "references": list(self.references),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextItemV2":
        item = cls(
            kind=data.get("kind", ItemKind.OBSERVATION.value),
            zone=data.get(
                "zone", TrustZone.TOOL_OBSERVATION.value
            ),
            text=str(data.get("text", "")),
            context_id=str(
                data.get("context_id", _context_id())
            ),
            source=str(data.get("source", "")),
            confidence=float(data.get("confidence", 0.7)),
            relevance=float(data.get("relevance", 0.5)),
            freshness=str(
                data.get("freshness", Freshness.FRESH.value)
            ),
            task_id=str(data.get("task_id", "")),
            workspace_id=str(data.get("workspace_id", "")),
            sensitivity=str(
                data.get(
                    "sensitivity",
                    Sensitivity.INTERNAL.value,
                )
            ),
            expires_at=str(data.get("expires_at", "")),
            provenance=str(data.get("provenance", "")),
            tags=tuple(data.get("tags") or ()),
            importance=float(data.get("importance", 0.5)),
            references=tuple(data.get("references") or ()),
        )
        item.created_at = str(
            data.get("created_at", item.created_at)
        )
        item.hits = int(data.get("hits", 0))
        return item


@dataclass
class TrajectoryEvent:
    """One structured long-horizon trajectory event.

    An execution record — goal, observations, decisions,
    actions, results, verifications, failures, recoveries,
    approvals, checkpoints, artifacts, state transitions —
    never a hidden reasoning transcript.
    """

    event_id: str = field(default_factory=_context_id)
    task_id: str = ""
    phase: TrajectoryPhase = TrajectoryPhase.OBSERVE
    kind: str = "observation"  # observation|decision|action|
    # result|verification|failure|recovery|approval|
    # checkpoint|artifact|state_transition
    zone: TrustZone = TrustZone.TOOL_OBSERVATION
    summary: str = ""
    action_ref: str = ""
    observation_ref: str = ""
    result: str = ""
    verified: bool | None = None
    confidence: float = 0.7
    failure: str = ""
    recovery: str = ""
    artifact_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        if isinstance(self.phase, str):
            self.phase = TrajectoryPhase(self.phase)
        if isinstance(self.zone, str):
            self.zone = TrustZone(self.zone)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "task_id": self.task_id,
            "phase": self.phase.value,
            "kind": self.kind,
            "zone": self.zone.value,
            "summary": self.summary,
            "action_ref": self.action_ref,
            "observation_ref": self.observation_ref,
            "result": self.result,
            "verified": self.verified,
            "confidence": self.confidence,
            "failure": self.failure,
            "recovery": self.recovery,
            "artifact_ref": self.artifact_ref,
            "metadata": dict(self.metadata or {}),
            "at": self.at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryEvent":
        return cls(
            event_id=str(
                data.get("event_id", _context_id())
            ),
            task_id=str(data.get("task_id", "")),
            phase=str(
                data.get("phase", TrajectoryPhase.OBSERVE.value)
            ),
            kind=str(data.get("kind", "observation")),
            zone=str(
                data.get(
                    "zone", TrustZone.TOOL_OBSERVATION.value
                )
            ),
            summary=str(data.get("summary", "")),
            action_ref=str(data.get("action_ref", "")),
            observation_ref=str(
                data.get("observation_ref", "")
            ),
            result=str(data.get("result", "")),
            verified=data.get("verified"),
            confidence=float(data.get("confidence", 0.7)),
            failure=str(data.get("failure", "")),
            recovery=str(data.get("recovery", "")),
            artifact_ref=str(data.get("artifact_ref", "")),
            metadata=dict(data.get("metadata") or {}),
            at=str(data.get("at", _utcnow())),
        )

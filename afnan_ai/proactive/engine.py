"""ProactiveEngine — controlled intelligence layer.

Detect Opportunity → Generate Suggestion → Risk/Permission
Check → User Notification → User Accepts → Create Task →
AgentLoop Execute → Verify → Complete.

The engine never executes actions itself and never
duplicates the AgentLoop: it produces Ideas, and only
after explicit user acceptance (or a configured,
read-only, low-risk auto-action) does it enqueue a normal
task for the existing TaskManager → AgentLoop path.
Sensitive/irreversible suggestions always need human
approval.

Timing intelligence: per-opportunity cooldown,
signature dedup, relevance threshold, quiet hours, and a
cap on suggestions per period.  Goal awareness: ideas
linked to active goals get a priority boost.

Privacy: suggestion text is secret-redacted; the user
controls enable/disable and which sources the engine may
read.  Feedback (accepted/dismissed/ignored/completed/
failed) tunes future ranking — it never writes inferred
preferences into memory.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from afnan_ai.log_config import get_logger
from afnan_ai.proactive.detectors import (
    DetectionContext,
    run_detectors,
)
from afnan_ai.proactive.models import (
    RISKS_REQUIRING_APPROVAL,
    RISK_READ_ONLY,
    Idea,
    IdeaStatus,
    ProactiveConfig,
    SuggestionType,
)
from afnan_ai.redaction import redact_text

logger = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ProactiveError(Exception):
    def __init__(self, message: str, code: str = "proactive_error",
                 details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class ProactiveEngine:
    """Generates, ranks and routes proactive ideas."""

    def __init__(
        self,
        *,
        memory_store=None,
        goal_manager=None,
        task_manager=None,
        scheduler=None,
        config: ProactiveConfig | None = None,
        store_path: str | Path | None = None,
        approver: Callable[[str], bool] | None = None,
        task_runner: Callable[..., Any] | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.goal_manager = goal_manager
        self.task_manager = task_manager
        self.scheduler = scheduler
        self.config = config or ProactiveConfig()
        self.approver = approver
        self.task_runner = task_runner
        self._store_path = (
            Path(str(store_path)).expanduser()
            if store_path else None
        )
        self._ideas: dict[str, Idea] = {}
        self._lock = threading.RLock()
        self._last_seen: dict[str, str] = {}  # sig -> iso
        self._notified_at: list[str] = []
        self._feedback_scores: dict[str, float] = {}
        self._load()

    # -- persistence ----------------------------------------------------
    def _load(self) -> None:
        if self._store_path is None or not self._store_path.exists():
            return
        try:
            data = json.loads(self._store_path.read_text())
            for raw in data.get("ideas", []):
                idea = Idea.from_dict(raw)
                if not idea.is_expired():
                    self._ideas[idea.idea_id] = idea
            self._last_seen = dict(data.get("last_seen", {}))
            self._notified_at = list(
                data.get("notified_at", [])
            )
        except Exception as e:  # noqa: BLE001 - degrade
            logger.warning("proactive store load failed: %s", e)

    def _save(self) -> None:
        if self._store_path is None:
            return
        try:
            self._store_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            payload = {
                "ideas": [
                    i.to_dict() for i in self._ideas.values()
                ],
                "last_seen": self._last_seen,
                "notified_at": self._notified_at[-50:],
            }
            tmp = self._store_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=1))
            tmp.replace(self._store_path)
        except Exception as e:  # noqa: BLE001 - degrade
            logger.warning("proactive store save failed: %s", e)

    # -- detection ------------------------------------------------------
    def _snapshot(self) -> DetectionContext:
        allowed = self.config.allowed_sources

        def _safe(fn, source):
            if source not in allowed:
                return []
            try:
                return fn() or []
            except Exception:
                return []

        goals = _safe(
            lambda: self.goal_manager.list(status="active")
            if self.goal_manager else [],
            "goals",
        )
        tasks = _safe(
            lambda: self.task_manager.list()
            if self.task_manager else [],
            "tasks",
        )
        schedules: list = []
        if "schedules" in allowed and self.scheduler:
            try:
                list_fn = getattr(
                    self.scheduler, "list_schedules",
                    None,
                ) or self.scheduler.list
                schedules = [
                    s for s in list_fn()
                    if getattr(s, "status", "") == "active"
                ]
            except Exception:
                schedules = []
        memories = _safe(
            lambda: self.memory_store.list()
            if self.memory_store else [],
            "memories",
        )
        return DetectionContext(
            goals=goals, tasks=tasks, schedules=schedules,
            memories=memories, allowed_sources=allowed,
        )

    def evaluate(self) -> list[Idea]:
        """Run one sweep; returns the new ideas surfaced."""
        if not self.config.enabled:
            return []
        ctx = self._snapshot()
        fresh: list[Idea] = []
        now = _utcnow()
        with self._lock:
            for cand in run_detectors(ctx):
                if (
                    cand["confidence"]
                    < self.config.relevance_threshold
                ):
                    continue  # irrelevant → rejected
                idea = Idea(
                    idea_id=Idea.new_id(),
                    title=cand["title"],
                    description=cand["description"],
                    reason=cand["reason"],
                    suggestion_type=cand["suggestion_type"],
                    confidence=cand["confidence"],
                    priority=self._prioritize(cand),
                    suggested_action=cand["suggested_action"],
                    risk_level=cand["risk_level"],
                    related_goal_id=cand["related_goal_id"],
                    related_task_id=cand["related_task_id"],
                    evidence=cand["evidence"],
                    expires_at=(
                        now + timedelta(
                            seconds=self.config.idea_ttl_s
                        )
                    ).isoformat(),
                )
                sig = idea.signature()
                last = self._last_seen.get(sig)
                if last:
                    try:
                        seen = datetime.fromisoformat(last)
                        if (
                            now - seen
                        ).total_seconds() < self.config.cooldown_s:
                            continue  # cooldown
                    except ValueError:
                        pass
                if any(
                    i.signature() == sig
                    and i.status is IdeaStatus.NEW
                    for i in self._ideas.values()
                ):
                    continue  # duplicate
                self._ideas[idea.idea_id] = idea
                self._last_seen[sig] = now.isoformat()
                fresh.append(idea)
            self._save()
        return sorted(
            fresh, key=lambda i: (-i.priority, -i.confidence)
        )

    def _prioritize(self, cand: dict[str, Any]) -> int:
        priority = int(cand.get("priority", 3))
        # Goal awareness: ideas tied to an active goal rank
        # higher — the user's stated goals outrank guesses.
        goal_id = cand.get("related_goal_id", "")
        if goal_id and self.goal_manager:
            try:
                goal = self.goal_manager.get(goal_id)
                if goal and getattr(
                    goal, "status", ""
                ) == "active":
                    priority = min(5, priority + 1)
            except Exception:
                pass
        # Feedback tuning: dismissed signatures sink.
        sig_score = self._feedback_scores.get(
            str(cand.get("suggestion_type", "")), 0.0
        )
        if sig_score < -1.0:
            priority = max(1, priority - 1)
        return max(1, min(5, priority))

    # -- notification surface -------------------------------------------
    def _in_quiet_hours(self, now: datetime) -> bool:
        start, end = (
            self.config.quiet_start.strip(),
            self.config.quiet_end.strip(),
        )
        if not start or not end:
            return False
        try:
            sh, sm = map(int, start.split(":"))
            eh, em = map(int, end.split(":"))
        except ValueError:
            return False
        cur = now.hour * 60 + now.minute
        begin, finish = sh * 60 + sm, eh * 60 + em
        if begin <= finish:
            return begin <= cur < finish
        return cur >= begin or cur < finish

    def pending_notifications(
        self, now: datetime | None = None
    ) -> list[Idea]:
        """Ideas ready to show: new, unexpired, within quota,
        outside quiet hours."""
        moment = now or _utcnow()
        if not self.config.enabled:
            return []
        if self._in_quiet_hours(moment):
            return []
        with self._lock:
            cutoff = (
                moment - timedelta(
                    seconds=self.config.period_s
                )
            ).isoformat()
            recent = sum(
                1 for t in self._notified_at if t >= cutoff
            )
            room = max(
                0, self.config.max_per_period - recent
            )
            out = [
                i for i in self._ideas.values()
                if i.status is IdeaStatus.NEW
                and not i.is_expired(moment.isoformat())
            ]
            out.sort(
                key=lambda i: (-i.priority, -i.confidence)
            )
            chosen = out[:room]
            for idea in chosen:
                self._notified_at.append(moment.isoformat())
            if chosen:
                self._save()
            return chosen

    def get_idea(self, idea_id: str) -> Idea:
        idea = self._ideas.get(idea_id)
        if idea is None:
            raise ProactiveError(
                f"unknown idea {idea_id!r}",
                code="unknown_idea",
            )
        return idea

    def list_ideas(
        self, *, status: IdeaStatus | None = None
    ) -> list[Idea]:
        ideas = list(self._ideas.values())
        if status is not None:
            ideas = [
                i for i in ideas if i.status is status
            ]
        return sorted(
            ideas, key=lambda i: i.created_at, reverse=True
        )

    # -- user decisions ---------------------------------------------------
    def dismiss(
        self, idea_id: str, reason: str = ""
    ) -> Idea:
        with self._lock:
            idea = self.get_idea(idea_id)
            idea.status = IdeaStatus.DISMISSED
            self._record_feedback(
                idea, "dismissed", {"reason": reason[:200]}
            )
            self._save()
            return idea

    def accept(self, idea_id: str) -> Idea:
        """User accepts → permission check → task created →
        AgentLoop executes (engine never runs it directly)."""
        with self._lock:
            idea = self.get_idea(idea_id)
            if idea.status is not IdeaStatus.NEW:
                raise ProactiveError(
                    f"idea {idea_id!r} is {idea.status.value}",
                    code="bad_state",
                )
            if (
                idea.risk_level
                in RISKS_REQUIRING_APPROVAL
            ):
                self._require_approval(idea)
            task = self._enqueue_task(idea)
            idea.status = IdeaStatus.ACCEPTED
            idea.suggested_action.setdefault(
                "params", {}
            )["task_id"] = task.task_id
            self._record_feedback(idea, "accepted", {})
            auto = (
                self.config.auto_execute_low_risk
                and idea.risk_level == RISK_READ_ONLY
                and self.task_runner is not None
            )
            self._save()
        if auto:
            # Read-only low-risk auto-action, explicitly
            # configured by the user.
            try:
                self.task_runner(task.task_id)
                idea.status = IdeaStatus.COMPLETED
            except Exception as e:  # noqa: BLE001
                idea.status = IdeaStatus.FAILED
                self._record_feedback(
                    idea, "failed", {"error": str(e)[:200]}
                )
            with self._lock:
                self._save()
        return idea

    def _require_approval(self, idea: Idea) -> None:
        request = (
            f"Proactive suggestion '{idea.title}': "
            f"{idea.description[:200]} "
            f"(risk={idea.risk_level}). Proceed?"
        )
        if self.approver is None:
            raise ProactiveError(
                "sensitive proactive action needs human "
                "approval; no approver configured",
                code="approval_required",
                details={"idea_id": idea.idea_id,
                         "resumable": True},
            )
        try:
            allowed = self.approver(redact_text(request))
        except Exception:
            allowed = False
        if not allowed:
            raise ProactiveError(
                "human denied the proactive action",
                code="approval_denied",
                details={"idea_id": idea.idea_id},
            )

    def _enqueue_task(self, idea: Idea):
        if self.task_manager is None:
            raise ProactiveError(
                "no task manager wired",
                code="no_task_manager",
            )
        action = idea.suggested_action or {}
        kind = action.get("kind", "assist")
        goal_text = (
            f"[idea:{idea.idea_id}] {kind}: {idea.title}"
        )
        return self.task_manager.enqueue(
            redact_text(goal_text)[:300],
            goal_id=idea.related_goal_id,
            priority=idea.priority,
            metadata={
                "idea_id": idea.idea_id,
                "suggestion_type": str(
                    idea.suggestion_type
                ),
                "action": action,
            },
        )

    def set_approver(
        self, approver: Callable[[str], bool] | None
    ) -> None:
        self.approver = approver

    # -- feedback loop ----------------------------------------------------
    def _record_feedback(
        self, idea: Idea, outcome: str,
        extra: dict[str, Any],
    ) -> None:
        idea.feedback.append({
            "outcome": outcome,
            "at": _utcnow().isoformat(),
            **extra,
        })
        key = str(idea.suggestion_type)
        delta = {
            "accepted": 0.5, "completed": 1.0,
            "dismissed": -1.0, "ignored": -0.5,
            "failed": -0.5,
        }.get(outcome, 0.0)
        self._feedback_scores[key] = (
            self._feedback_scores.get(key, 0.0) + delta
        )
        # Feedback tunes ranking only — inferred
        # preferences are never written to memory.

    def record_feedback(
        self, idea_id: str, outcome: str
    ) -> Idea:
        if outcome not in (
            "accepted", "dismissed", "ignored",
            "completed", "failed",
        ):
            raise ProactiveError(
                f"unknown feedback {outcome!r}",
                code="bad_feedback",
            )
        with self._lock:
            idea = self.get_idea(idea_id)
            self._record_feedback(idea, outcome, {})
            if outcome == "completed":
                idea.status = IdeaStatus.COMPLETED
            elif outcome == "failed":
                idea.status = IdeaStatus.FAILED
            self._save()
            return idea

    # -- background ---------------------------------------------------------
    def run_sweep(self) -> list[Idea]:
        """Periodic background evaluation: detect, expire,
        and queue — never execute."""
        with self._lock:
            now = _utcnow().isoformat()
            for idea in self._ideas.values():
                if (
                    idea.status is IdeaStatus.NEW
                    and idea.is_expired(now)
                ):
                    idea.status = IdeaStatus.EXPIRED
            self._save()
        return self.evaluate()

    # -- user control ---------------------------------------------------------
    def update_config(
        self, **kwargs: Any
    ) -> ProactiveConfig:
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        return self.config


__all__ = [
    "ProactiveEngine", "ProactiveError", "ProactiveConfig",
]

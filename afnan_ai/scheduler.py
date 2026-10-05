"""TaskScheduler — future and recurring tasks, platform-
independent (pure stdlib datetime math; no OS scheduler
dependency, so it behaves identically on Windows, Linux and
macOS).

A schedule is a durable template for a task: what to run,
when to first run it, and how it repeats (``once``,
``interval`` seconds, ``daily`` at a time of day, ``weekly``
on a weekday).  When a schedule comes due, :meth:`tick`
creates a normal pending task in the TaskManager — execution
itself always goes through the BackgroundTaskRunner and the
AgentLoop, so scheduled work inherits every approval, limit
and recovery protection.  The scheduler never executes
anything itself.

All times are UTC ISO-8601; naive datetimes are treated as
UTC.  Recurring schedules advance from their *scheduled*
time (never bursting to catch up missed occurrences).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from afnan_ai.persistence import JsonFileStore

RECURRENCES = frozenset({"once", "interval", "daily", "weekly"})
SCHEDULE_STATUSES = frozenset({
    "active", "paused", "completed", "cancelled",
})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def _parse(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        moment = value
    else:
        moment = datetime.fromisoformat(str(value))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


class ScheduleError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ScheduledTask:
    schedule_id: str
    goal_text: str
    run_at: str  # first (or next anchor) fire time, UTC ISO
    recurrence: str = "once"
    interval_seconds: int | None = None
    time_of_day: str | None = None  # "HH:MM"
    weekday: int | None = None  # 0=Monday .. 6=Sunday
    goal_id: str = ""
    priority: int = 3
    timeout_s: float | None = None
    max_retries: int = 2
    status: str = "active"
    next_run_at: str | None = None
    last_run_at: str | None = None
    last_task_id: str = ""
    run_count: int = 0
    created_at: str = field(default_factory=lambda: _iso(_now()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "goal_text": self.goal_text,
            "run_at": self.run_at,
            "recurrence": self.recurrence,
            "interval_seconds": self.interval_seconds,
            "time_of_day": self.time_of_day,
            "weekday": self.weekday,
            "goal_id": self.goal_id,
            "priority": self.priority,
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
            "status": self.status,
            "next_run_at": self.next_run_at,
            "last_run_at": self.last_run_at,
            "last_task_id": self.last_task_id,
            "run_count": self.run_count,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduledTask":
        return cls(
            schedule_id=str(data.get("schedule_id", "")),
            goal_text=str(data.get("goal_text", "")),
            run_at=str(data.get("run_at", "")),
            recurrence=str(data.get("recurrence", "once")),
            interval_seconds=data.get("interval_seconds"),
            time_of_day=data.get("time_of_day"),
            weekday=data.get("weekday"),
            goal_id=str(data.get("goal_id", "")),
            priority=int(data.get("priority", 3)),
            timeout_s=data.get("timeout_s"),
            max_retries=int(data.get("max_retries", 2)),
            status=str(data.get("status", "active")),
            next_run_at=data.get("next_run_at"),
            last_run_at=data.get("last_run_at"),
            last_task_id=str(data.get("last_task_id", "")),
            run_count=int(data.get("run_count", 0)),
            created_at=str(
                data.get("created_at") or _iso(_now())
            ),
            metadata=dict(data.get("metadata") or {}),
        )


def _apply_time_of_day(moment: datetime, hhmm: str) -> datetime:
    try:
        hour, minute = (int(p) for p in hhmm.split(":", 1))
    except (ValueError, TypeError):
        raise ScheduleError(
            "invalid_time_of_day",
            f"time_of_day must be 'HH:MM', got {hhmm!r}",
        )
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ScheduleError(
            "invalid_time_of_day",
            f"time_of_day out of range: {hhmm!r}",
        )
    return moment.replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )


def compute_next_run(
    schedule: ScheduledTask, *, after: datetime
) -> datetime | None:
    """The next fire time strictly after ``after``."""
    if schedule.recurrence == "once":
        return None
    if schedule.recurrence == "interval":
        if not schedule.interval_seconds:
            raise ScheduleError(
                "invalid_interval",
                "interval recurrence needs interval_seconds",
            )
        step = timedelta(seconds=int(schedule.interval_seconds))
        candidate = _parse(schedule.run_at)
        while candidate <= after:
            candidate += step
        return candidate
    if schedule.recurrence == "daily":
        anchor = (
            _apply_time_of_day(after, schedule.time_of_day)
            if schedule.time_of_day
            else _parse(schedule.run_at) + timedelta(days=1)
        )
        candidate = anchor
        if schedule.time_of_day:
            if candidate <= after:
                candidate += timedelta(days=1)
        else:
            base = _parse(schedule.run_at)
            candidate = base
            while candidate <= after:
                candidate += timedelta(days=1)
        return candidate
    if schedule.recurrence == "weekly":
        wanted = (
            int(schedule.weekday)
            if schedule.weekday is not None
            else _parse(schedule.run_at).weekday()
        )
        candidate = after + timedelta(
            days=(wanted - after.weekday()) % 7
        )
        if schedule.time_of_day:
            candidate = _apply_time_of_day(
                candidate, schedule.time_of_day
            )
        if candidate <= after:
            candidate += timedelta(days=7)
        return candidate
    raise ScheduleError(
        "invalid_recurrence",
        f"Unknown recurrence {schedule.recurrence!r}",
    )


class TaskScheduler:
    """Durable schedule registry over a local JSON store."""

    def __init__(self, path: str, task_manager: Any):
        self._store = JsonFileStore(path)
        self.task_manager = task_manager
        self._schedules: dict[str, ScheduledTask] | None = None

    # -- storage ---------------------------------------------------
    def _load(self) -> dict[str, ScheduledTask]:
        if self._schedules is None:
            data = self._store.read(
                {"version": 1, "schedules": []}
            )
            schedules: dict[str, ScheduledTask] = {}
            for item in data.get("schedules") or []:
                try:
                    schedule = ScheduledTask.from_dict(item)
                except Exception:
                    continue
                if schedule.schedule_id:
                    schedules[schedule.schedule_id] = schedule
            self._schedules = schedules
        return self._schedules

    def _save(self) -> None:
        self._store.write({
            "version": 1,
            "schedules": [
                s.to_dict() for s in self._load().values()
            ],
        })

    def _require(self, schedule_id: str) -> ScheduledTask:
        schedule = self._load().get(schedule_id)
        if schedule is None:
            raise ScheduleError(
                "not_found", f"No schedule {schedule_id!r}"
            )
        return schedule

    # -- registry -----------------------------------------------------
    def schedule_task(
        self,
        goal_text: str,
        *,
        run_at: datetime | str | None = None,
        delay_s: float | None = None,
        recurrence: str = "once",
        interval_seconds: int | None = None,
        time_of_day: str | None = None,
        weekday: int | None = None,
        goal_id: str = "",
        priority: int = 3,
        timeout_s: float | None = None,
        max_retries: int = 2,
        metadata: dict[str, Any] | None = None,
    ) -> ScheduledTask:
        goal_text = str(goal_text or "").strip()
        if not goal_text:
            raise ScheduleError(
                "empty_goal", "A scheduled task needs a goal"
            )
        if recurrence not in RECURRENCES:
            raise ScheduleError(
                "invalid_recurrence",
                f"recurrence must be one of "
                f"{sorted(RECURRENCES)}, got {recurrence!r}",
            )
        start = _now()
        if run_at is not None:
            first = _parse(run_at)
        elif delay_s is not None:
            first = start + timedelta(seconds=float(delay_s))
        elif recurrence in ("daily", "weekly") and time_of_day:
            first = _apply_time_of_day(start, time_of_day)
            if recurrence == "weekly" and weekday is not None:
                first += timedelta(
                    days=(int(weekday) - first.weekday()) % 7
                )
            if first <= start:
                first += timedelta(
                    days=1 if recurrence == "daily" else 7
                )
        else:
            first = start
        schedule = ScheduledTask(
            schedule_id=f"sched_{uuid.uuid4().hex[:12]}",
            goal_text=goal_text,
            run_at=_iso(first),
            recurrence=recurrence,
            interval_seconds=(
                int(interval_seconds)
                if interval_seconds is not None
                else None
            ),
            time_of_day=time_of_day,
            weekday=weekday,
            goal_id=str(goal_id or ""),
            priority=max(1, min(5, int(priority))),
            timeout_s=timeout_s,
            max_retries=max(0, int(max_retries)),
            next_run_at=_iso(first),
            metadata=dict(metadata or {}),
        )
        # Validate recurrence parameters eagerly.
        if recurrence != "once":
            compute_next_run(schedule, after=first)
        self._load()[schedule.schedule_id] = schedule
        self._save()
        return schedule

    def get(self, schedule_id: str) -> ScheduledTask | None:
        return self._load().get(schedule_id)

    def list(
        self, *, status: str | None = None
    ) -> list[ScheduledTask]:
        schedules = list(self._load().values())
        if status is not None:
            schedules = [
                s for s in schedules if s.status == status
            ]
        return sorted(
            schedules, key=lambda s: s.next_run_at or s.run_at
        )

    def pause(self, schedule_id: str) -> ScheduledTask:
        schedule = self._require(schedule_id)
        schedule.status = "paused"
        self._save()
        return schedule

    def resume(self, schedule_id: str) -> ScheduledTask:
        schedule = self._require(schedule_id)
        if schedule.status == "paused":
            schedule.status = "active"
            self._save()
        return schedule

    def cancel(self, schedule_id: str) -> ScheduledTask:
        schedule = self._require(schedule_id)
        schedule.status = "cancelled"
        schedule.next_run_at = None
        self._save()
        return schedule

    # -- firing --------------------------------------------------------
    def due(
        self, *, now: datetime | None = None
    ) -> list[ScheduledTask]:
        moment = now or _now()
        due = []
        for schedule in self._load().values():
            if schedule.status != "active":
                continue
            if schedule.next_run_at is None:
                continue
            if _parse(schedule.next_run_at) <= moment:
                due.append(schedule)
        return sorted(due, key=lambda s: s.next_run_at or "")

    def tick(self, *, now: datetime | None = None) -> list[Any]:
        """Create pending tasks for everything due.

        Returns the created ManagedTask list.  Each due
        schedule advances (or completes) so a repeated tick
        never creates duplicates.
        """
        moment = now or _now()
        created = []
        for schedule in self.due(now=moment):
            task = self.task_manager.enqueue(
                schedule.goal_text,
                goal_id=schedule.goal_id,
                priority=schedule.priority,
                max_retries=schedule.max_retries,
                timeout_s=schedule.timeout_s,
                metadata={
                    "schedule_id": schedule.schedule_id,
                    "scheduled_run_at": schedule.next_run_at,
                },
            )
            schedule.last_run_at = _iso(moment)
            schedule.last_task_id = task.task_id
            schedule.run_count += 1
            followup = compute_next_run(schedule, after=moment)
            if followup is None:
                schedule.status = "completed"
                schedule.next_run_at = None
            else:
                schedule.next_run_at = _iso(followup)
            created.append(task)
        if created:
            self._save()
        return created

"""Centralized, serializable AgentState for Afnan AI.

One :class:`AgentState` tracks a single task end to end:

- **goal** — what the task is trying to achieve
- **current_step** — the step running right now (or None)
- **completed_steps / failed_steps** — ordered step history
- **observations** — things the agent saw/heard (commands, screens, ...)
- **tool_results** — results returned by tools/apps the agent used
- **status** — pending / running / paused / completed / failed / cancelled

Design rules:

- **Platform independent** — this module only uses the Python standard
  library (``dataclasses``, ``json``, ``pathlib``, ``uuid``,
  ``datetime``).  It contains no Windows, macOS or Linux specific code,
  paths or line endings, so a state saved on one OS loads on another.
- **Serializable** — :meth:`AgentState.to_dict`, ``to_json``,
  ``save`` and the matching ``from_dict`` / ``from_json`` / ``load``
  round-trip losslessly.
- **Reusable** — the voice agent, the platform adapters, tests, a CLI
  or a future UI can all read/update the same object.  Nothing in
  here depends on speech, a microphone, or a specific platform
  adapter.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from afnan_ai.redaction import redact_text, redact_value


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string (timezone aware)."""
    return datetime.now(timezone.utc).isoformat()


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepRecord:
    """One step of a task, whether completed, failed or in between."""

    name: str
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepRecord":
        return cls(
            name=data["name"],
            status=StepStatus(data.get("status", StepStatus.PENDING.value)),
            result=data.get("result"),
            error=data.get("error"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class Observation:
    """Something the agent observed (a command, a screen, a sensor...)."""

    text: str
    source: str = "agent"
    timestamp: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        return cls(
            text=data["text"],
            source=data.get("source", "agent"),
            timestamp=data.get("timestamp") or _now_iso(),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ToolResult:
    """Result of one tool / app / action used while working on a task."""

    tool: str
    success: bool
    output: Any = None
    error: str | None = None
    timestamp: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "success": bool(self.success),
            "output": self.output,
            "error": self.error,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResult":
        return cls(
            tool=data["tool"],
            success=bool(data.get("success", False)),
            output=data.get("output"),
            error=data.get("error"),
            timestamp=data.get("timestamp") or _now_iso(),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class AgentState:
    """Central state for one task — the single source of truth.

    Example:
        >>> state = AgentState.create("Open Chrome and search for Python")
        >>> state.start_step("open chrome")
        >>> state.add_observation("User said: open chrome", source="microphone")
        >>> state.add_tool_result("chrome", success=True, output="opened")
        >>> state.complete_step("open chrome", result="opened")
        >>> state.complete_task()
        >>> state.status
        <TaskStatus.COMPLETED: 'completed'>
        >>> restored = AgentState.from_json(state.to_json())
        >>> restored.goal == state.goal
        True
    """

    goal: str
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: TaskStatus = TaskStatus.PENDING
    current_step: str | None = None
    completed_steps: list[StepRecord] = field(default_factory=list)
    failed_steps: list[StepRecord] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _running_step: StepRecord | None = field(default=None, repr=False, compare=False)

    # -- creation -------------------------------------------------------
    @classmethod
    def create(
        cls,
        goal: str,
        *,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "AgentState":
        if not goal or not str(goal).strip():
            raise ValueError("AgentState goal must not be empty")
        state = cls(goal=str(goal).strip(), metadata=dict(metadata or {}))
        if task_id:
            state.task_id = task_id
        return state

    # -- internal ---------------------------------------------------------
    def _touch(self) -> None:
        self.updated_at = _now_iso()

    # -- status -------------------------------------------------------------
    def set_status(self, status: TaskStatus | str) -> None:
        self.status = status if isinstance(status, TaskStatus) else TaskStatus(status)
        if self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            self.current_step = None
            self._running_step = None
        self._touch()

    def start_task(self) -> None:
        self.set_status(TaskStatus.RUNNING)

    def pause_task(self) -> None:
        self.set_status(TaskStatus.PAUSED)

    def complete_task(self, result: Any = None) -> None:
        if self._running_step is not None:
            self.complete_step(self._running_step.name, result=result)
        self.set_status(TaskStatus.COMPLETED)

    def fail_task(self, error: str) -> None:
        if self._running_step is not None:
            self.fail_step(self._running_step.name, error)
        self.error = str(error)
        self.set_status(TaskStatus.FAILED)

    def cancel_task(self) -> None:
        self.set_status(TaskStatus.CANCELLED)

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )

    @property
    def is_successful(self) -> bool:
        return self.status == TaskStatus.COMPLETED and not self.failed_steps

    # -- steps ----------------------------------------------------------------
    def start_step(self, name: str, *, metadata: dict[str, Any] | None = None) -> StepRecord:
        if not name or not str(name).strip():
            raise ValueError("step name must not be empty")
        # A previously running step that never finished is left in
        # history as-is; the new step becomes the current one.
        step = StepRecord(
            name=str(name).strip(),
            status=StepStatus.RUNNING,
            started_at=_now_iso(),
            metadata=dict(metadata or {}),
        )
        self._running_step = step
        self.current_step = step.name
        if self.status == TaskStatus.PENDING:
            self.status = TaskStatus.RUNNING
        self._touch()
        return step

    def complete_step(self, name: str, *, result: Any = None) -> StepRecord:
        step = self._finish_step(name, StepStatus.COMPLETED, result=result)
        self.completed_steps.append(step)
        return step

    def fail_step(self, name: str, error: str, *, result: Any = None) -> StepRecord:
        step = self._finish_step(name, StepStatus.FAILED, result=result, error=str(error))
        self.failed_steps.append(step)
        return step

    def _finish_step(
        self,
        name: str,
        status: StepStatus,
        *,
        result: Any = None,
        error: str | None = None,
    ) -> StepRecord:
        name = str(name).strip()
        if self._running_step is not None and self._running_step.name == name:
            step = self._running_step
        else:
            step = StepRecord(name=name, started_at=_now_iso())
        step.status = status
        step.result = result
        step.error = error
        step.finished_at = _now_iso()
        if self.current_step == name:
            self.current_step = None
        if self._running_step is not None and self._running_step.name == name:
            self._running_step = None
        self._touch()
        return step

    @property
    def all_steps(self) -> list[StepRecord]:
        steps = list(self.completed_steps) + list(self.failed_steps)
        if self._running_step is not None:
            steps.append(self._running_step)
        return steps

    # -- observations & tool results ----------------------------------------------
    def add_observation(
        self,
        text: str,
        *,
        source: str = "agent",
        metadata: dict[str, Any] | None = None,
    ) -> Observation:
        obs = Observation(text=str(text), source=source, metadata=dict(metadata or {}))
        self.observations.append(obs)
        self._touch()
        return obs

    def add_tool_result(
        self,
        tool: str,
        *,
        success: bool,
        output: Any = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        # Records are secrets-safe: outputs/errors are redacted
        # before they land in state (and therefore in summaries,
        # planner prompts and anything serialized from here).
        result = ToolResult(
            tool=str(tool),
            success=bool(success),
            output=redact_value(output),
            error=redact_text(error) if error else None,
            metadata=dict(metadata or {}),
        )
        self.tool_results.append(result)
        self._touch()
        return result

    # -- summary -----------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "status": self.status.value,
            "current_step": self.current_step,
            "completed": [s.name for s in self.completed_steps],
            "failed": [s.name for s in self.failed_steps],
            "observations": len(self.observations),
            "tool_results": len(self.tool_results),
            "error": self.error,
        }

    # -- serialization -------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "status": self.status.value,
            "current_step": self.current_step,
            "completed_steps": [s.to_dict() for s in self.completed_steps],
            "failed_steps": [s.to_dict() for s in self.failed_steps],
            "observations": [o.to_dict() for o in self.observations],
            "tool_results": [t.to_dict() for t in self.tool_results],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "metadata": dict(self.metadata),
            "running_step": self._running_step.to_dict() if self._running_step else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        state = cls(
            goal=data["goal"],
            task_id=data.get("task_id") or uuid.uuid4().hex,
            status=TaskStatus(data.get("status", TaskStatus.PENDING.value)),
            current_step=data.get("current_step"),
            completed_steps=[StepRecord.from_dict(s) for s in data.get("completed_steps", [])],
            failed_steps=[StepRecord.from_dict(s) for s in data.get("failed_steps", [])],
            observations=[Observation.from_dict(o) for o in data.get("observations", [])],
            tool_results=[ToolResult.from_dict(t) for t in data.get("tool_results", [])],
            created_at=data.get("created_at") or _now_iso(),
            updated_at=data.get("updated_at") or _now_iso(),
            error=data.get("error"),
            metadata=dict(data.get("metadata") or {}),
        )
        running = data.get("running_step")
        if running:
            state._running_step = StepRecord.from_dict(running)
        return state

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    @classmethod
    def from_json(cls, text: str) -> "AgentState":
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> Path:
        """Save to *path* (UTF-8 JSON). Works identically on every OS."""
        file_path = Path(path)
        if file_path.parent and str(file_path.parent) not in ("", "."):
            file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(self.to_json(), encoding="utf-8")
        return file_path

    @classmethod
    def load(cls, path: str | Path) -> "AgentState":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

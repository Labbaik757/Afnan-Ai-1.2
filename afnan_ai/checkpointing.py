"""Task checkpointing: crash-safe persistence of in-flight tasks.

A checkpoint captures everything needed to resume a task after a
browser crash, a process restart or a temporary failure:

* the goal and the current plan (arguments redacted),
* the full :class:`~afnan_ai.state.AgentState` (completed and
  failed steps, observations, recovery history),
* the iteration count, and
* an opaque ``extra`` dict for layer-specific state — the browser
  layer puts its session snapshot (tabs + purposes) there; this
  module itself knows nothing about browsers.

Safety properties:

* **Redaction before persistence** — the whole snapshot passes
  through :func:`afnan_ai.redaction.redact_value`, so passwords,
  tokens and cookies never reach the disk copy.
* **Integrity** — every file carries a SHA-256 checksum over its
  canonical payload; a corrupted or tampered checkpoint is
  rejected on load, never silently trusted.
* **Validation** — schema version and required fields are checked
  before any state is rebuilt.
* **Atomic writes** — checkpoints are written to a temporary file
  and renamed into place, so a crash mid-write cannot leave a
  half-valid checkpoint behind.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .log_config import get_logger
from .planner import TaskPlan
from .redaction import redact_value
from .state import AgentState

__all__ = ["CheckpointError", "CheckpointManager"]

logger = get_logger(__name__)

SCHEMA_VERSION = 1


class CheckpointError(Exception):
    """A checkpoint is missing, corrupted or fails validation."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CheckpointManager:
    """Saves and loads task checkpoints in a directory."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)

    # -- building ----------------------------------------------------------

    def build_snapshot(
        self,
        *,
        state: AgentState,
        plan: TaskPlan | None,
        goal: str,
        status: str,
        iteration: int,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """A redacted, checksum-protected snapshot dict."""
        snapshot: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "task_id": state.task_id,
            "goal": goal,
            "status": status,
            "iteration": int(iteration),
            "saved_at": _now_iso(),
            "state": state.to_dict(),
            "plan": plan.to_dict() if plan is not None else None,
            "recovery_attempts": list(
                state.metadata.get("recovery_attempts") or []
            ),
            "extra": dict(extra or {}),
        }
        snapshot = redact_value(snapshot)
        snapshot["checksum"] = self._checksum(snapshot)
        return snapshot

    # -- persistence -------------------------------------------------------

    def save_snapshot(
        self,
        *,
        state: AgentState,
        plan: TaskPlan | None,
        goal: str,
        status: str,
        iteration: int,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        snapshot = self.build_snapshot(
            state=state,
            plan=plan,
            goal=goal,
            status=status,
            iteration=iteration,
            extra=extra,
        )
        return self.save(snapshot)

    def save(self, snapshot: dict[str, Any]) -> Path:
        """Atomically write *snapshot* and return its path."""
        task_id = str(snapshot.get("task_id") or "unknown")
        target = self.directory / f"{task_id}.json"
        tmp = self.directory / f".{task_id}.tmp"
        tmp.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, target)
        logger.info(
            "checkpoint saved: task=%s iteration=%s -> %s",
            task_id,
            snapshot.get("iteration"),
            target,
        )
        return target

    def load(self, source: str | Path) -> dict[str, Any]:
        """Load and validate a checkpoint by path or task_id."""
        path = Path(source).expanduser()
        if not path.is_file():
            path = self.directory / f"{source}.json"
        if not path.is_file():
            raise CheckpointError(
                f"No checkpoint found at {source!r}."
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(
                f"Checkpoint at {path} is unreadable: {exc}"
            ) from exc
        self._validate(data, path)
        return data

    def latest(self) -> dict[str, Any] | None:
        """The most recently saved valid checkpoint, if any."""
        candidates = sorted(
            self.directory.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            try:
                return self.load(path)
            except CheckpointError:
                continue
        return None

    # -- validation ----------------------------------------------------------

    @staticmethod
    def _checksum(snapshot: dict[str, Any]) -> str:
        payload = {
            key: value
            for key, value in snapshot.items()
            if key != "checksum"
        }
        canonical = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, default=str
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _validate(self, data: Any, path: Path) -> None:
        if not isinstance(data, dict):
            raise CheckpointError(
                f"Checkpoint at {path} is not a JSON object."
            )
        if data.get("schema_version") != SCHEMA_VERSION:
            raise CheckpointError(
                f"Checkpoint at {path} has unsupported schema "
                f"version {data.get('schema_version')!r}."
            )
        for key in ("task_id", "goal", "state"):
            if key not in data:
                raise CheckpointError(
                    f"Checkpoint at {path} is missing {key!r}."
                )
        expected = data.get("checksum")
        actual = self._checksum(data)
        if not expected or expected != actual:
            raise CheckpointError(
                f"Checkpoint at {path} failed its integrity "
                "check (corrupted or tampered)."
            )
        # The payloads must be rebuildable before anyone trusts them.
        try:
            AgentState.from_dict(data["state"])
            if data.get("plan") is not None:
                TaskPlan.from_dict(data["plan"])
        except Exception as exc:
            raise CheckpointError(
                f"Checkpoint at {path} cannot be restored: {exc}"
            ) from exc

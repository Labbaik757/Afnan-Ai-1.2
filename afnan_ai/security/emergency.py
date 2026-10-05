"""Global EmergencyStop / kill switch.

The authorized owner can halt everything, right now:

* current task
* all background tasks
* all subagents
* pending sensitive actions
* connector actions
* browser / computer automation

Subsystems register halt callbacks; tripping the stop
invokes every callback exactly once and denies all new
actions at the policy engine until an explicit,
authorized reset.  The agent NEVER auto-resumes — reset
requires the owner to say so, on the record.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class StopRecord:
    tripped_at: float
    reason: str
    actor: str
    halted: list[str] = field(default_factory=list)
    reset_at: float | None = None
    reset_by: str = ""
    reset_reason: str = ""


class EmergencyStop:
    """Process-wide kill switch."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tripped = False
        self._record: StopRecord | None = None
        self._callbacks: dict[str, Callable[[], None]] = {}

    def register_halt(
        self, name: str, callback: Callable[[], None]
    ) -> None:
        with self._lock:
            self._callbacks[str(name)] = callback

    def register(
        self, callback: Callable[[], None], *,
        name: str = "",
    ) -> str:
        """Register a halt callback, returning an opaque
        token for unregister()."""
        import uuid

        token = name or f"halt-{uuid.uuid4().hex[:8]}"
        self.register_halt(token, callback)
        return token

    def unregister(self, token: str) -> None:
        self.unregister_halt(token)

    def unregister_halt(self, name: str) -> None:
        with self._lock:
            self._callbacks.pop(str(name), None)

    def is_tripped(self) -> bool:
        with self._lock:
            return self._tripped

    def trip(
        self, reason: str, *, actor: str = "owner"
    ) -> StopRecord:
        """Halt everything.  Idempotent — the second trip
        just re-reports."""
        with self._lock:
            if self._tripped and self._record is not None:
                return self._record
            halted: list[str] = []
            for name, callback in list(
                self._callbacks.items()
            ):
                try:
                    callback()
                    halted.append(name)
                except Exception:
                    halted.append(f"{name}:halt_failed")
            self._tripped = True
            self._record = StopRecord(
                tripped_at=time.time(),
                reason=str(reason),
                actor=str(actor),
                halted=halted,
            )
            return self._record

    def reset(
        self,
        *,
        authorized: bool,
        actor: str = "owner",
        reason: str = "",
    ) -> bool:
        """Resume ONLY on explicit owner authorization.
        Anything else is refused and stays refused."""
        with self._lock:
            if not self._tripped:
                return True
            if not authorized:
                return False
            if self._record is not None:
                self._record.reset_at = time.time()
                self._record.reset_by = str(actor)
                self._record.reset_reason = str(reason)
            self._tripped = False
            return True

    def record(self) -> StopRecord | None:
        with self._lock:
            return self._record

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tripped": self._tripped,
                "halt_callbacks": sorted(self._callbacks),
                "record": (
                    {
                        "reason": self._record.reason,
                        "actor": self._record.actor,
                        "halted": list(
                            self._record.halted
                        ),
                    }
                    if self._record
                    else None
                ),
            }

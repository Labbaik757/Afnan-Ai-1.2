"""Controlled subagent communication.

Subagents never talk to each other directly and share no
memory.  All messages route through the parent-owned
mailbox: they are structured, injection-scanned, logged to
the audit trail, and delivered either to the parent or —
at the parent's discretion — as context into another
subagent's next run.  There is no hidden channel.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SubAgentMessage:
    from_id: str
    to_id: str  # another subagent_id, or "parent"
    kind: str  # "result" | "question" | "evidence" | "notice"
    payload: dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_id": self.from_id,
            "to_id": self.to_id,
            "kind": self.kind,
            "payload": dict(self.payload),
            "at": self.at,
        }


class MessageRejected(Exception):
    """A message failed the mailbox policy."""


class SubAgentMailbox:
    """Parent-mediated message passing with audit."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inbox: dict[str, list[SubAgentMessage]] = {}
        self._log: list[SubAgentMessage] = []

    def _scan(self, payload: dict[str, Any]) -> bool:
        from afnan_ai.agent_loop import scan_for_injection

        blob = str(payload)
        return bool(scan_for_injection(blob))

    def send(
        self,
        from_id: str,
        to_id: str,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        known_ids: set[str] | None = None,
    ) -> SubAgentMessage:
        """Route a message via the parent.  Raises
        MessageRejected on policy violation."""
        payload = dict(payload or {})
        if kind not in (
            "result", "question", "evidence", "notice"
        ):
            raise MessageRejected(f"unknown kind {kind!r}")
        if known_ids is not None and to_id != "parent":
            if to_id not in known_ids:
                raise MessageRejected(
                    f"unknown recipient {to_id!r}"
                )
        if self._scan(payload):
            raise MessageRejected(
                "instruction-like content in message payload"
            )
        message = SubAgentMessage(
            from_id=from_id, to_id=to_id, kind=kind,
            payload=payload,
        )
        with self._lock:
            self._inbox.setdefault(to_id, []).append(message)
            self._log.append(message)
        return message

    def receive(self, recipient: str) -> list[SubAgentMessage]:
        """Drain one recipient's inbox."""
        with self._lock:
            return self._inbox.pop(recipient, [])

    def audit_log(self) -> list[dict[str, Any]]:
        with self._lock:
            return [m.to_dict() for m in self._log]

"""Tamper-resistant AuditLogger.

Every important action gets a structured record:
timestamp, task id, actor, action, target, risk level,
permission result, approval status, execution result and
verification result.  Records form a hash chain (each
carries the previous record's hash), so tampering with
history is detectable via ``verify_chain``.

Secrets and sensitive payloads are redacted before a
record is written — the audit log is safe to read.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from afnan_ai.log_config import get_logger
from afnan_ai.redaction import redact_value

logger = get_logger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(record: dict[str, Any]) -> str:
    canonical = json.dumps(
        record, sort_keys=True, ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class AuditLogger:
    """Append-only, hash-chained, redacted audit log."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        on_event: Callable[[dict[str, Any]], None]
        | None = None,
        max_memory: int = 1000,
    ) -> None:
        self._path = (
            Path(str(path)).expanduser()
            if path else None
        )
        self._on_event = on_event
        self._records: list[dict[str, Any]] = []
        self._max_memory = max(100, max_memory)
        self._lock = threading.RLock()
        self._last_hash = "GENESIS"
        if self._path and self._path.exists():
            self._recover_tail()

    # -- writing ------------------------------------------------------------
    def log(
        self,
        event: str,
        *,
        actor: str = "",
        action: str = "",
        target: str = "",
        task_id: str = "",
        risk_level: str = "",
        permission_result: str = "",
        approval_status: str = "",
        execution_result: str = "",
        verification_result: str = "",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "seq": None,  # filled below
            "at": _utcnow(),
            "event": str(event),
            "actor": str(actor)[:120],
            "action": str(action)[:200],
            "target": str(target)[:200],
            "task_id": str(task_id)[:64],
            "risk_level": str(risk_level),
            "permission_result": str(permission_result),
            "approval_status": str(approval_status),
            "execution_result": str(execution_result)[:200],
            "verification_result": str(
                verification_result
            )[:120],
            "details": redact_value(details or {}),
            "prev_hash": self._last_hash,
        }
        with self._lock:
            record["seq"] = self._next_seq()
            record["hash"] = _hash(
                {k: v for k, v in record.items()
                 if k != "hash"}
            )
            self._last_hash = record["hash"]
            self._records.append(record)
            if len(self._records) > self._max_memory:
                del self._records[
                    : len(self._records) - self._max_memory
                ]
            self._persist(record)
        if self._on_event:
            try:
                self._on_event(dict(record))
            except Exception:
                pass
        return record

    # -- reading --------------------------------------------------------------
    def query(
        self,
        *,
        event: str = "",
        actor: str = "",
        task_id: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            out = list(self._records)
        if event:
            out = [r for r in out if r["event"] == event]
        if actor:
            out = [r for r in out if r["actor"] == actor]
        if task_id:
            out = [r for r in out if r["task_id"] == task_id]
        return out[-max(1, limit):]

    def verify_chain(self) -> dict[str, Any]:
        """Replay the hash chain; report tampering."""
        with self._lock:
            prev = "GENESIS"
            checked = 0
            for record in self._records:
                if record.get("prev_hash") != prev:
                    return {
                        "ok": False,
                        "checked": checked,
                        "broken_at": record.get("seq"),
                    }
                expect = _hash(
                    {k: v for k, v in record.items()
                     if k != "hash"}
                )
                if record.get("hash") != expect:
                    return {
                        "ok": False,
                        "checked": checked,
                        "broken_at": record.get("seq"),
                    }
                prev = record["hash"]
                checked += 1
            return {"ok": True, "checked": checked}

    # -- internals --------------------------------------------------------------
    def _next_seq(self) -> int:
        if self._records:
            return int(self._records[-1].get("seq", 0)) + 1
        if self._path and self._path.exists():
            try:
                lines = self._path.read_text().splitlines()
                if lines:
                    return int(
                        json.loads(lines[-1]).get("seq", 0)
                    ) + 1
            except Exception:
                pass
        return 1

    def _persist(self, record: dict[str, Any]) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(
                parents=True, exist_ok=True
            )
            with open(self._path, "a",
                       encoding="utf-8") as handle:
                handle.write(
                    json.dumps(record, ensure_ascii=False,
                               default=str) + "\n"
                )
        except Exception as e:  # noqa: BLE001 - never breaks
            logger.warning("audit persist failed: %s", e)

    def _recover_tail(self) -> None:
        try:
            lines = self._path.read_text().splitlines()
            tail = lines[-self._max_memory:]
            for line in tail:
                record = json.loads(line)
                self._records.append(record)
            if self._records:
                self._last_hash = self._records[-1]["hash"]
        except Exception as e:  # noqa: BLE001 - degrade
            logger.warning("audit recovery failed: %s", e)

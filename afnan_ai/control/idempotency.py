"""Idempotency for remote commands.

Network retries must never cause a dangerous command to execute
twice.  Callers attach an ``idempotency_key``; the router records
the outcome of the first execution and replays the stored result
for duplicates.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IdempotencyRecord:
    key: str
    command_id: str
    command_type: str
    device_id: str
    session_id: str
    result: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    # A completed record may be replayed; an in-flight key lets a
    # concurrent duplicate wait instead of executing twice.
    in_flight: bool = False


class IdempotencyStore:
    """Thread-safe idempotency-key store with TTL."""

    def __init__(self, *, ttl_s: float = 24 * 3600) -> None:
        self._ttl_s = ttl_s
        self._lock = threading.RLock()
        self._records: dict[str, IdempotencyRecord] = {}

    def _scoped(self, device_id: str, key: str) -> str:
        return f"{device_id}:{key}"

    def check(
        self, device_id: str, key: str
    ) -> IdempotencyRecord | None:
        """Return the existing record for a key, if any."""
        if not key:
            return None
        scoped = self._scoped(device_id, key)
        with self._lock:
            record = self._records.get(scoped)
            if record is None:
                return None
            if time.time() - record.created_at > self._ttl_s:
                del self._records[scoped]
                return None
            return record

    def claim(
        self,
        device_id: str,
        key: str,
        *,
        command_id: str,
        command_type: str,
        session_id: str,
    ) -> bool:
        """Claim a key for execution.  Returns False when another
        execution already owns it (duplicate)."""
        if not key:
            return True
        scoped = self._scoped(device_id, key)
        with self._lock:
            if scoped in self._records:
                return False
            self._records[scoped] = IdempotencyRecord(
                key=key,
                command_id=command_id,
                command_type=command_type,
                device_id=device_id,
                session_id=session_id,
                in_flight=True,
            )
            return True

    def complete(
        self,
        device_id: str,
        key: str,
        result: dict[str, Any],
    ) -> None:
        if not key:
            return
        scoped = self._scoped(device_id, key)
        with self._lock:
            record = self._records.get(scoped)
            if record is not None:
                record.result = dict(result)
                record.in_flight = False

    def release(self, device_id: str, key: str) -> None:
        """Drop a claim (used when execution never completed)."""
        if not key:
            return
        scoped = self._scoped(device_id, key)
        with self._lock:
            record = self._records.get(scoped)
            if record is not None and record.in_flight:
                del self._records[scoped]

    def reap(self) -> int:
        now = time.time()
        count = 0
        with self._lock:
            stale = [
                k
                for k, r in self._records.items()
                if now - r.created_at > self._ttl_s
            ]
            for key in stale:
                del self._records[key]
                count += 1
        return count

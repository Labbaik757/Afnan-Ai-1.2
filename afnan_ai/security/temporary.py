"""Temporary capability grants.

A grant carries capability, scope, duration, task id,
reason and the approval that authorized it.  When it
expires the permission is revoked automatically — the
check prunes expired grants, so nothing lingers.

Permanent permissions are never created implicitly;
anything beyond a task's own lifetime needs a fresh
grant.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TemporaryGrant:
    grant_id: str
    capability_id: str
    actor: str
    scope: dict[str, Any] = field(default_factory=dict)
    task_id: str = ""
    reason: str = ""
    approved_by: str = ""
    granted_at: float = field(default_factory=time.time)
    expires_at: float = 0.0

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at

    @property
    def ttl_remaining(self) -> float:
        return max(0.0, self.expires_at - time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "capability_id": self.capability_id,
            "actor": self.actor,
            "scope": dict(self.scope),
            "task_id": self.task_id,
            "reason": self.reason,
            "approved_by": self.approved_by,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "expired": self.expired,
        }


class TemporaryGrantStore:
    """Thread-safe store; expiry is enforced on read."""

    def __init__(self) -> None:
        self._grants: dict[str, TemporaryGrant] = {}
        self._lock = threading.RLock()

    def grant(
        self,
        capability_id: str,
        actor: str,
        *,
        duration_s: float,
        scope: dict[str, Any] | None = None,
        task_id: str = "",
        reason: str = "",
        approved_by: str = "",
    ) -> TemporaryGrant:
        if duration_s <= 0:
            raise ValueError("duration must be positive")
        grant = TemporaryGrant(
            grant_id=f"tg-{uuid.uuid4().hex[:12]}",
            capability_id=str(capability_id),
            actor=str(actor),
            scope=dict(scope or {}),
            task_id=str(task_id),
            reason=str(reason),
            approved_by=str(approved_by),
            expires_at=time.time() + float(duration_s),
        )
        with self._lock:
            self._grants[grant.grant_id] = grant
            self._prune_locked()
        return grant

    def revoke(self, grant_id: str) -> bool:
        with self._lock:
            return self._grants.pop(grant_id, None) is not None

    def active_for(
        self, actor: str, capability_id: str
    ) -> TemporaryGrant | None:
        """Newest live grant for (actor, capability)."""
        with self._lock:
            self._prune_locked()
            best: TemporaryGrant | None = None
            for grant in self._grants.values():
                if (
                    grant.actor == actor
                    and grant.capability_id == capability_id
                    and not grant.expired
                ):
                    if (
                        best is None
                        or grant.expires_at > best.expires_at
                    ):
                        best = grant
            return best

    def active_for_actor(
        self, actor: str
    ) -> list[TemporaryGrant]:
        with self._lock:
            self._prune_locked()
            return [
                g for g in self._grants.values()
                if g.actor == actor and not g.expired
            ]

    def _prune_locked(self) -> None:
        dead = [
            gid for gid, g in self._grants.items()
            if g.expired
        ]
        for gid in dead:
            del self._grants[gid]

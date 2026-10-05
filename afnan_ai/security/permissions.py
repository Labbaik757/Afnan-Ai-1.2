"""Capability-based PermissionManager.

Permissions are capabilities ("browser.read",
"file.write", "email.send") — not roles.  Every actor
(user, agent, subagent, tool, skill, connector) holds an
explicit capability set; least privilege means each task,
subagent and skill gets only what it needs.

A subagent's capabilities can never exceed its parent's:
``derive_child`` intersects, so privilege cannot be
escalated by delegation.
"""

from __future__ import annotations

import threading
from typing import Any

from afnan_ai.security.models import Actor, ActorKind
from afnan_ai.security.risk import risk_for_capability


class PermissionManager:
    """Central capability registry."""

    def __init__(self) -> None:
        self._grants: dict[str, set[str]] = {}
        self._lock = threading.RLock()

    # -- administration ---------------------------------------------------
    def grant(
        self, actor: Actor | str, *capabilities: str
    ) -> None:
        key = self._key(actor)
        with self._lock:
            granted = self._grants.setdefault(key, set())
            for cap in capabilities:
                cap = str(cap or "").strip().lower()
                if cap:
                    granted.add(cap)

    def revoke(
        self, actor: Actor | str, *capabilities: str
    ) -> None:
        key = self._key(actor)
        with self._lock:
            granted = self._grants.get(key, set())
            for cap in capabilities:
                granted.discard(
                    str(cap or "").strip().lower()
                )

    def revoke_all(self, actor: Actor | str) -> None:
        with self._lock:
            self._grants.pop(self._key(actor), None)

    def capabilities_for(
        self, actor: Actor | str
    ) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                sorted(self._grants.get(self._key(actor), ()))
            )

    # -- checks -------------------------------------------------------------
    def check(
        self, actor: Actor | str, capability: str
    ) -> bool:
        """Exact or prefix-wildcard match ('browser.*')."""
        capability = str(capability or "").strip().lower()
        if not capability:
            return False
        with self._lock:
            granted = self._grants.get(self._key(actor), set())
            if capability in granted:
                return True
            parts = capability.split(".")
            for i in range(1, len(parts)):
                if ".".join(parts[:i]) + ".*" in granted:
                    return True
            return False

    def check_any(
        self, actor: Actor | str,
        capabilities: list[str],
    ) -> bool:
        return any(
            self.check(actor, c) for c in capabilities
        )

    def missing(
        self, actor: Actor | str, capability: str
    ) -> str:
        return "" if self.check(actor, capability) else capability

    # -- least privilege ------------------------------------------------------
    def derive_child(
        self, parent: Actor, child_kind: ActorKind,
        child_id: str, requested: list[str],
    ) -> Actor:
        """A child's capabilities are the intersection of
        what was requested and what the parent holds —
        delegation can never escalate privilege."""
        parent_caps = set(self.capabilities_for(parent))
        allowed = tuple(
            c for c in requested
            if c in parent_caps or f"{c.split('.')[0]}.*"
            in parent_caps
        )
        child = Actor(
            kind=child_kind, actor_id=child_id,
            capabilities=allowed,
        )
        self.grant(child, *allowed)
        return child

    def grant_least_privilege(
        self,
        actor: Actor | str,
        required: list[str],
        *,
        read_only: bool = False,
    ) -> tuple[str, ...]:
        """Grant exactly the required capabilities (optionally
        read-only ones), nothing more."""
        from afnan_ai.security.models import RiskLevel

        wanted = []
        for cap in required:
            cap = str(cap or "").strip().lower()
            if not cap:
                continue
            if read_only and risk_for_capability(
                cap
            ) is not RiskLevel.READ_ONLY:
                continue
            wanted.append(cap)
        self.grant(actor, *wanted)
        return tuple(wanted)

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _key(actor: Actor | str) -> str:
        if isinstance(actor, Actor):
            return actor.label
        return str(actor)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                k: sorted(v)
                for k, v in self._grants.items()
            }

"""Shared-resource locks for parallel subagents.

Independent subgoals may run in parallel, but some resources
cannot be shared safely:

* the same browser tab (conflicting navigation/actions),
* the same file (simultaneous destructive writes),
* a stateful connector session.

A subagent declares the resources it needs in its spec; the
manager acquires them before starting it.  Acquisition
blocks up to a timeout — on timeout the subagent fails fast
with a structured ``resource_conflict`` instead of
corrupting shared state.
"""

from __future__ import annotations

import threading
import time
from typing import Any


class ResourceConflict(Exception):
    """A subagent could not acquire its resources in time."""

    def __init__(
        self, owner: str, resources: list[str]
    ) -> None:
        super().__init__(
            f"subagent {owner!r} could not acquire resources: "
            + ", ".join(resources)
        )
        self.owner = owner
        self.resources = list(resources)


class ResourceLockManager:
    """Named exclusive locks, thread-safe."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        # resource -> owner subagent_id
        self._held: dict[str, str] = {}
        # owner -> set of resources (for release-all)
        self._by_owner: dict[str, set[str]] = {}

    def acquire(
        self,
        owner: str,
        resources: list[str],
        timeout: float = 30.0,
    ) -> list[str]:
        """Acquire *resources* for *owner* (blocking, bounded).

        Returns the acquired list.  Raises ResourceConflict
        on timeout.  Re-acquiring an already-held resource by
        the same owner is a no-op.
        """
        wanted = [r for r in dict.fromkeys(resources or [])]
        if not wanted:
            return []
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while True:
                blocking = [
                    r for r in wanted
                    if r in self._held
                    and self._held[r] != owner
                ]
                if not blocking:
                    for resource in wanted:
                        self._held[resource] = owner
                    self._by_owner.setdefault(
                        owner, set()
                    ).update(wanted)
                    return wanted
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ResourceConflict(owner, blocking)
                self._condition.wait(timeout=remaining)

    def release(self, owner: str) -> list[str]:
        """Release everything held by *owner*."""
        with self._condition:
            owned = self._by_owner.pop(owner, set())
            for resource in owned:
                if self._held.get(resource) == owner:
                    del self._held[resource]
            self._condition.notify_all()
            return sorted(owned)

    def held_by(self, owner: str) -> list[str]:
        with self._condition:
            return sorted(self._by_owner.get(owner, set()))

    def status(self) -> dict[str, str]:
        with self._condition:
            return dict(self._held)

"""Context caching with invalidation.

Repeated retrieval is cached; the cache invalidates on
state change, observation change, TTL expiry, source
update or task transition.  Stale cached context is never
reused for critical actions — callers pass
``critical=True`` to force a fresh retrieval.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Callable


class ContextCache:
    """TTL + event-invalidated cache for retrieval results."""

    def __init__(self, *, default_ttl_s: float = 120.0) -> None:
        self.default_ttl_s = max(0.01, default_ttl_s)
        self._entries: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._version = 0  # bumped on any invalidation event

    @staticmethod
    def _key(query: str, params: dict[str, Any]) -> str:
        payload = query + "|" + repr(
            sorted((params or {}).items())
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        critical: bool = False,
    ) -> Any | None:
        """Return cached value, or None on miss/invalidation.

        ``critical=True`` always misses (fresh retrieval).
        """
        if critical:
            return None
        key = self._key(query, params or {})
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry["version"] != self._version:
                return None
            if time.monotonic() > entry["expires_at"]:
                del self._entries[key]
                return None
            return entry["value"]

    def put(
        self,
        query: str,
        value: Any,
        params: dict[str, Any] | None = None,
        *,
        ttl_s: float | None = None,
    ) -> None:
        key = self._key(query, params or {})
        with self._lock:
            self._entries[key] = {
                "value": value,
                "expires_at": time.monotonic()
                + (ttl_s or self.default_ttl_s),
                "version": self._version,
            }
            # Bound the cache.
            while len(self._entries) > 200:
                oldest = min(
                    self._entries,
                    key=lambda k: self._entries[k][
                        "expires_at"
                    ],
                )
                del self._entries[oldest]

    def invalidate(
        self, *, reason: str = ""
    ) -> int:
        """Invalidate everything (state/observation/task change)."""
        with self._lock:
            self._version += 1
            dropped = len(self._entries)
            self._entries.clear()
            return dropped

    def get_or_compute(
        self,
        query: str,
        compute: Callable[[], Any],
        params: dict[str, Any] | None = None,
        *,
        critical: bool = False,
        ttl_s: float | None = None,
    ) -> Any:
        cached = self.get(
            query, params, critical=critical
        )
        if cached is not None:
            return cached
        value = compute()
        self.put(query, value, params, ttl_s=ttl_s)
        return value

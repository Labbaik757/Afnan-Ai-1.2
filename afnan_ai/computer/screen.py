"""ScreenObserver — throttled, reference-based observation.

Raw screenshots are never embedded in AgentState or the LLM
context unnecessarily.  The observer:

- captures on trigger (event / action / state-change) or at a
  configurable interval, never in a hot loop;
- stores the PNG bytes in a bounded artifact-style cache and
  hands out *references* (screenshot_ref);
- fingerprints observations so unchanged screens skip
  re-processing;
- supports region capture to keep payloads small.

The actual desktop reading delegates to the existing
ComputerController.observe(); this layer adds the
production policy around it.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from typing import Any

from afnan_ai.computer.models import ComputerObservation


class ScreenObserver:
    """Throttled observation with screenshot references."""

    def __init__(
        self,
        controller: Any,
        *,
        min_interval_s: float = 0.5,
        cache_size: int = 8,
        on_observation: Any = None,
    ) -> None:
        self.controller = controller
        self.min_interval_s = max(0.0, min_interval_s)
        self._cache: dict[str, bytes] = {}
        self._order: list[str] = []
        self._cache_size = max(1, cache_size)
        self._lock = threading.RLock()
        self._last_at = 0.0
        self._last_fingerprint = ""
        self._on_observation = on_observation

    # -- capture ---------------------------------------------------------
    def observe(
        self,
        *,
        force: bool = False,
        region: dict[str, int] | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Return the controller observation, throttled.

        ``force=True`` bypasses throttling (use after actions).
        """
        now = time.monotonic()
        with self._lock:
            if (
                not force
                and now - self._last_at < self.min_interval_s
                and self._last_fingerprint
            ):
                return {"throttled": True, "reason": reason}
            self._last_at = now
        obs = self.controller.observe()
        if not isinstance(obs, dict):
            obs = {}
        fingerprint = str(obs.get("fingerprint", ""))
        with self._lock:
            changed = fingerprint != self._last_fingerprint
            self._last_fingerprint = fingerprint
        obs["throttled"] = False
        obs["screen_changed"] = changed
        if region:
            obs["region"] = dict(region)
        hook = self._on_observation
        if hook is not None:
            try:
                hook(obs)
            except Exception:
                pass
        return obs

    def observe_after_action(
        self, action_kind: str = ""
    ) -> dict[str, Any]:
        """Mandatory fresh observation after an action."""
        return self.observe(
            force=True, reason=f"after:{action_kind}"
        )

    # -- screenshot references ---------------------------------------------
    def store_screenshot(self, png: bytes) -> str:
        """Cache PNG bytes; return the reference id."""
        if not png:
            return ""
        ref = "shot-" + uuid.uuid4().hex[:12]
        with self._lock:
            self._cache[ref] = bytes(png)
            self._order.append(ref)
            while len(self._order) > self._cache_size:
                old = self._order.pop(0)
                self._cache.pop(old, None)
        return ref

    def get_screenshot(self, ref: str) -> bytes | None:
        with self._lock:
            data = self._cache.get(ref)
            return bytes(data) if data else None

    def drop_screenshots(self) -> int:
        with self._lock:
            n = len(self._cache)
            self._cache.clear()
            self._order.clear()
            return n

    # -- change detection ----------------------------------------------------
    @staticmethod
    def fingerprint_of(observation: dict[str, Any]) -> str:
        """Stable fingerprint for change detection."""
        payload = repr(
            sorted(
                (k, str(v)[:200])
                for k, v in observation.items()
                if k
                not in (
                    "captured_at", "screenshot_saved",
                    "elements",
                )
            )
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def has_changed_since(
        self, fingerprint: str
    ) -> bool:
        with self._lock:
            return (
                bool(fingerprint)
                and self._last_fingerprint != fingerprint
            )

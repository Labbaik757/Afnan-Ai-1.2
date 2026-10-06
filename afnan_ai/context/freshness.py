"""Observation freshness tracking.

Browser pages, desktop screens and connector results go
stale.  Every observation carries a captured timestamp and
a fingerprint; the tracker labels it FRESH, POTENTIALLY_STALE
or STALE, and the agent must re-observe before critical
actions on stale data.

State changes (navigation, app switch, window move) mark
affected observations stale immediately — no waiting for
a timeout.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _age_seconds(captured_at: str) -> float:
    try:
        ts = datetime.fromisoformat(captured_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(
            0.0,
            (datetime.now(timezone.utc) - ts).total_seconds(),
        )
    except Exception:
        return float("inf")


class FreshnessTracker:
    """Track freshness of observations by source."""

    def __init__(
        self,
        *,
        fresh_s: float = 60.0,
        stale_s: float = 900.0,
    ) -> None:
        self.fresh_s = fresh_s
        self.stale_s = stale_s
        # source -> list of {"id", "captured_at",
        # "fingerprint", "stale": bool}
        self._observations: dict[str, list[dict]] = {}

    def record(
        self,
        source: str,
        obs_id: str,
        *,
        fingerprint: str = "",
        captured_at: str | None = None,
    ) -> None:
        bucket = self._observations.setdefault(source, [])
        bucket.append(
            {
                "id": obs_id,
                "captured_at": captured_at or _utcnow(),
                "fingerprint": fingerprint,
                "stale": False,
            }
        )
        # Bound the bucket.
        del bucket[:-50]

    def mark_stale(
        self, source: str = "", obs_id: str = ""
    ) -> int:
        """Mark observations stale after a state change.

        Empty source/obs_id marks everything stale.
        Returns the number marked.
        """
        count = 0
        sources = (
            [source] if source else list(self._observations)
        )
        for src in sources:
            for obs in self._observations.get(src, []):
                if obs_id and obs["id"] != obs_id:
                    continue
                if not obs["stale"]:
                    obs["stale"] = True
                    count += 1
        return count

    def freshness_of(
        self, source: str, obs_id: str
    ) -> str:
        """'fresh' | 'potentially_stale' | 'stale' | 'unknown'."""
        for obs in self._observations.get(source, []):
            if obs["id"] != obs_id:
                continue
            if obs["stale"]:
                return "stale"
            age = _age_seconds(obs["captured_at"])
            if age <= self.fresh_s:
                return "fresh"
            if age <= self.stale_s:
                return "potentially_stale"
            return "stale"
        return "unknown"

    def needs_reobservation(
        self, source: str, obs_id: str, *, critical: bool = False
    ) -> bool:
        state = self.freshness_of(source, obs_id)
        if state in ("stale", "unknown"):
            return True
        if critical and state == "potentially_stale":
            return True
        return False

    def fingerprint_changed(
        self, source: str, obs_id: str, fingerprint: str
    ) -> bool:
        for obs in self._observations.get(source, []):
            if obs["id"] == obs_id:
                changed = bool(
                    obs["fingerprint"]
                    and fingerprint
                    and obs["fingerprint"] != fingerprint
                )
                obs["fingerprint"] = fingerprint
                return changed
        return False

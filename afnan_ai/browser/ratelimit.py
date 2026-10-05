"""Rate-limit / anti-bot awareness for the BrowserController.

Two defensive mechanisms — neither of them retries aggressively
or tries to look less like a bot:

* **Detection.**  :class:`RateLimitDetector` recognises
  rate-limit pages, bot-detection warnings and blocking signals
  from the page observation (title/text/URL) and from network
  diagnostics (HTTP 429 responses).  When detected, browser
  actions stop with a structured ``rate_limited`` error.
* **Pacing.**  :class:`ActionPacer` counts the controller's own
  actions per host inside a sliding window.  When a configured
  ``max_actions_per_minute`` would be exceeded, actions pause
  instead of hammering the site (structured error with a
  ``retry_after_s`` hint).

Recovery advice for this failure class is a controlled backoff
or a human — never "retry harder".
"""

from __future__ import annotations

import re
import time
from collections import deque
from typing import Any

__all__ = [
    "ActionPacer",
    "RateLimitDetector",
    "RateLimitPolicy",
]

_TEXT_SIGNALS = (
    "too many requests",
    "rate limit exceeded",
    "rate-limited",
    "rate limited",
    "slow down",
    "try again later",
    "unusual traffic",
    "automated queries",
    "temporarily blocked",
    "request blocked",
    "your request has been blocked",
)

_URL_SIGNALS = ("rate-limit", "too-many-requests", "ratelimit")

_RETRY_AFTER_RE = re.compile(r"retry after (\d+)", re.IGNORECASE)


class RateLimitDetector:
    """Detects rate-limit / bot-block pages from an observation."""

    def detect(
        self,
        observation: dict[str, Any],
        network: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = str(observation.get("url") or "")
        haystack = (
            f"{observation.get('title') or ''}\n"
            f"{observation.get('text') or ''}"
        ).lower()
        signals: list[str] = []
        for fragment in _URL_SIGNALS:
            if fragment in url.lower():
                signals.append(f"url:{fragment}")
        for phrase in _TEXT_SIGNALS:
            if phrase in haystack:
                signals.append(f"text:{phrase}")

        kind: str | None = None
        retry_after_s: int | None = None
        if network and network.get("rate_limited_responses"):
            kind = "rate_limited"
            signals.append("network:http_429")
        if signals and kind is None:
            kind = (
                "bot_blocked"
                if any(
                    "blocked" in s or "automated" in s
                    for s in signals
                )
                else "rate_limited"
            )
        match = _RETRY_AFTER_RE.search(haystack)
        if match:
            retry_after_s = int(match.group(1))

        detected = bool(signals)
        return {
            "detected": detected,
            "kind": kind if detected else None,
            "signals": sorted(set(signals)),
            "retry_after_s": retry_after_s,
            "url": url,
        }


class RateLimitPolicy:
    """Pacing limits for the controller's own actions.

    ``max_actions_per_minute`` None (the default) disables pacing;
    page-signal detection stays on regardless.
    """

    def __init__(
        self,
        max_actions_per_minute: int | None = None,
        window_s: float = 60.0,
    ) -> None:
        if max_actions_per_minute is not None and (
            max_actions_per_minute < 1
        ):
            raise ValueError(
                "max_actions_per_minute must be positive or None"
            )
        self.max_actions_per_minute = max_actions_per_minute
        self.window_s = float(window_s)


class ActionPacer:
    """Sliding-window action counter, per host."""

    def __init__(self, policy: RateLimitPolicy) -> None:
        self.policy = policy
        self._hits: dict[str, deque[float]] = {}

    def _prune(self, host: str, now: float) -> deque[float]:
        hits = self._hits.setdefault(host, deque())
        cutoff = now - self.policy.window_s
        while hits and hits[0] < cutoff:
            hits.popleft()
        return hits

    def check(self, host: str) -> tuple[bool, float]:
        """(allowed, retry_after_s) for one more action on *host*."""
        limit = self.policy.max_actions_per_minute
        if limit is None or not host:
            return True, 0.0
        now = time.monotonic()
        hits = self._prune(host, now)
        if len(hits) < limit:
            return True, 0.0
        retry_after = max(0.0, (hits[0] + self.policy.window_s) - now)
        return False, retry_after

    def note(self, host: str) -> None:
        if not host:
            return
        now = time.monotonic()
        self._prune(host, now).append(now)

    def status(self, host: str) -> dict[str, Any]:
        limit = self.policy.max_actions_per_minute
        if limit is None:
            return {
                "host": host,
                "pacing_enabled": False,
                "actions_in_window": 0,
                "limit": None,
            }
        hits = self._prune(host, time.monotonic())
        allowed, retry_after = self.check(host)
        return {
            "host": host,
            "pacing_enabled": True,
            "actions_in_window": len(hits),
            "limit": limit,
            "window_s": self.policy.window_s,
            "allowed": allowed,
            "retry_after_s": round(retry_after, 2),
        }

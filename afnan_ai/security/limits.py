"""Rate & abuse controls.

Per actor (and per task) the limiter enforces:

* tool-call limit        (max_tool_calls)
* action frequency limit (max calls per window)
* runtime limit          (max seconds of activity)
* retry limit            (max consecutive failures)
* resource limit         (max actions per task)

Repeated suspicious behavior (denials, failures,
injection hits) escalates: first a warning event, then
``should_pause_task()`` advises the orchestrator to pause
or terminate the task.  The limiter never executes
anything — it only advises.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RatePolicy:
    max_tool_calls: int = 200
    max_calls_per_minute: int = 60
    max_runtime_s: float = 3600.0
    max_consecutive_failures: int = 5
    max_denials_before_pause: int = 3
    max_injection_hits_before_pause: int = 2


@dataclass
class _Usage:
    calls: int = 0
    window_start: float = field(
        default_factory=time.monotonic
    )
    window_calls: int = 0
    first_seen: float = field(
        default_factory=time.monotonic
    )
    consecutive_failures: int = 0
    denials: int = 0
    injection_hits: int = 0


class RateLimiter:
    """Advisory rate/abuse limiter per actor key."""

    def __init__(
        self, policy: RatePolicy | None = None
    ) -> None:
        self.policy = policy or RatePolicy()
        self._usage: dict[str, _Usage] = {}
        self._lock = threading.RLock()

    def _get(self, key: str) -> _Usage:
        with self._lock:
            return self._usage.setdefault(key, _Usage())

    def check(
        self, key: str, *, now: float | None = None
    ) -> dict[str, Any]:
        """Pre-action check.  Returns {'ok': bool,
        'code': str, 'reason': str}."""
        moment = now if now is not None else time.monotonic()
        usage = self._get(key)
        policy = self.policy
        with self._lock:
            if usage.calls >= policy.max_tool_calls:
                return self._deny(
                    key, "tool_call_limit",
                    f"{usage.calls} calls exceeds "
                    f"{policy.max_tool_calls}",
                )
            if (
                moment - usage.window_start >= 60.0
            ):
                usage.window_start = moment
                usage.window_calls = 0
            if (
                usage.window_calls
                >= policy.max_calls_per_minute
            ):
                return self._deny(
                    key, "frequency_limit",
                    f"{usage.window_calls}/min exceeds "
                    f"{policy.max_calls_per_minute}",
                )
            if (
                moment - usage.first_seen
                >= policy.max_runtime_s
            ):
                return self._deny(
                    key, "runtime_limit",
                    "actor runtime budget exhausted",
                )
            return {"ok": True, "code": "ok", "reason": ""}

    def record_call(self, key: str) -> None:
        usage = self._get(key)
        with self._lock:
            usage.calls += 1
            usage.window_calls += 1

    def record_result(
        self, key: str, success: bool
    ) -> None:
        usage = self._get(key)
        with self._lock:
            if success:
                usage.consecutive_failures = 0
            else:
                usage.consecutive_failures += 1

    def record_denial(self, key: str) -> None:
        with self._lock:
            self._get(key).denials += 1

    def record_injection_hit(self, key: str) -> None:
        with self._lock:
            self._get(key).injection_hits += 1

    def should_pause_task(self, key: str) -> dict[str, Any]:
        """True when behavior is suspicious enough to pause
        or terminate the task."""
        usage = self._get(key)
        policy = self.policy
        with self._lock:
            if (
                usage.consecutive_failures
                >= policy.max_consecutive_failures
            ):
                return {
                    "pause": True,
                    "code": "repeated_failed_action",
                    "reason": (
                        f"{usage.consecutive_failures} "
                        "consecutive failures"
                    ),
                }
            if (
                usage.denials
                >= policy.max_denials_before_pause
            ):
                return {
                    "pause": True,
                    "code": "repeated_denied",
                    "reason": (
                        f"{usage.denials} permission denials"
                    ),
                }
            if (
                usage.injection_hits
                >= policy.max_injection_hits_before_pause
            ):
                return {
                    "pause": True,
                    "code": "prompt_injection",
                    "reason": (
                        f"{usage.injection_hits} injection "
                        "hits"
                    ),
                }
            return {"pause": False, "code": "ok",
                    "reason": ""}

    def reset(self, key: str) -> None:
        with self._lock:
            self._usage.pop(key, None)

    def _deny(
        self, key: str, code: str, reason: str
    ) -> dict[str, Any]:
        self.record_denial(key)
        return {"ok": False, "code": code,
                "reason": reason}

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


# ---------------------------------------------------------------------------
# Resource limits: per task/agent/subagent/skill/connector/browser session/
# background worker — max steps, runtime, tool calls, retries, network
# requests, resource usage and concurrent operations.  Exceeding a limit
# pauses or terminates safely; it never fails open.


@dataclass
class ResourceLimits:
    max_steps: int = 200
    max_runtime_s: float = 3600.0
    max_tool_calls: int = 500
    max_retries: int = 5
    max_network_requests: int = 200
    max_concurrent_ops: int = 4
    max_memory_mb: int = 1024


@dataclass
class _ResourceUsage:
    steps: int = 0
    tool_calls: int = 0
    retries: int = 0
    network_requests: int = 0
    concurrent_ops: int = 0
    started_at: float = field(
        default_factory=time.monotonic
    )


class ResourceGovernor:
    """Enforces ResourceLimits per scope key."""

    def __init__(
        self, limits: ResourceLimits | None = None
    ) -> None:
        self.limits = limits or ResourceLimits()
        self._usage: dict[str, _ResourceUsage] = {}
        self._lock = threading.RLock()

    def _get(self, key: str) -> _ResourceUsage:
        with self._lock:
            return self._usage.setdefault(
                key, _ResourceUsage()
            )

    def check(self, key: str) -> dict[str, Any]:
        usage = self._get(key)
        limits = self.limits
        now = time.monotonic()
        with self._lock:
            if usage.steps >= limits.max_steps:
                return self._over(
                    "max_steps", usage.steps, limits.max_steps
                )
            if usage.tool_calls >= limits.max_tool_calls:
                return self._over(
                    "max_tool_calls", usage.tool_calls,
                    limits.max_tool_calls,
                )
            if usage.retries >= limits.max_retries:
                return self._over(
                    "max_retries", usage.retries,
                    limits.max_retries,
                )
            if (
                usage.network_requests
                >= limits.max_network_requests
            ):
                return self._over(
                    "max_network_requests",
                    usage.network_requests,
                    limits.max_network_requests,
                )
            if (
                usage.concurrent_ops
                >= limits.max_concurrent_ops
            ):
                return self._over(
                    "max_concurrent_ops",
                    usage.concurrent_ops,
                    limits.max_concurrent_ops,
                )
            if now - usage.started_at >= limits.max_runtime_s:
                return {
                    "ok": False,
                    "code": "max_runtime",
                    "limit": "max_runtime_s",
                    "reason": "runtime budget exhausted",
                    "pause": True,
                }
            return {"ok": True, "code": "ok", "pause": False}

    def record_step(self, key: str) -> None:
        with self._lock:
            self._get(key).steps += 1

    def record_tool_call(self, key: str) -> None:
        with self._lock:
            self._get(key).tool_calls += 1

    def record_retry(self, key: str) -> None:
        with self._lock:
            self._get(key).retries += 1

    def record_network_request(self, key: str) -> None:
        with self._lock:
            self._get(key).network_requests += 1

    def acquire_op(self, key: str) -> bool:
        with self._lock:
            usage = self._get(key)
            if (
                usage.concurrent_ops
                >= self.limits.max_concurrent_ops
            ):
                return False
            usage.concurrent_ops += 1
            return True

    def release_op(self, key: str) -> None:
        with self._lock:
            usage = self._get(key)
            usage.concurrent_ops = max(
                0, usage.concurrent_ops - 1
            )

    def reset(self, key: str) -> None:
        with self._lock:
            self._usage.pop(key, None)

    @staticmethod
    def _over(
        limit: str, used: int, allowed: int
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "code": limit,
            "limit": limit,
            "reason": f"{used} exceeds {limit}={allowed}",
            "pause": True,
        }

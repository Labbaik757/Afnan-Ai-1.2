"""State-based waiting — never fixed sleeps.

``wait_for`` polls a condition callable until it is true or
the timeout expires.  Desktop UI is dynamic (reloads,
dialogs, delayed rendering); waiting on *state* instead of
wall-clock makes the agent robust without wasting time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class WaitResult:
    satisfied: bool
    elapsed_s: float
    attempts: int
    last_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "satisfied": self.satisfied,
            "elapsed_s": round(self.elapsed_s, 3),
            "attempts": self.attempts,
        }


def wait_for(
    condition: Callable[[], Any],
    *,
    timeout_s: float = 10.0,
    poll_s: float = 0.25,
    truthy: bool = True,
) -> WaitResult:
    """Poll ``condition`` until truthy (or timeout).

    ``condition`` may return a value; the last value is kept
    for the caller.  Never raises on timeout — check
    ``satisfied``.
    """
    timeout_s = max(0.1, float(timeout_s))
    poll_s = max(0.05, min(2.0, float(poll_s)))
    start = time.monotonic()
    attempts = 0
    last: Any = None
    while True:
        attempts += 1
        try:
            last = condition()
        except Exception:
            last = None
        done = (not last) if not truthy else bool(last)
        if done:
            break
        if time.monotonic() - start >= timeout_s:
            break
        time.sleep(poll_s)
    elapsed = time.monotonic() - start
    satisfied = (not bool(last)) if not truthy else bool(last)
    return WaitResult(
        satisfied=satisfied,
        elapsed_s=elapsed,
        attempts=attempts,
        last_value=last,
    )


def wait_for_stable(
    fingerprint_fn: Callable[[], str],
    *,
    stable_for_s: float = 1.0,
    timeout_s: float = 10.0,
    poll_s: float = 0.25,
) -> WaitResult:
    """Wait until the UI fingerprint stops changing.

    Useful after launches / navigations / dialogs: proceed
    only when the screen has settled.
    """
    stable_for_s = max(0.1, float(stable_for_s))
    start = time.monotonic()
    attempts = 0
    last_fp = ""
    stable_since: float | None = None
    while True:
        attempts += 1
        try:
            fp = fingerprint_fn() or ""
        except Exception:
            fp = ""
        now = time.monotonic()
        if fp and fp == last_fp:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= stable_for_s:
                return WaitResult(
                    satisfied=True,
                    elapsed_s=now - start,
                    attempts=attempts,
                    last_value=fp,
                )
        else:
            stable_since = None
        last_fp = fp
        if now - start >= timeout_s:
            return WaitResult(
                satisfied=False,
                elapsed_s=now - start,
                attempts=attempts,
                last_value=fp,
            )
        time.sleep(poll_s)

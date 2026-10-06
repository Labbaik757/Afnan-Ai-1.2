"""Reconnect manager with exponential backoff.

``ReconnectManager`` drives a transport function (for example an SSE
or WebSocket connect) and retries it on :class:`ConnectionError`
with exponential backoff.  Authentication failures stop the loop
immediately: retrying bad credentials never helps.  The last event
cursor is persisted through a pluggable dict-like ``cursor_store``
so a reconnect can resume where it left off.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Callable

from afnan_ai.control.client.errors import (
    AuthenticationError,
    ConnectionError,
)

_CURSOR_KEY = "event_cursor"


class ReconnectManager:
    """Retry a transport callable with backoff until it succeeds."""

    def __init__(
        self,
        *,
        base_delay_s: float = 1.0,
        factor: float = 2.0,
        max_delay_s: float = 60.0,
        jitter: bool = True,
        cursor_store: Any | None = None,
    ) -> None:
        self.base_delay_s = base_delay_s
        self.factor = factor
        self.max_delay_s = max_delay_s
        self.jitter = jitter
        self.cursor_store = cursor_store if cursor_store is not None else {}
        self._stop = threading.Event()
        self.attempts = 0

    # -- cursor persistence ------------------------------------------------

    def save_cursor(self, seq: int) -> None:
        """Persist the last processed event cursor."""
        try:
            self.cursor_store[_CURSOR_KEY] = int(seq)
        except (TypeError, ValueError):
            pass

    def load_cursor(self) -> int:
        """Return the persisted cursor, or 0 when none is stored."""
        try:
            return int(self.cursor_store.get(_CURSOR_KEY, 0) or 0)
        except (TypeError, ValueError):
            return 0

    # -- backoff -------------------------------------------------------------

    def next_delay(self) -> float:
        """Current backoff delay for the in-progress attempt streak."""
        delay = self.base_delay_s * (self.factor**self.attempts)
        delay = min(delay, self.max_delay_s)
        if self.jitter:
            delay = delay + random.uniform(0.0, delay * 0.25)
        return delay

    def reset(self) -> None:
        """Reset the attempt streak after a successful connection."""
        self.attempts = 0

    def stop(self) -> None:
        """Ask an in-progress :meth:`run` loop to stop."""
        self._stop.set()

    # -- main loop -------------------------------------------------------------

    def run(self, fn: Callable[[], Any]) -> Any:
        """Call ``fn`` until it returns, retrying on connection loss.

        ``fn`` returning (or raising ``StopIteration``) ends the
        loop successfully.  :class:`ConnectionError` triggers
        backoff and retry; :class:`AuthenticationError` is re-raised
        immediately; anything else propagates untouched.
        """
        self._stop.clear()
        while not self._stop.is_set():
            try:
                result = fn()
            except AuthenticationError:
                # Bad credentials never heal with retries.
                raise
            except ConnectionError:
                self.attempts += 1
                delay = self.next_delay()
                if self._stop.wait(delay):
                    break
                continue
            self.reset()
            return result
        return None

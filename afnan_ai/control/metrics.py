"""Control-plane observability.

Lightweight in-process metrics for the control plane.  Aggregates
only; feeds the existing Activity/Audit pipeline for persistence.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any


class MetricsCollector:
    """Counters, latencies and gauges for remote operations."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._started_at = time.time()
        self._commands_total = 0
        self._commands_by_type: dict[str, int] = defaultdict(int)
        self._commands_failed = 0
        self._latency_sum = 0.0
        self._latency_max = 0.0
        self._auth_failures = 0
        self._authz_denials = 0
        self._reconnects = 0
        self._events_delivered = 0
        self._events_dropped = 0
        self._approvals_requested = 0
        self._approvals_decided = 0
        self._approval_latency_sum = 0.0
        self._devices_revoked = 0
        self._rate_limit_hits = 0
        self._ws_disconnects = 0
        self._sse_disconnects = 0
        self._event_gaps = 0
        self._duplicate_commands = 0
        self._active_devices = 0
        self._active_sessions = 0

    # -- recorders ------------------------------------------------------------

    def record_command(
        self, command_type: str, latency_s: float
    ) -> None:
        with self._lock:
            self._commands_total += 1
            self._commands_by_type[command_type] += 1
            self._latency_sum += latency_s
            self._latency_max = max(
                self._latency_max, latency_s
            )

    def record_command_failed(self) -> None:
        with self._lock:
            self._commands_failed += 1

    def record_auth_failure(self) -> None:
        with self._lock:
            self._auth_failures += 1

    def record_authz_denial(self) -> None:
        with self._lock:
            self._authz_denials += 1

    def record_reconnect(self) -> None:
        with self._lock:
            self._reconnects += 1

    def record_events_delivered(self, count: int = 1) -> None:
        with self._lock:
            self._events_delivered += count

    def record_events_dropped(self, count: int = 1) -> None:
        with self._lock:
            self._events_dropped += count

    def record_approval_requested(self) -> None:
        with self._lock:
            self._approvals_requested += 1

    def record_approval_decided(self, latency_s: float) -> None:
        with self._lock:
            self._approvals_decided += 1
            self._approval_latency_sum += latency_s

    def record_device_revoked(self) -> None:
        with self._lock:
            self._devices_revoked += 1

    def record_rate_limit_hit(self) -> None:
        with self._lock:
            self._rate_limit_hits += 1

    def record_ws_disconnect(self) -> None:
        with self._lock:
            self._ws_disconnects += 1

    def record_sse_disconnect(self) -> None:
        with self._lock:
            self._sse_disconnects += 1

    def record_event_gap(self) -> None:
        with self._lock:
            self._event_gaps += 1

    def record_duplicate_command(self) -> None:
        with self._lock:
            self._duplicate_commands += 1

    def set_presence(
        self, *, devices: int, sessions: int
    ) -> None:
        with self._lock:
            self._active_devices = devices
            self._active_sessions = sessions

    # -- snapshot ---------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            uptime = time.time() - self._started_at
            avg_latency = (
                self._latency_sum / self._commands_total
                if self._commands_total
                else 0.0
            )
            avg_approval_latency = (
                self._approval_latency_sum
                / self._approvals_decided
                if self._approvals_decided
                else 0.0
            )
            return {
                "uptime_s": round(uptime, 1),
                "active_devices": self._active_devices,
                "active_sessions": self._active_sessions,
                "commands_total": self._commands_total,
                "commands_by_type": dict(
                    self._commands_by_type
                ),
                "commands_failed": self._commands_failed,
                "command_latency_avg_s": round(avg_latency, 4),
                "command_latency_max_s": round(
                    self._latency_max, 4
                ),
                "auth_failures": self._auth_failures,
                "authz_denials": self._authz_denials,
                "reconnects": self._reconnects,
                "events_delivered": self._events_delivered,
                "events_dropped": self._events_dropped,
                "event_gaps": self._event_gaps,
                "duplicate_commands": self._duplicate_commands,
                "ws_disconnects": self._ws_disconnects,
                "sse_disconnects": self._sse_disconnects,
                "approvals_requested": self._approvals_requested,
                "approvals_decided": self._approvals_decided,
                "approval_latency_avg_s": round(
                    avg_approval_latency, 2
                ),
                "devices_revoked": self._devices_revoked,
                "rate_limit_hits": self._rate_limit_hits,
            }

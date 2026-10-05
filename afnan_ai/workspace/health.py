"""WorkspaceHealthMonitor — liveness and resource health.

Monitors process/browser/disk/memory/network/task-heartbeat
health and records structured health events.  Never
exposes secrets.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import Workspace, WorkspaceEvent


@dataclass
class HealthEvent:
    workspace_id: str
    kind: str  # "ok" | "warning" | "critical"
    check: str
    detail: str = ""
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "kind": self.kind,
            "check": self.check,
            "detail": self.detail,
            "at": self.at,
        }


@dataclass
class HealthStatus:
    workspace_id: str
    healthy: bool
    checks: list[HealthEvent] = field(default_factory=list)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "healthy": self.healthy,
            "at": self.at,
            "checks": [c.to_dict() for c in self.checks],
        }


class WorkspaceHealthMonitor:
    """Periodic health checks for workspaces."""

    STALE_HEARTBEAT_S = 300.0

    def __init__(
        self,
        *,
        on_event: (
            Callable[[WorkspaceEvent], None] | None
        ) = None,
    ) -> None:
        self.on_event = on_event
        self._lock = threading.Lock()
        self._last: dict[str, HealthStatus] = {}
        self._browser_alive: dict[str, bool] = {}
        self._proc_alive: dict[str, bool] = {}

    # -- external signals -------------------------------------------

    def report_browser(self, workspace_id: str, alive: bool) -> None:
        with self._lock:
            self._browser_alive[workspace_id] = alive

    def report_process(
        self, workspace_id: str, alive: bool
    ) -> None:
        with self._lock:
            self._proc_alive[workspace_id] = alive

    # -- checks -------------------------------------------------------

    def check(
        self,
        workspace: Workspace,
        *,
        last_heartbeat: float = 0.0,
        disk_usage_mb: float = 0.0,
        memory_mb: float = 0.0,
        cpu_percent: float = 0.0,
        network_ok: bool = True,
        checkpoint_ok: bool = True,
    ) -> HealthStatus:
        limits = workspace.config.resource_limits
        events: list[HealthEvent] = []
        wid = workspace.workspace_id

        def add(kind: str, check: str, detail: str = "") -> None:
            events.append(
                HealthEvent(wid, kind, check, detail)
            )

        # Task heartbeat
        if last_heartbeat:
            age = time.time() - last_heartbeat
            if age > self.STALE_HEARTBEAT_S:
                add(
                    "critical", "heartbeat",
                    f"stale heartbeat: {age:.0f}s",
                )
            else:
                add("ok", "heartbeat")
        # Disk
        if disk_usage_mb > limits.max_disk_mb:
            add(
                "critical", "disk",
                f"{disk_usage_mb:.1f}MB > "
                f"{limits.max_disk_mb:.1f}MB",
            )
        elif disk_usage_mb > 0.8 * limits.max_disk_mb:
            add(
                "warning", "disk",
                f"{disk_usage_mb:.1f}MB near limit",
            )
        else:
            add("ok", "disk")
        # Memory
        if memory_mb > limits.max_memory_mb:
            add(
                "critical", "memory",
                f"{memory_mb:.1f}MB > {limits.max_memory_mb:.1f}MB",
            )
        else:
            add("ok", "memory")
        # CPU
        if cpu_percent > limits.max_cpu_percent:
            add(
                "warning", "cpu",
                f"{cpu_percent:.1f}% > {limits.max_cpu_percent:.1f}%",
            )
        else:
            add("ok", "cpu")
        # Browser
        with self._lock:
            browser_alive = self._browser_alive.get(wid, True)
            proc_alive = self._proc_alive.get(wid, True)
        add(
            "ok" if browser_alive else "critical",
            "browser",
            "" if browser_alive else "browser not responding",
        )
        add(
            "ok" if proc_alive else "critical",
            "process",
            "" if proc_alive else "workspace process dead",
        )
        # Network
        add(
            "ok" if network_ok else "warning",
            "network",
            "" if network_ok else "network unreachable",
        )
        # Checkpoint integrity
        add(
            "ok" if checkpoint_ok else "critical",
            "checkpoint",
            "" if checkpoint_ok else "checkpoint corrupted",
        )

        healthy = all(e.kind != "critical" for e in events)
        status = HealthStatus(wid, healthy, events)
        with self._lock:
            self._last[wid] = status
        for e in events:
            if e.kind != "ok" and self.on_event is not None:
                try:
                    self.on_event(
                        WorkspaceEvent(
                            workspace_id=wid,
                            task_id=workspace.task_id,
                            event_type=f"health.{e.kind}",
                            actor="health_monitor",
                            details={
                                "check": e.check,
                                "detail": e.detail,
                            },
                        )
                    )
                except Exception:
                    pass
        return status

    def last_status(
        self, workspace_id: str
    ) -> HealthStatus | None:
        with self._lock:
            return self._last.get(workspace_id)

    def stale_workspaces(
        self, workspace_ids: list[str]
    ) -> list[str]:
        """Workspaces whose last check was unhealthy."""
        stale: list[str] = []
        with self._lock:
            for wid in workspace_ids:
                status = self._last.get(wid)
                if status is not None and not status.healthy:
                    stale.append(wid)
        return stale

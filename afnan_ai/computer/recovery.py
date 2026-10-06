"""Application crash recovery.

Detects crashes (app vanished / not responding), then:

    health check → restart if permitted → restore task
    context → fresh observation → continue from checkpoint

Already-completed actions are never blindly repeated: the
caller supplies the checkpoint state, and recovery only
re-establishes the *environment*, not the work.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from afnan_ai.computer import wait as wait_mod


@dataclass
class RecoveryReport:
    app: str
    crashed: bool = False
    restarted: bool = False
    context_restored: bool = False
    observation: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CrashRecovery:
    """Detect + recover crashed applications."""

    def __init__(
        self,
        *,
        app_manager: Any,
        screen_observer: Any = None,
        max_restart_attempts: int = 1,
        restart_timeout_s: float = 15.0,
    ) -> None:
        self.apps = app_manager
        self.observer = screen_observer
        self.max_restart_attempts = max(0, max_restart_attempts)
        self.restart_timeout_s = restart_timeout_s
        self._restarts: dict[str, int] = {}

    def detect_crash(self, app_name: str) -> bool:
        """True when the app should be running but isn't."""
        try:
            return not self.apps.is_running(app_name)
        except Exception:
            return False

    def recover(
        self,
        app_name: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> RecoveryReport:
        """Re-establish the app environment (not the work)."""
        report = RecoveryReport(app=app_name)
        if not self.detect_crash(app_name):
            report.note = "app healthy; no recovery needed"
            return report
        report.crashed = True

        attempts = self._restarts.get(app_name, 0)
        if attempts >= self.max_restart_attempts:
            report.note = (
                "restart budget exhausted; manual intervention "
                "required"
            )
            return report
        if not self.apps.is_allowed(app_name):
            report.note = (
                "restart refused: app not in allowed list"
            )
            return report

        try:
            self.apps.restart(app_name)
            self._restarts[app_name] = attempts + 1
        except Exception as exc:
            report.note = f"restart failed: {exc}"
            return report

        # Wait for the app to actually come back (state-based).
        result = wait_mod.wait_for(
            lambda: self.apps.is_running(app_name),
            timeout_s=self.restart_timeout_s,
            poll_s=0.5,
        )
        report.restarted = result.satisfied
        if not result.satisfied:
            report.note = "app did not come back in time"
            return report

        # Fresh observation so the agent re-grounds itself.
        if self.observer is not None:
            try:
                report.observation = self.observer.observe(
                    force=True, reason="after:crash_recovery"
                )
            except Exception:
                report.observation = {}
        report.context_restored = bool(context)
        report.note = (
            "app restarted; re-observed; continue from "
            "checkpoint (completed steps are not repeated)"
        )
        # Give the UI a beat to settle.
        time.sleep(0.5)
        return report

    def reset_budget(self, app_name: str = "") -> None:
        if app_name:
            self._restarts.pop(app_name, None)
        else:
            self._restarts.clear()

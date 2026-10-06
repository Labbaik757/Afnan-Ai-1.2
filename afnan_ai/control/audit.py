"""Control-plane audit adapter.

Every security-relevant remote action is recorded through the
existing tamper-evident audit pipeline
(:class:`~afnan_ai.security.audit_center.SecurityAuditCenter`).
This adapter only shapes control-plane events into that pipeline —
it never implements its own log.
"""

from __future__ import annotations

import time
from typing import Any


class ControlPlaneAuditAdapter:
    """Audit every remote action through the existing audit center."""

    def __init__(
        self, audit_center, *, source: str = "control-plane"
    ) -> None:
        self.audit = audit_center
        self._source = source

    def _event(
        self,
        event: str,
        *,
        actor: str,
        action: str,
        device_id: str = "",
        session_id: str = "",
        task_id: str = "",
        capability: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        if self.audit is None:
            return
        try:
            self.audit.security_event(
                event,
                actor=actor,
                action=action,
                task_id=task_id,
                capability=capability,
                details={
                    "source": self._source,
                    "device_id": device_id,
                    "session_id": session_id,
                    "at": time.time(),
                    **(details or {}),
                },
            )
        except Exception:
            pass

    # -- session / device ---------------------------------------------------------

    def login(
        self, *, device_id: str, session_id: str, principal: str
    ) -> None:
        self._event(
            "remote.login",
            actor=f"remote:{device_id}",
            action="login",
            device_id=device_id,
            session_id=session_id,
            details={"principal": principal},
        )

    def device_paired(
        self, *, device_id: str, approved_by: str
    ) -> None:
        self._event(
            "device.paired",
            actor=f"remote:{device_id}",
            action="device_paired",
            device_id=device_id,
            details={"approved_by": approved_by},
        )

    def device_event(
        self, event: str, *, device_id: str, actor: str = ""
    ) -> None:
        """Audit a generic device-lifecycle event (no secrets)."""
        self._event(
            f"device.{event}",
            actor=actor or f"remote:{device_id}",
            action=f"device_{event}",
            device_id=device_id,
        )

    def permission_granted(
        self,
        *,
        device_id: str,
        capability: str,
        granted_by: str,
    ) -> None:
        self._event(
            "permission.granted",
            actor=granted_by,
            action="permission_granted",
            device_id=device_id,
            capability=capability,
        )

    def session_revoked(
        self, *, session_id: str, device_id: str, actor: str
    ) -> None:
        self._event(
            "remote.session.revoked",
            actor=actor,
            action="session_revoked",
            device_id=device_id,
            session_id=session_id,
        )

    # -- commands --------------------------------------------------------------------

    def command(
        self, *, command, result, capability: str
    ) -> None:
        status = result.status.value
        event = (
            "remote.command.denied"
            if status == "denied"
            else "remote.command.executed"
            if status == "completed"
            else "remote.command.requested"
        )
        self._event(
            event,
            actor=f"remote:{command.device_id}",
            action=command.command_type.value,
            device_id=command.device_id,
            session_id=command.session_id,
            capability=capability or command.requested_capability,
            details={
                "command_id": command.command_id,
                "status": status,
                "error_code": result.error_code,
                "correlation_id": command.correlation_id,
            },
        )

    # -- approvals ---------------------------------------------------------------------

    def approval_requested(
        self, *, approval_id: str, device_id: str = ""
    ) -> None:
        self._event(
            "remote.approval.requested",
            actor=f"remote:{device_id}" if device_id else "system",
            action="approval_requested",
            device_id=device_id,
            details={"approval_id": approval_id},
        )

    def approval_decided(
        self,
        *,
        approval_id: str,
        granted: bool,
        device_id: str,
        session_id: str,
    ) -> None:
        self._event(
            "remote.approval.approved"
            if granted
            else "remote.approval.denied",
            actor=f"remote:{device_id}",
            action="approval_decided",
            device_id=device_id,
            session_id=session_id,
            details={
                "approval_id": approval_id,
                "granted": granted,
            },
        )

    # -- safety --------------------------------------------------------------------------

    def emergency_stop(
        self, *, device_id: str, reason: str
    ) -> None:
        self._event(
            "remote.emergency_stop",
            actor=f"remote:{device_id}",
            action="emergency_stop",
            device_id=device_id,
            details={"reason": reason[:300]},
        )

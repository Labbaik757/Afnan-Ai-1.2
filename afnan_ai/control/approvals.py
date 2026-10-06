"""Remote approval gateway.

Bridges the existing :class:`~afnan_ai.activity.approval.ApprovalCenter`
to remote clients.  Approvals stay unique, short-lived, action-bound
and single-decision — the gateway only transports the request outward
and the user's decision inward.  A stale approval can never authorize
a different action: ``ApprovalCenter.decide()`` enforces that.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from afnan_ai.control import views


class ApprovalGateway:
    """Projects pending approvals to remote sessions and carries
    remote decisions back into the ApprovalCenter."""

    def __init__(
        self,
        approval_center,
        *,
        event_stream=None,
        on_remote_approval: (
            Callable[[dict[str, Any]], None] | None
        ) = None,
    ) -> None:
        self.approvals = approval_center
        self.events = event_stream
        self._on_remote_approval = on_remote_approval
        self._lock = threading.RLock()
        # approval_id -> set of session_ids that were notified
        self._notified: dict[str, set[str]] = {}

    # -- outward: notify remote sessions ----------------------------------------

    def pending_for_remote(self) -> list[dict[str, Any]]:
        if self.approvals is None:
            return []
        return [
            views.approval_view(a)
            for a in self.approvals.pending()
        ]

    def notify_sessions(
        self, session_ids: list[str]
    ) -> int:
        """Push pending approvals into the event stream for the
        given sessions.  Returns the number of approvals pushed."""
        if self.events is None or self.approvals is None:
            return 0
        count = 0
        for tracked in self.approvals.pending():
            view = views.approval_view(tracked)
            with self._lock:
                notified = self._notified.setdefault(
                    tracked.approval_id, set()
                )
            for session_id in session_ids:
                with self._lock:
                    if session_id in notified:
                        continue
                    notified.add(session_id)
                self.events.publish(
                    "approval.requested",
                    {"approval": view},
                    session_id=session_id,
                )
                count += 1
        return count

    # -- inward: remote decisions --------------------------------------------------

    def decide_remote(
        self,
        *,
        approval_id: str,
        granted: bool,
        device_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Apply a remote user's decision.  The ApprovalCenter owns
        all safety semantics (expiry, single decision, binding)."""
        if self.approvals is None:
            raise ValueError("approval center not wired")
        tracked = self.approvals.get(approval_id)
        if tracked is None:
            raise KeyError(
                f"unknown approval: {approval_id}"
            )
        outcome = self.approvals.decide(
            approval_id,
            granted=granted,
            actor=f"remote:{device_id}:{session_id}",
        )
        if self.events is not None:
            self.events.publish(
                "approval.resolved",
                {
                    "approval_id": approval_id,
                    "granted": granted,
                    "device_id": device_id,
                },
            )
        if self._on_remote_approval is not None:
            try:
                self._on_remote_approval(
                    {
                        "approval_id": approval_id,
                        "granted": granted,
                        "device_id": device_id,
                        "session_id": session_id,
                    }
                )
            except Exception:
                pass
        with self._lock:
            self._notified.pop(approval_id, None)
        return {
            "approval_id": approval_id,
            "granted": granted,
            "outcome": outcome,
        }

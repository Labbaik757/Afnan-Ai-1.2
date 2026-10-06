"""Control plane server.

Owns the lifecycle of a ``ControlPlane`` plus its transport.
Constructed from an existing agent (or its components) so the
server never builds parallel runtime state.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.control.plane import ControlPlane
from afnan_ai.control.transport import (
    ControlTransport,
    TransportError,
)


class ControlPlaneServer:
    """Lifecycle owner: plane + transport."""

    def __init__(
        self,
        plane: ControlPlane,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        tls_cert: str | None = None,
        tls_key: str | None = None,
        allow_insecure: bool = False,
        owner_token_path: str | None = None,
    ) -> None:
        self.plane = plane
        self.transport = ControlTransport(
            plane,
            host=host,
            port=port,
            tls_cert=tls_cert,
            tls_key=tls_key,
            allow_insecure=allow_insecure,
            owner_token_path=owner_token_path,
        )

    @classmethod
    def from_agent(
        cls,
        agent,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        tls_cert: str | None = None,
        tls_key: str | None = None,
        allow_insecure: bool = False,
        device_path: str | None = None,
        owner_token_path: str | None = None,
    ) -> "ControlPlaneServer":
        """Build a server from a live AfnanAgent.

        Every component is injected from the agent — nothing is
        duplicated.  Missing optional components simply leave the
        corresponding remote commands unwired.
        """
        security = getattr(agent, "security_center", None)
        if security is None:
            raise TransportError(
                "agent has no security_center; refusing to start "
                "a control plane without central authorization"
            )
        browser = getattr(agent, "browser", None)
        computer = getattr(agent, "computer", None)
        activity = getattr(agent, "activity_center", None)
        approvals = getattr(activity, "approvals", None)
        plane = ControlPlane(
            security_center=security,
            task_manager=getattr(agent, "task_manager", None),
            goal_manager=getattr(agent, "goal_manager", None),
            scheduler=getattr(agent, "scheduler", None),
            activity_center=activity,
            approval_center=approvals,
            artifact_manager=getattr(
                agent, "artifact_manager", None
            ),
            browser_controller=browser,
            computer_runtime=computer,
            device_path=device_path,
        )
        if owner_token_path is None and device_path:
            from pathlib import Path

            owner_token_path = str(
                Path(device_path).expanduser().parent
                / "owner_token"
            )
        return cls(
            plane,
            host=host,
            port=port,
            tls_cert=tls_cert,
            tls_key=tls_key,
            allow_insecure=allow_insecure,
            owner_token_path=owner_token_path,
        )

    def start(self) -> None:
        self.plane.start()
        self.transport.start()

    def stop(self) -> None:
        try:
            self.transport.stop()
        finally:
            self.plane.stop()

    @property
    def url(self) -> str:
        return self.transport.url

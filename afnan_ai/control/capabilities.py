"""Remote control capability definitions.

Capability-based authorization for the control plane, reusing the
existing :class:`~afnan_ai.security.permissions.PermissionManager`
and :class:`~afnan_ai.security.capabilities.CapabilityManager`.
Nothing here duplicates the security layer: these are *definitions*
in the existing dotted capability namespace, registered through the
existing ``CapabilityManager.define()``.
"""

from __future__ import annotations

from afnan_ai.security.capabilities import Capability
from afnan_ai.security.models import RiskLevel

# Every remote capability lives under the ``remote.`` namespace so it
# can never collide with local agent capabilities.
NAMESPACE = "remote."

# capability id -> (risk level, description)
REMOTE_CAPABILITIES: dict[str, tuple[RiskLevel, str]] = {
    # Monitoring (read-only).
    "remote.status.read": (
        RiskLevel.READ_ONLY,
        "Read sanitized agent runtime status snapshots.",
    ),
    "remote.tasks.read": (
        RiskLevel.READ_ONLY,
        "List and inspect tasks, progress, failures, checkpoints.",
    ),
    "remote.goals.read": (
        RiskLevel.READ_ONLY,
        "Inspect goals, progress and milestones.",
    ),
    "remote.activity.read": (
        RiskLevel.READ_ONLY,
        "Subscribe to and replay the activity event stream.",
    ),
    "remote.artifacts.read": (
        RiskLevel.READ_ONLY,
        "List and read authorized artifacts.",
    ),
    "remote.devices.read": (
        RiskLevel.READ_ONLY,
        "List registered devices and sessions.",
    ),
    "remote.metrics.read": (
        RiskLevel.READ_ONLY,
        "Read control-plane metrics and health.",
    ),
    # Task control.
    "remote.tasks.control": (
        RiskLevel.SENSITIVE,
        "Start, pause, resume, retry and cancel tasks.",
    ),
    # Goal control.
    "remote.goals.control": (
        RiskLevel.SENSITIVE,
        "Pause, resume and request cancellation of goals.",
    ),
    # Approvals.
    "remote.approvals.decide": (
        RiskLevel.SENSITIVE,
        "Approve or deny pending approval requests.",
    ),
    # Browser / computer (routed through existing runtimes only).
    "remote.browser.read": (
        RiskLevel.READ_ONLY,
        "Inspect browser state, tabs and session status.",
    ),
    "remote.browser.control": (
        RiskLevel.SENSITIVE,
        "Navigate and interact via the browser runtime.",
    ),
    "remote.computer.observe": (
        RiskLevel.SENSITIVE,
        "Observe screen, window and application state.",
    ),
    "remote.computer.control": (
        RiskLevel.IRREVERSIBLE,
        "Issue input actions through the computer runtime.",
    ),
    # Scheduling.
    "remote.schedule.manage": (
        RiskLevel.SENSITIVE,
        "Create, pause, resume and cancel scheduled tasks.",
    ),
    # Device / session administration.
    "remote.devices.manage": (
        RiskLevel.SENSITIVE,
        "Revoke devices and sessions, rotate credentials.",
    ),
    # Safety.
    "remote.safety.emergency_stop": (
        RiskLevel.IRREVERSIBLE,
        "Trigger the emergency stop (audited, high priority).",
    ),
    # Full administration.  Never granted by default.
    "remote.admin": (
        RiskLevel.IRREVERSIBLE,
        "Full remote administration. Granted explicitly only.",
    ),
}


def define_remote_capabilities(capability_manager) -> list[str]:
    """Register remote capabilities; returns the registered ids."""
    registered: list[str] = []
    for capability_id, (risk, description) in (
        REMOTE_CAPABILITIES.items()
    ):
        try:
            capability_manager.require(capability_id)
        except Exception:
            capability_manager.define(
                Capability(
                    capability_id=capability_id,
                    name=capability_id.replace("remote.", "").replace(
                        ".", " "
                    ).title(),
                    description=description,
                    category="remote",
                    risk_level=risk,
                )
            )
        registered.append(capability_id)
    return registered


# Default capability set for a newly paired device: monitoring only.
# Anything mutating must be granted explicitly afterwards.
DEFAULT_PAIRED_CAPABILITIES: tuple[str, ...] = (
    "remote.status.read",
    "remote.tasks.read",
    "remote.goals.read",
    "remote.activity.read",
    "remote.artifacts.read",
    "remote.devices.read",
    "remote.metrics.read",
)

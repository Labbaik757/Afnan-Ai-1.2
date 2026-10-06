"""Control plane data model.

Strongly-typed models for the Afnan AI remote & multi-device control
plane.  Everything crossing the device boundary is validated here
before it touches the agent runtime.

The control plane never duplicates agent architecture: these models
are the *transport* shape.  Runtime state stays in TaskManager,
GoalManager, ActivityCenter, SecurityCenter and friends; the views
in :mod:`afnan_ai.control.views` project sanitized snapshots.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

PROTOCOL_VERSION = "v1"


def _now() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class SessionState(str, Enum):
    CONNECTING = "connecting"
    AUTHENTICATING = "authenticating"
    AUTHORIZED = "authorized"
    ACTIVE = "active"
    IDLE = "idle"
    RECONNECTING = "reconnecting"
    SUSPENDED = "suspended"
    REVOKED = "revoked"
    EXPIRED = "expired"
    CLOSED = "closed"


_VALID_TRANSITIONS: dict[SessionState, frozenset[SessionState]] = {
    SessionState.CONNECTING: frozenset(
        {SessionState.AUTHENTICATING, SessionState.CLOSED}
    ),
    SessionState.AUTHENTICATING: frozenset(
        {
            SessionState.AUTHORIZED,
            SessionState.CLOSED,
            SessionState.EXPIRED,
        }
    ),
    SessionState.AUTHORIZED: frozenset(
        {
            SessionState.ACTIVE,
            SessionState.IDLE,
            SessionState.SUSPENDED,
            SessionState.REVOKED,
            SessionState.EXPIRED,
            SessionState.CLOSED,
        }
    ),
    SessionState.ACTIVE: frozenset(
        {
            SessionState.IDLE,
            SessionState.RECONNECTING,
            SessionState.SUSPENDED,
            SessionState.REVOKED,
            SessionState.EXPIRED,
            SessionState.CLOSED,
        }
    ),
    SessionState.IDLE: frozenset(
        {
            SessionState.ACTIVE,
            SessionState.RECONNECTING,
            SessionState.SUSPENDED,
            SessionState.REVOKED,
            SessionState.EXPIRED,
            SessionState.CLOSED,
        }
    ),
    SessionState.RECONNECTING: frozenset(
        {
            SessionState.AUTHENTICATING,
            SessionState.ACTIVE,
            SessionState.REVOKED,
            SessionState.EXPIRED,
            SessionState.CLOSED,
        }
    ),
    SessionState.SUSPENDED: frozenset(
        {
            SessionState.AUTHENTICATING,
            SessionState.REVOKED,
            SessionState.EXPIRED,
            SessionState.CLOSED,
        }
    ),
    # Terminal states: no outgoing transitions.
    SessionState.REVOKED: frozenset(),
    SessionState.EXPIRED: frozenset(),
    SessionState.CLOSED: frozenset(),
}


def valid_session_transition(
    current: SessionState, target: SessionState
) -> bool:
    """True when the session state machine allows current -> target."""
    return target in _VALID_TRANSITIONS.get(current, frozenset())


class SessionTransitionError(ValueError):
    """Raised when a session state transition is not allowed."""


# ---------------------------------------------------------------------------
# Device trust
# ---------------------------------------------------------------------------


class DeviceTrust(str, Enum):
    UNPAIRED = "unpaired"
    PENDING = "pending"
    PAIRED = "paired"
    REVOKED = "revoked"


class ConnectionState(str, Enum):
    OFFLINE = "offline"
    CONNECTING = "connecting"
    ONLINE = "online"
    DEGRADED = "degraded"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


class CommandType(str, Enum):
    # Monitoring (read-only)
    GET_AGENT_STATUS = "get_agent_status"
    LIST_TASKS = "list_tasks"
    GET_TASK = "get_task"
    LIST_GOALS = "list_goals"
    GET_GOAL = "get_goal"
    LIST_ARTIFACTS = "list_artifacts"
    GET_ARTIFACT = "get_artifact"
    LIST_APPROVALS = "list_approvals"
    GET_APPROVAL = "get_approval"
    LIST_DEVICES = "list_devices"
    GET_DEVICE = "get_device"
    LIST_SESSIONS = "list_sessions"
    QUERY_ACTIVITY = "query_activity"
    GET_METRICS = "get_metrics"
    HEALTH_CHECK = "health_check"
    # Task control
    START_TASK = "start_task"
    PAUSE_TASK = "pause_task"
    RESUME_TASK = "resume_task"
    CANCEL_TASK = "cancel_task"
    RETRY_TASK = "retry_task"
    # Goal control
    PAUSE_GOAL = "pause_goal"
    RESUME_GOAL = "resume_goal"
    CANCEL_GOAL = "cancel_goal"
    # Approvals
    APPROVE_ACTION = "approve_action"
    DENY_ACTION = "deny_action"
    # Browser / computer (permission-gated, routed through existing runtimes)
    BROWSER_NAVIGATE = "browser_navigate"
    BROWSER_SCREENSHOT = "browser_screenshot"
    BROWSER_STATE = "browser_state"
    COMPUTER_OBSERVE = "computer_observe"
    COMPUTER_ACT = "computer_act"
    # Scheduling
    LIST_SCHEDULES = "list_schedules"
    MANAGE_SCHEDULE = "manage_schedule"
    # Safety
    EMERGENCY_STOP = "emergency_stop"
    # Session/device management
    HEARTBEAT = "heartbeat"
    REVOKE_DEVICE = "revoke_device"
    REVOKE_SESSION = "revoke_session"
    ROTATE_CREDENTIALS = "rotate_credentials"


class CommandStatus(str, Enum):
    ACCEPTED = "accepted"
    COMPLETED = "completed"
    FAILED = "failed"
    DENIED = "denied"
    EXPIRED = "expired"
    DUPLICATE = "duplicate"


class QueuePolicy(str, Enum):
    """Offline handling class for a command type."""

    SAFE_QUEUEABLE = "safe_queueable"
    EXPIRING = "expiring"
    REQUIRES_LIVE_SESSION = "requires_live_session"
    NEVER_QUEUEABLE = "never_queueable"


# Command types that may never be queued for later execution: they
# require a live session, fresh authorization and a fresh policy check.
_NEVER_QUEUEABLE = frozenset(
    {
        CommandType.APPROVE_ACTION,
        CommandType.DENY_ACTION,
        CommandType.EMERGENCY_STOP,
        CommandType.COMPUTER_ACT,
        CommandType.CANCEL_TASK,
        CommandType.CANCEL_GOAL,
        CommandType.REVOKE_DEVICE,
        CommandType.REVOKE_SESSION,
    }
)

# Read-only commands that are safe to answer from a queue/snapshot.
_SAFE_QUEUEABLE = frozenset(
    {
        CommandType.GET_AGENT_STATUS,
        CommandType.LIST_TASKS,
        CommandType.GET_TASK,
        CommandType.LIST_GOALS,
        CommandType.GET_GOAL,
        CommandType.LIST_ARTIFACTS,
        CommandType.QUERY_ACTIVITY,
        CommandType.GET_METRICS,
        CommandType.HEALTH_CHECK,
        CommandType.LIST_APPROVALS,
        CommandType.GET_APPROVAL,
        CommandType.LIST_DEVICES,
        CommandType.GET_DEVICE,
        CommandType.LIST_SESSIONS,
        CommandType.LIST_SCHEDULES,
        CommandType.BROWSER_STATE,
    }
)


def queue_policy_for(command_type: CommandType) -> QueuePolicy:
    """Offline queue policy for a command type (default-deny)."""
    if command_type in _NEVER_QUEUEABLE:
        return QueuePolicy.NEVER_QUEUEABLE
    if command_type in _SAFE_QUEUEABLE:
        return QueuePolicy.SAFE_QUEUEABLE
    return QueuePolicy.REQUIRES_LIVE_SESSION


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


class PairingState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    CONSUMED = "consumed"
    EXPIRED = "expired"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DeviceInfo:
    device_id: str
    device_name: str = ""
    platform: str = ""
    platform_version: str = ""
    client_version: str = ""
    agent_runtime_version: str = ""
    capabilities: tuple[str, ...] = ()
    connection_state: ConnectionState = ConnectionState.OFFLINE
    trust: DeviceTrust = DeviceTrust.UNPAIRED
    last_seen: float = 0.0
    registered_at: float = field(default_factory=_now)
    # SHA-256 of the device secret; the secret itself is never stored.
    secret_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["connection_state"] = self.connection_state.value
        data["trust"] = self.trust.value
        # The secret hash is internal; never serialized to clients.
        data.pop("secret_hash", None)
        return data


@dataclass
class ClientSession:
    session_id: str
    device_id: str
    principal: str = ""
    capabilities: tuple[str, ...] = ()
    state: SessionState = SessionState.CONNECTING
    created_at: float = field(default_factory=_now)
    last_activity: float = field(default_factory=_now)
    expires_at: float = 0.0
    correlation_id: str = ""
    # Token bookkeeping: only hashes/refs, never the token value.
    token_hash: str = ""
    token_ref: str = ""
    last_event_seq: int = 0
    heartbeat_at: float = 0.0
    missed_heartbeats: int = 0
    client_version: str = ""
    transport: str = ""

    def touch(self) -> None:
        self.last_activity = _now()

    def is_live(self) -> bool:
        return self.state in (
            SessionState.AUTHORIZED,
            SessionState.ACTIVE,
            SessionState.IDLE,
        )

    def is_expired(self, now: float | None = None) -> bool:
        now = _now() if now is None else now
        return bool(self.expires_at) and now >= self.expires_at

    def transition(self, target: SessionState) -> None:
        if not valid_session_transition(self.state, target):
            raise SessionTransitionError(
                f"invalid session transition "
                f"{self.state.value} -> {target.value}"
            )
        self.state = target
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        data.pop("token_hash", None)
        data.pop("token_ref", None)
        return data


@dataclass
class RemoteCommand:
    command_id: str = field(default_factory=lambda: _new_id("cmd"))
    session_id: str = ""
    device_id: str = ""
    actor_id: str = ""
    command_type: CommandType = CommandType.HEALTH_CHECK
    target: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    requested_capability: str = ""
    created_at: float = field(default_factory=_now)
    expires_at: float = 0.0
    idempotency_key: str = ""
    correlation_id: str = ""
    protocol_version: str = PROTOCOL_VERSION

    def is_expired(self, now: float | None = None) -> bool:
        now = _now() if now is None else now
        return bool(self.expires_at) and now >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["command_type"] = self.command_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RemoteCommand":
        known = set(cls.__dataclass_fields__)
        clean = {k: v for k, v in data.items() if k in known}
        raw_type = clean.get("command_type", CommandType.HEALTH_CHECK)
        try:
            clean["command_type"] = CommandType(raw_type)
        except ValueError:
            raise CommandValidationError(
                f"unknown command_type: {raw_type!r}"
            )
        payload = clean.get("payload", {})
        if not isinstance(payload, dict):
            raise CommandValidationError("payload must be an object")
        if len(str(payload)) > 65536:
            raise CommandValidationError("payload too large")
        return cls(**clean)


class CommandValidationError(ValueError):
    """Raised when a remote command fails schema validation."""


@dataclass
class RemoteCommandResult:
    command_id: str
    status: CommandStatus
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    error_code: str = ""
    verification: dict[str, Any] = field(default_factory=dict)
    completed_at: float = field(default_factory=_now)
    correlation_id: str = ""
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class PairingRequest:
    pairing_id: str = field(default_factory=lambda: _new_id("pair"))
    code: str = ""
    # SHA-256 of the code; the code itself is shown once, then dropped.
    code_hash: str = ""
    requesting_device_id: str = ""
    requesting_device_name: str = ""
    platform: str = ""
    requested_capabilities: tuple[str, ...] = ()
    state: PairingState = PairingState.PENDING
    created_at: float = field(default_factory=_now)
    expires_at: float = 0.0
    attempts: int = 0
    approved_by: str = ""
    consumed_at: float = 0.0

    def is_expired(self, now: float | None = None) -> bool:
        now = _now() if now is None else now
        return bool(self.expires_at) and now >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        data.pop("code", None)
        data.pop("code_hash", None)
        return data


@dataclass
class ControlEvent:
    """Versioned, structured event delivered to remote clients."""

    event_id: str = field(default_factory=lambda: _new_id("evt"))
    category: str = ""
    seq: int = 0
    at: float = field(default_factory=_now)
    device_id: str = ""
    session_id: str = ""
    correlation_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Event categories emitted by the control plane (subset mirrors the
# ActivityCenter vocabulary so clients see one coherent stream).
EVENT_CATEGORIES = frozenset(
    {
        "agent.status",
        "task.created",
        "task.started",
        "task.progress",
        "task.paused",
        "task.resumed",
        "task.completed",
        "task.failed",
        "task.cancelled",
        "approval.requested",
        "approval.resolved",
        "browser.state_changed",
        "computer.state_changed",
        "artifact.created",
        "artifact.updated",
        "security.alert",
        "device.connected",
        "device.disconnected",
        "checkpoint.created",
        "research.updated",
        "evaluation.completed",
        "session.revoked",
        "command.completed",
    }
)

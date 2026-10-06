"""Secure remote & multi-device control plane for Afnan AI.

Authorized clients and devices can monitor and control the running
agent runtime — tasks, goals, approvals, artifacts, activity, browser
and computer — through an authenticated, capability-authorized,
audited command channel.

The control plane reuses the existing architecture and never
duplicates it: SecurityCenter stays the central authority,
TaskManager/GoalManager stay the sources of truth, ActivityCenter
stays the event bus, and the existing AgentLoop/Executor/Verifier
pipeline stays authoritative for execution.
"""

from afnan_ai.control.approvals import ApprovalGateway
from afnan_ai.control.audit import ControlPlaneAuditAdapter
from afnan_ai.control.auth import AuthenticationError, AuthProvider
from afnan_ai.control.capabilities import (
    DEFAULT_PAIRED_CAPABILITIES,
    REMOTE_CAPABILITIES,
    define_remote_capabilities,
)
from afnan_ai.control.commands import (
    CommandRouterError,
    RemoteCommandRouter,
)
from afnan_ai.control.devices import DeviceRegistry
from afnan_ai.control.events import EventStream
from afnan_ai.control.idempotency import IdempotencyStore
from afnan_ai.control.metrics import MetricsCollector
from afnan_ai.control.models import (
    ClientSession,
    CommandStatus,
    CommandType,
    ControlEvent,
    DeviceInfo,
    DeviceTrust,
    PairingRequest,
    PairingState,
    QueuePolicy,
    RemoteCommand,
    RemoteCommandResult,
    SessionState,
    SessionTransitionError,
    queue_policy_for,
    valid_session_transition,
)
from afnan_ai.control.pairing import PairingError, PairingManager
from afnan_ai.control.plane import ControlPlane, ControlPlaneError
from afnan_ai.control.server import ControlPlaneServer
from afnan_ai.control.sessions import ClientSessionManager
from afnan_ai.control.transport import (
    ControlTransport,
    TransportError,
)
from afnan_ai.control import views
from afnan_ai.control import client

__all__ = [
    "ApprovalGateway",
    "AuthenticationError",
    "AuthProvider",
    "ClientSession",
    "ClientSessionManager",
    "CommandRouterError",
    "CommandStatus",
    "CommandType",
    "ControlEvent",
    "ControlPlane",
    "ControlPlaneAuditAdapter",
    "ControlPlaneError",
    "ControlPlaneServer",
    "ControlTransport",
    "DEFAULT_PAIRED_CAPABILITIES",
    "DeviceInfo",
    "DeviceRegistry",
    "DeviceTrust",
    "EventStream",
    "IdempotencyStore",
    "MetricsCollector",
    "PairingError",
    "PairingManager",
    "PairingRequest",
    "PairingState",
    "QueuePolicy",
    "REMOTE_CAPABILITIES",
    "RemoteCommand",
    "RemoteCommandResult",
    "RemoteCommandRouter",
    "SessionState",
    "SessionTransitionError",
    "TransportError",
    "define_remote_capabilities",
    "queue_policy_for",
    "valid_session_transition",
    "views",
    "client",
]

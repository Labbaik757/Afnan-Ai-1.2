"""Production-grade Activity Center + Agent Control System.

User-facing live and historical visibility into what the
agent is doing — without duplicating the existing
orchestration (AgentLoop, TaskManager, SecureWorkspace,
SecurityCenter/AuditCenter, ArtifactManager stay the
authorities; this package observes and controls through
them).
"""

from afnan_ai.activity.approval import ApprovalCenter
from afnan_ai.activity.center import ActivityCenter, EventStore
from afnan_ai.activity.control import (
    ControlCommand,
    TaskControlService,
)
from afnan_ai.activity.errors import PresentedError
from afnan_ai.activity.events import ActivityEvent
from afnan_ai.activity.evidence import Evidence
from afnan_ai.activity.integration import (
    ActivityIntegration,
    attach_agent_loop,
    build_integration,
)
from afnan_ai.activity.notifications import NotificationService
from afnan_ai.activity.query import (
    ActivityQueryService,
    ArtifactQueryService,
    TaskQueryService,
)
from afnan_ai.activity.status import AgentStatusTracker
from afnan_ai.activity.timeline import (
    ActivityFilter,
    ActivitySummary,
    ActivityTimeline,
    ProgressTracker,
    TaskProgress,
)
from afnan_ai.activity.voice import parse_voice_command

__all__ = [
    "ActivityCenter",
    "EventStore",
    "ActivityEvent",
    "ActivityFilter",
    "ActivityTimeline",
    "ActivitySummary",
    "TaskProgress",
    "ProgressTracker",
    "AgentStatusTracker",
    "ControlCommand",
    "TaskControlService",
    "ApprovalCenter",
    "TaskQueryService",
    "ActivityQueryService",
    "ArtifactQueryService",
    "NotificationService",
    "PresentedError",
    "Evidence",
    "ActivityIntegration",
    "attach_agent_loop",
    "build_integration",
    "parse_voice_command",
]

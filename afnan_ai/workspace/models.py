"""Secure Agent Workspace — models.

OS-independent workspace abstraction: every long-running
task gets an isolated, persistent, policy-controlled
execution environment.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WorkspaceStatus(str, Enum):
    CREATING = "creating"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    RECOVERING = "recovering"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    COMPLETED = "completed"


class CleanupMode(str, Enum):
    EPHEMERAL = "ephemeral"  # destroy when task ends
    PERSISTENT = "persistent"  # keep until manual delete
    UNTIL_TASK_COMPLETE = "until_task_complete"
    UNTIL_MANUAL_DELETE = "until_manual_delete"


@dataclass
class WorkspaceResourceLimits:
    """Per-workspace configurable resource limits."""

    max_execution_time_s: float = 3600.0
    max_task_duration_s: float = 7200.0
    max_cpu_percent: float = 80.0
    max_memory_mb: float = 2048.0
    max_disk_mb: float = 2048.0
    max_download_mb: float = 512.0
    max_network_requests: int = 1000
    max_subprocesses: int = 16
    max_browser_tabs: int = 10
    max_retries: int = 3
    max_loop_iterations: int = 50

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_execution_time_s": self.max_execution_time_s,
            "max_task_duration_s": self.max_task_duration_s,
            "max_cpu_percent": self.max_cpu_percent,
            "max_memory_mb": self.max_memory_mb,
            "max_disk_mb": self.max_disk_mb,
            "max_download_mb": self.max_download_mb,
            "max_network_requests": self.max_network_requests,
            "max_subprocesses": self.max_subprocesses,
            "max_browser_tabs": self.max_browser_tabs,
            "max_retries": self.max_retries,
            "max_loop_iterations": self.max_loop_iterations,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "WorkspaceResourceLimits":
        kwargs = {}
        for f in cls.__dataclass_fields__:
            if f in data:
                kwargs[f] = data[f]
        return cls(**kwargs)


@dataclass
class WorkspaceNetworkPolicy:
    """Network isolation policy.  Default-deny unless
    explicitly allowed."""

    default_allow: bool = False
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    allowed_protocols: tuple[str, ...] = ("https",)
    max_upload_mb: float = 64.0
    max_download_mb: float = 512.0
    connector_permissions: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )

    def allows_domain(self, domain: str) -> bool:
        domain = domain.lower().strip()
        for blocked in self.blocked_domains:
            b = blocked.lower().strip()
            if domain == b or domain.endswith("." + b):
                return False
        if not self.allowed_domains:
            return self.default_allow
        for allowed in self.allowed_domains:
            a = allowed.lower().strip()
            if domain == a or domain.endswith("." + a):
                return True
        return False

    def allows_protocol(self, protocol: str) -> bool:
        return protocol.lower().strip() in {
            p.lower() for p in self.allowed_protocols
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_allow": self.default_allow,
            "allowed_domains": list(self.allowed_domains),
            "blocked_domains": list(self.blocked_domains),
            "allowed_protocols": list(self.allowed_protocols),
            "max_upload_mb": self.max_upload_mb,
            "max_download_mb": self.max_download_mb,
            "connector_permissions": {
                k: list(v)
                for k, v in self.connector_permissions.items()
            },
        }


@dataclass
class WorkspacePolicy:
    """Authorization policy for a workspace."""

    allowed_capabilities: tuple[str, ...] = ()
    denied_capabilities: tuple[str, ...] = ()
    allowed_apps: tuple[str, ...] = ()
    allowed_paths: tuple[str, ...] = ()
    sensitive_paths: tuple[str, ...] = ()
    require_approval_for: tuple[str, ...] = (
        "filesystem.delete",
        "connector.email.send",
        "browser.purchase",
        "computer.input",
    )
    allow_cross_workspace_handoff: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_capabilities": list(
                self.allowed_capabilities
            ),
            "denied_capabilities": list(
                self.denied_capabilities
            ),
            "allowed_apps": list(self.allowed_apps),
            "allowed_paths": list(self.allowed_paths),
            "sensitive_paths": list(self.sensitive_paths),
            "require_approval_for": list(
                self.require_approval_for
            ),
            "allow_cross_workspace_handoff": (
                self.allow_cross_workspace_handoff
            ),
        }


@dataclass
class WorkspaceConfig:
    """Configuration for creating a workspace."""

    task_id: str = ""
    owner: str = "agent:main"
    cleanup_mode: CleanupMode = (
        CleanupMode.UNTIL_TASK_COMPLETE
    )
    resource_limits: WorkspaceResourceLimits = field(
        default_factory=WorkspaceResourceLimits
    )
    network_policy: WorkspaceNetworkPolicy = field(
        default_factory=WorkspaceNetworkPolicy
    )
    policy: WorkspacePolicy = field(
        default_factory=WorkspacePolicy
    )
    environment: dict[str, str] = field(default_factory=dict)
    base_dir: str = ""  # "" → platform default
    retain_on_failure: bool = True


@dataclass
class Workspace:
    """A persistent, isolated execution environment."""

    workspace_id: str
    task_id: str
    owner: str
    root_dir: str
    status: WorkspaceStatus = WorkspaceStatus.CREATING
    config: WorkspaceConfig = field(
        default_factory=WorkspaceConfig
    )
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    session_id: str = ""
    checkpoint_id: str = ""
    # Subdirectory names (relative to root_dir).
    dirs: dict[str, str] = field(default_factory=lambda: {
        "files": "files",
        "downloads": "downloads",
        "artifacts": "artifacts",
        "browser": "browser",
        "tmp": "tmp",
        "logs": "logs",
        "checkpoints": "checkpoints",
    })
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def work_dir(self) -> str:
        import os

        return os.path.join(self.root_dir, self.dirs["files"])

    @property
    def download_dir(self) -> str:
        import os

        return os.path.join(
            self.root_dir, self.dirs["downloads"]
        )

    @property
    def artifact_dir(self) -> str:
        import os

        return os.path.join(
            self.root_dir, self.dirs["artifacts"]
        )

    @property
    def browser_profile_dir(self) -> str:
        import os

        return os.path.join(
            self.root_dir, self.dirs["browser"], "profile"
        )

    @property
    def tmp_dir(self) -> str:
        import os

        return os.path.join(self.root_dir, self.dirs["tmp"])

    @property
    def log_dir(self) -> str:
        import os

        return os.path.join(self.root_dir, self.dirs["logs"])

    @property
    def checkpoint_dir(self) -> str:
        import os

        return os.path.join(
            self.root_dir, self.dirs["checkpoints"]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "task_id": self.task_id,
            "owner": self.owner,
            "root_dir": self.root_dir,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "session_id": self.session_id,
            "checkpoint_id": self.checkpoint_id,
            "dirs": dict(self.dirs),
            "metadata": dict(self.metadata),
        }


@dataclass
class WorkspaceSession:
    """A live execution session inside a workspace."""

    session_id: str = field(
        default_factory=lambda: f"sess-{uuid.uuid4().hex[:12]}"
    )
    workspace_id: str = ""
    task_id: str = ""
    started_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    loop_iterations: int = 0
    actions_executed: int = 0
    actions_failed: int = 0
    approvals_requested: int = 0
    checkpoints_taken: int = 0
    recoveries: int = 0

    def heartbeat(self) -> None:
        self.last_heartbeat = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "task_id": self.task_id,
            "started_at": self.started_at,
            "last_heartbeat": self.last_heartbeat,
            "loop_iterations": self.loop_iterations,
            "actions_executed": self.actions_executed,
            "actions_failed": self.actions_failed,
            "approvals_requested": self.approvals_requested,
            "checkpoints_taken": self.checkpoints_taken,
            "recoveries": self.recoveries,
        }


@dataclass
class WorkspaceSnapshot:
    """A restorable snapshot of workspace state."""

    snapshot_id: str = field(
        default_factory=lambda: f"snap-{uuid.uuid4().hex[:12]}"
    )
    workspace_id: str = ""
    task_id: str = ""
    created_at: float = field(default_factory=time.time)
    agent_state_ref: str = ""
    current_goal: str = ""
    current_task: str = ""
    completed_steps: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    browser_state_ref: str = ""
    files_metadata: dict[str, Any] = field(default_factory=dict)
    artifact_refs: list[str] = field(default_factory=list)
    connector_refs: list[str] = field(default_factory=list)
    recovery_info: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "workspace_id": self.workspace_id,
            "task_id": self.task_id,
            "created_at": self.created_at,
            "agent_state_ref": self.agent_state_ref,
            "current_goal": self.current_goal,
            "current_task": self.current_task,
            "completed_steps": list(self.completed_steps),
            "pending_steps": list(self.pending_steps),
            "browser_state_ref": self.browser_state_ref,
            "files_metadata": dict(self.files_metadata),
            "artifact_refs": list(self.artifact_refs),
            "connector_refs": list(self.connector_refs),
            "recovery_info": dict(self.recovery_info),
            "checksum": self.checksum,
        }


@dataclass
class WorkspaceEvent:
    """Structured, redacted workspace activity event."""

    event_id: str = field(
        default_factory=lambda: f"wevt-{uuid.uuid4().hex[:12]}"
    )
    workspace_id: str = ""
    task_id: str = ""
    event_type: str = ""
    at: float = field(default_factory=time.time)
    actor: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "workspace_id": self.workspace_id,
            "task_id": self.task_id,
            "event_type": self.event_type,
            "at": self.at,
            "actor": self.actor,
            "details": dict(self.details),
        }


def new_workspace_id() -> str:
    return f"ws-{uuid.uuid4().hex[:12]}"

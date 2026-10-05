"""Secure Agent Workspace — persistent, isolated execution."""

from .file_manager import FileOpResult, WorkspaceFileManager, WorkspacePathError
from .health import HealthEvent, HealthStatus, WorkspaceHealthMonitor
from .integration import (
    WorkspaceTaskBinding,
    bind_workspace_sandbox,
    scoped_tool_runner,
    workspace_artifact_workspace_kwargs,
    workspace_browser_runtime_kwargs,
)
from .manager import WorkspaceError, WorkspaceManager
from .models import (
    CleanupMode,
    Workspace,
    WorkspaceConfig,
    WorkspaceEvent,
    WorkspaceNetworkPolicy,
    WorkspacePolicy,
    WorkspaceResourceLimits,
    WorkspaceSession,
    WorkspaceSnapshot,
    WorkspaceStatus,
    new_workspace_id,
)
from .network import NetworkDecision, WorkspaceNetworkGuard
from .platform import (
    BaseWorkspaceAdapter,
    LinuxWorkspaceAdapter,
    MacOSWorkspaceAdapter,
    WindowsWorkspaceAdapter,
    current_platform,
    get_adapter,
)
from .sandbox import (
    SandboxedOperation,
    SandboxVerdict,
    WorkspaceExecutionSandbox,
)

__all__ = [
    "WorkspaceManager",
    "WorkspaceError",
    "Workspace",
    "WorkspaceConfig",
    "WorkspaceSession",
    "WorkspaceSnapshot",
    "WorkspacePolicy",
    "WorkspaceResourceLimits",
    "WorkspaceNetworkPolicy",
    "WorkspaceEvent",
    "WorkspaceStatus",
    "CleanupMode",
    "new_workspace_id",
    "WorkspaceFileManager",
    "WorkspacePathError",
    "FileOpResult",
    "WorkspaceNetworkGuard",
    "NetworkDecision",
    "WorkspaceExecutionSandbox",
    "SandboxedOperation",
    "SandboxVerdict",
    "WorkspaceHealthMonitor",
    "HealthStatus",
    "HealthEvent",
    "BaseWorkspaceAdapter",
    "LinuxWorkspaceAdapter",
    "MacOSWorkspaceAdapter",
    "WindowsWorkspaceAdapter",
    "get_adapter",
    "current_platform",
    "WorkspaceTaskBinding",
    "bind_workspace_sandbox",
    "scoped_tool_runner",
    "workspace_browser_runtime_kwargs",
    "workspace_artifact_workspace_kwargs",
]

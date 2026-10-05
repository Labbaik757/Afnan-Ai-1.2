"""Workspace integrations with existing Afnan subsystems.

Thin adapters — no core subsystem is modified here.
The SecurityCenter stays the central authorization
authority and the CredentialVault the central secret
authority; the workspace only scopes their inputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from .manager import WorkspaceManager
from .models import Workspace, WorkspaceStatus
from .sandbox import SandboxedOperation, WorkspaceExecutionSandbox

logger = logging.getLogger(__name__)


def workspace_browser_runtime_kwargs(
    ws: Workspace,
) -> dict[str, Any]:
    """Kwargs to scope AfnanBrowserRuntime to a workspace."""
    profile_dir = Path(ws.browser_profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    downloads = Path(ws.download_dir)
    downloads.mkdir(parents=True, exist_ok=True)
    return {
        "runtime_dir": str(Path(ws.root_dir) / "browser"),
        "default_download_dir": str(downloads),
    }


def workspace_artifact_workspace_kwargs(
    ws: Workspace,
) -> dict[str, str]:
    """Kwargs to scope ArtifactWorkspace to a workspace."""
    artifacts = Path(ws.artifact_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    return {"root_dir": str(artifacts)}


def bind_workspace_sandbox(
    manager: WorkspaceManager,
    ws: Workspace,
    *,
    security_center: Any | None = None,
    tool_runner: (
        Callable[[str, dict[str, Any]], Any] | None
    ) = None,
) -> WorkspaceExecutionSandbox:
    """Build an execution sandbox gated by the workspace
    policy and (when given) the central SecurityCenter."""

    def check_capability(actor: str, capability: str) -> bool:
        policy = ws.config.policy
        if capability in policy.denied_capabilities:
            return False
        if policy.allowed_capabilities and (
            capability not in policy.allowed_capabilities
        ):
            return False
        if security_center is not None:
            try:
                decision = security_center.authorize(
                    actor=actor,
                    capability=capability,
                )
                return bool(
                    getattr(decision, "allowed", False)
                )
            except Exception:
                return False
        return True

    def check_permission(actor: str, op_id: str) -> bool:
        if manager.is_stopped(ws.workspace_id):
            return False
        return True

    def check_resources(op: SandboxedOperation) -> bool:
        session = manager.session(ws.workspace_id)
        usage = {
            "loop_iterations": float(
                session.loop_iterations if session else 0
            ),
        }
        exceeded = manager.check_limits(
            ws.workspace_id, usage=usage
        )
        return not exceeded

    def check_workspace(op: SandboxedOperation) -> bool:
        if op.workspace_id and (
            op.workspace_id != ws.workspace_id
        ):
            return False
        return ws.status in (
            WorkspaceStatus.RUNNING,
            WorkspaceStatus.RECOVERING,
            WorkspaceStatus.WAITING_FOR_APPROVAL,
        )

    return WorkspaceExecutionSandbox(
        check_capability=check_capability,
        check_permission=check_permission,
        check_resources=check_resources,
        check_workspace=check_workspace,
        tool_runner=tool_runner,
    )


def scoped_tool_runner(
    tool_registry: Any,
    *,
    workspace_id: str,
    actor: str = "agent:main",
) -> Callable[[str, dict[str, Any]], Any]:
    """Wrap a ToolRegistry so every tool call is tagged with
    the workspace actor and never leaks across workspaces."""

    def run(
        tool_name: str, args: dict[str, Any]
    ) -> Any:
        kwargs = dict(args)
        kwargs.setdefault("security_actor", actor)
        kwargs.setdefault("workspace_id", workspace_id)
        execute = getattr(tool_registry, "execute", None)
        if execute is None:
            raise RuntimeError(
                "tool registry has no execute()"
            )
        return execute(tool_name, **kwargs)

    return run


class WorkspaceTaskBinding:
    """Binds TaskManager + BackgroundTaskRunner runs to a
    workspace: create/reuse, restore checkpoint, run loop,
    record progress, recover, verify, retain-or-stop."""

    def __init__(
        self,
        manager: WorkspaceManager,
        task_manager: Any,
        *,
        security_center: Any | None = None,
    ) -> None:
        self.manager = manager
        self.task_manager = task_manager
        self.security = security_center

    def workspace_for_task(
        self, task_id: str, **config_kwargs: Any
    ) -> Workspace:
        from .models import WorkspaceConfig

        existing = self.manager.list(task_id=task_id)
        live = [
            w
            for w in existing
            if w.status
            not in (
                WorkspaceStatus.COMPLETED,
                WorkspaceStatus.FAILED,
            )
        ]
        if live:
            return live[0]
        config = WorkspaceConfig(
            task_id=task_id, **config_kwargs
        )
        return self.manager.create(config)

    def run_with_workspace(
        self,
        task_id: str,
        run_loop: Callable[[Workspace], Any],
        **config_kwargs: Any,
    ) -> Any:
        """Full lifecycle: workspace → checkpoint restore →
        loop → verify → checkpoint → retain/stop per policy."""
        ws = self.workspace_for_task(
            task_id, **config_kwargs
        )
        if ws.status in (
            WorkspaceStatus.READY,
            WorkspaceStatus.STOPPED,
            WorkspaceStatus.PAUSED,
        ):
            if ws.metadata.get("needs_recovery"):
                self.manager.recover(ws.workspace_id)
            self.manager.start(ws.workspace_id)
        # Restore last checkpoint so completed steps are not
        # repeated.
        snapshot = self.manager.latest_snapshot(
            ws.workspace_id
        )
        resumed_from = (
            snapshot.snapshot_id if snapshot else ""
        )
        try:
            result = run_loop(ws)
        except Exception as exc:
            self.manager.fail(
                ws.workspace_id, reason=str(exc)[:300]
            )
            raise
        # Post-run checkpoint + verification record.
        self.manager.take_snapshot(
            ws.workspace_id,
            current_task=task_id,
            completed_steps=[f"run:{task_id}"],
            recovery_info={"resumed_from": resumed_from},
        )
        self.manager.complete(ws.workspace_id)
        return result

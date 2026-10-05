"""WorkspaceManager — persistent workspace lifecycle.

Owns creation, start/pause/resume/stop/restart/recover/
destroy, snapshots, crash recovery, emergency stop,
cleanup policy and the structured audit trail.

Everything is persisted under the workspace root and the
manager registry so a machine restart can recover.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..redaction import redact_value
from .file_manager import WorkspaceFileManager, WorkspacePathError
from .health import WorkspaceHealthMonitor
from .models import (
    CleanupMode,
    Workspace,
    WorkspaceConfig,
    WorkspaceEvent,
    WorkspaceSession,
    WorkspaceSnapshot,
    WorkspaceStatus,
    new_workspace_id,
)
from .network import WorkspaceNetworkGuard
from .platform import BaseWorkspaceAdapter, get_adapter

logger = logging.getLogger(__name__)

_REGISTRY_FILE = "registry.json"
_EVENTS_FILE = "events.jsonl"
_LOCK_FILE = ".lock"


class WorkspaceError(Exception):
    """Workspace lifecycle error."""


def _snapshot_checksum(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class WorkspaceManager:
    """Manages multiple isolated, persistent workspaces."""

    _TRANSITIONS: dict[WorkspaceStatus, set[WorkspaceStatus]] = {
        WorkspaceStatus.CREATING: {
            WorkspaceStatus.READY,
            WorkspaceStatus.FAILED,
        },
        WorkspaceStatus.READY: {
            WorkspaceStatus.RUNNING,
            WorkspaceStatus.STOPPING,
            WorkspaceStatus.FAILED,
        },
        WorkspaceStatus.RUNNING: {
            WorkspaceStatus.PAUSED,
            WorkspaceStatus.WAITING_FOR_APPROVAL,
            WorkspaceStatus.RECOVERING,
            WorkspaceStatus.STOPPING,
            WorkspaceStatus.FAILED,
            WorkspaceStatus.COMPLETED,
        },
        WorkspaceStatus.PAUSED: {
            WorkspaceStatus.RUNNING,
            WorkspaceStatus.STOPPING,
            WorkspaceStatus.FAILED,
        },
        WorkspaceStatus.WAITING_FOR_APPROVAL: {
            WorkspaceStatus.RUNNING,
            WorkspaceStatus.PAUSED,
            WorkspaceStatus.STOPPING,
            WorkspaceStatus.FAILED,
        },
        WorkspaceStatus.RECOVERING: {
            WorkspaceStatus.RUNNING,
            WorkspaceStatus.PAUSED,
            WorkspaceStatus.FAILED,
        },
        WorkspaceStatus.STOPPING: {
            WorkspaceStatus.STOPPED,
            WorkspaceStatus.FAILED,
        },
        WorkspaceStatus.STOPPED: {
            WorkspaceStatus.RUNNING,  # restart
            WorkspaceStatus.RECOVERING,
        },
        WorkspaceStatus.FAILED: {
            WorkspaceStatus.RECOVERING,
            WorkspaceStatus.STOPPING,
        },
        WorkspaceStatus.COMPLETED: set(),
    }

    def __init__(
        self,
        base_dir: str | Path | None = None,
        *,
        adapter: BaseWorkspaceAdapter | None = None,
        health_monitor: WorkspaceHealthMonitor | None = None,
        on_event: (
            Callable[[WorkspaceEvent], None] | None
        ) = None,
    ) -> None:
        self.adapter = adapter or get_adapter()
        if base_dir:
            self.base_dir = Path(base_dir).expanduser()
        else:
            self.base_dir = self.adapter.default_base_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.health = (
            health_monitor
            or WorkspaceHealthMonitor(on_event=on_event)
        )
        self.on_event = on_event
        self._lock = threading.RLock()
        self._workspaces: dict[str, Workspace] = {}
        self._sessions: dict[str, WorkspaceSession] = {}
        self._guards: dict[str, WorkspaceNetworkGuard] = {}
        self._file_managers: dict[
            str, WorkspaceFileManager
        ] = {}
        # Emergency stop: {"global": bool, workspace_id: bool}
        self._emergency: dict[str, bool] = {}
        self._load_registry()

    # -- registry persistence ---------------------------------------

    def _registry_path(self) -> Path:
        return self.base_dir / _REGISTRY_FILE

    def _load_registry(self) -> None:
        path = self._registry_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("workspace registry unreadable")
            return
        for entry in data.get("workspaces", []):
            try:
                cfg = WorkspaceConfig()
                ws = Workspace(
                    workspace_id=entry["workspace_id"],
                    task_id=entry.get("task_id", ""),
                    owner=entry.get("owner", "agent:main"),
                    root_dir=entry["root_dir"],
                    status=WorkspaceStatus(
                        entry.get("status", "stopped")
                    ),
                    config=cfg,
                    created_at=entry.get("created_at", 0.0),
                    updated_at=entry.get("updated_at", 0.0),
                    session_id=entry.get("session_id", ""),
                    checkpoint_id=entry.get("checkpoint_id", ""),
                    dirs=entry.get("dirs", Workspace(
                        workspace_id="", task_id="",
                        owner="", root_dir="",
                    ).dirs),
                    metadata=entry.get("metadata", {}),
                )
                # Crash during RUNNING → mark for recovery.
                if ws.status in (
                    WorkspaceStatus.RUNNING,
                    WorkspaceStatus.RECOVERING,
                    WorkspaceStatus.WAITING_FOR_APPROVAL,
                ):
                    ws.status = WorkspaceStatus.STOPPED
                    ws.metadata["needs_recovery"] = True
                self._workspaces[ws.workspace_id] = ws
            except Exception:
                continue

    def _save_registry(self) -> None:
        with self._lock:
            data = {
                "saved_at": time.time(),
                "workspaces": [
                    w.to_dict()
                    for w in self._workspaces.values()
                ],
            }
        tmp = self._registry_path().with_suffix(".tmp")
        tmp.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        os.replace(tmp, self._registry_path())

    # -- events / audit ----------------------------------------------

    def _emit(
        self,
        ws: Workspace,
        event_type: str,
        *,
        actor: str = "workspace_manager",
        **details: Any,
    ) -> WorkspaceEvent:
        event = WorkspaceEvent(
            workspace_id=ws.workspace_id,
            task_id=ws.task_id,
            event_type=event_type,
            actor=actor,
            details=dict(redact_value(details)),
        )
        path = self.base_dir / _EVENTS_FILE
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(event.to_dict(),
                               ensure_ascii=False) + "\n"
                )
        except OSError:
            pass
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                pass
        return event

    # -- lifecycle ----------------------------------------------------

    def create(
        self, config: WorkspaceConfig | None = None
    ) -> Workspace:
        config = config or WorkspaceConfig()
        wid = new_workspace_id()
        root = (
            Path(config.base_dir).expanduser()
            if config.base_dir
            else self.base_dir / wid
        )
        ws = Workspace(
            workspace_id=wid,
            task_id=config.task_id,
            owner=config.owner,
            root_dir=str(root),
            config=config,
        )
        self.adapter.prepare_root(root)
        for sub in ws.dirs.values():
            (root / sub).mkdir(parents=True, exist_ok=True)
        # Workspace-local lock file (crash detection).
        (root / _LOCK_FILE).write_text(
            json.dumps(
                {"workspace_id": wid, "pid": os.getpid()}
            ),
            encoding="utf-8",
        )
        ws.status = WorkspaceStatus.READY
        ws.updated_at = time.time()
        with self._lock:
            self._workspaces[wid] = ws
            self._guards[wid] = WorkspaceNetworkGuard(
                config.network_policy,
                max_requests=(
                    config.resource_limits.max_network_requests
                ),
                max_upload_mb=(
                    config.network_policy.max_upload_mb
                ),
                max_download_mb=(
                    config.network_policy.max_download_mb
                ),
            )
            self._file_managers[wid] = WorkspaceFileManager(
                root,
                sensitive_paths=config.policy.sensitive_paths,
            )
        self._save_registry()
        self._emit(ws, "workspace.created", task_id=config.task_id)
        logger.info("workspace created: %s", wid)
        return ws

    def get(self, workspace_id: str) -> Workspace:
        with self._lock:
            ws = self._workspaces.get(workspace_id)
        if ws is None:
            raise WorkspaceError(
                f"unknown workspace: {workspace_id}"
            )
        return ws

    def list(
        self, *, task_id: str = "",
        status: WorkspaceStatus | None = None,
    ) -> list[Workspace]:
        with self._lock:
            result = list(self._workspaces.values())
        if task_id:
            result = [w for w in result if w.task_id == task_id]
        if status is not None:
            result = [w for w in result if w.status == status]
        return result

    def _transition(
        self, ws: Workspace, target: WorkspaceStatus
    ) -> None:
        allowed = self._TRANSITIONS.get(ws.status, set())
        if target not in allowed:
            raise WorkspaceError(
                f"illegal transition "
                f"{ws.status.value} -> {target.value}"
            )
        ws.status = target
        ws.updated_at = time.time()

    def start(self, workspace_id: str) -> WorkspaceSession:
        ws = self.get(workspace_id)
        self._assert_not_stopped(ws)
        with self._lock:
            self._transition(ws, WorkspaceStatus.RUNNING)
            session = WorkspaceSession(
                workspace_id=ws.workspace_id,
                task_id=ws.task_id,
            )
            self._sessions[session.session_id] = session
            ws.session_id = session.session_id
        self._save_registry()
        self._emit(ws, "task.started", session_id=session.session_id)
        return session

    def pause(self, workspace_id: str) -> None:
        ws = self.get(workspace_id)
        with self._lock:
            self._transition(ws, WorkspaceStatus.PAUSED)
        self._save_registry()
        self._emit(ws, "workspace.paused")

    def resume(self, workspace_id: str) -> WorkspaceSession:
        ws = self.get(workspace_id)
        self._assert_not_stopped(ws)
        with self._lock:
            if ws.status == WorkspaceStatus.PAUSED:
                self._transition(ws, WorkspaceStatus.RUNNING)
            elif ws.status == WorkspaceStatus.STOPPED:
                self._transition(ws, WorkspaceStatus.RUNNING)
            else:
                raise WorkspaceError(
                    f"cannot resume from {ws.status.value}"
                )
            session = WorkspaceSession(
                workspace_id=ws.workspace_id,
                task_id=ws.task_id,
            )
            self._sessions[session.session_id] = session
            ws.session_id = session.session_id
        self._save_registry()
        self._emit(
            ws, "workspace.resumed",
            session_id=session.session_id,
        )
        return session

    def stop(
        self, workspace_id: str, *, reason: str = ""
    ) -> None:
        ws = self.get(workspace_id)
        with self._lock:
            if ws.status not in (
                WorkspaceStatus.STOPPING,
                WorkspaceStatus.STOPPED,
                WorkspaceStatus.COMPLETED,
            ):
                self._transition(ws, WorkspaceStatus.STOPPING)
                self._transition(ws, WorkspaceStatus.STOPPED)
        self._release_lock(ws)
        self._save_registry()
        self._emit(ws, "workspace.stopped", reason=reason)
        self._maybe_cleanup(ws)

    def restart(self, workspace_id: str) -> WorkspaceSession:
        """Restart and resume from the last valid checkpoint."""
        ws = self.get(workspace_id)
        with self._lock:
            if ws.status not in (
                WorkspaceStatus.STOPPED,
                WorkspaceStatus.FAILED,
                WorkspaceStatus.PAUSED,
            ):
                raise WorkspaceError(
                    f"cannot restart from {ws.status.value}"
                )
            self._transition(ws, WorkspaceStatus.RECOVERING)
        snapshot = self.latest_snapshot(workspace_id)
        if snapshot is not None:
            ok = self._verify_snapshot(snapshot)
            if not ok:
                raise WorkspaceError(
                    "latest snapshot failed integrity check"
                )
            ws.checkpoint_id = snapshot.snapshot_id
            ws.metadata["resumed_from"] = snapshot.snapshot_id
        self._cleanup_stale_locks(ws)
        with self._lock:
            self._transition(ws, WorkspaceStatus.RUNNING)
            session = WorkspaceSession(
                workspace_id=ws.workspace_id,
                task_id=ws.task_id,
            )
            session.recoveries += 1
            self._sessions[session.session_id] = session
            ws.session_id = session.session_id
        self._save_registry()
        self._emit(
            ws, "workspace.restarted",
            resumed_from=ws.checkpoint_id,
        )
        return session

    def recover(self, workspace_id: str) -> WorkspaceSnapshot | None:
        """Crash recovery: verify integrity, clean stale locks,
        identify the last valid checkpoint and the incomplete
        action — without blindly repeating it."""
        ws = self.get(workspace_id)
        with self._lock:
            self._transition(ws, WorkspaceStatus.RECOVERING)
        self._emit(ws, "recovery.triggered")
        self._cleanup_stale_locks(ws)
        snapshot = self.latest_snapshot(workspace_id)
        if snapshot is None:
            with self._lock:
                self._transition(ws, WorkspaceStatus.FAILED)
            self._save_registry()
            self._emit(ws, "recovery.failed",
                       reason="no valid snapshot")
            raise WorkspaceError("no snapshot to recover from")
        if not self._verify_snapshot(snapshot):
            with self._lock:
                self._transition(ws, WorkspaceStatus.FAILED)
            self._save_registry()
            self._emit(ws, "recovery.failed",
                       reason="snapshot integrity check failed")
            raise WorkspaceError(
                "snapshot integrity check failed"
            )
        # Identify incomplete action for observation-driven resume.
        recovery_info = dict(snapshot.recovery_info)
        recovery_info["incomplete_action"] = (
            snapshot.pending_steps[0]
            if snapshot.pending_steps else None
        )
        recovery_info["completed_steps"] = list(
            snapshot.completed_steps
        )
        ws.checkpoint_id = snapshot.snapshot_id
        ws.metadata["recovery_info"] = redact_value(recovery_info)
        with self._lock:
            self._transition(ws, WorkspaceStatus.PAUSED)
        self._save_registry()
        self._emit(
            ws, "recovery.completed",
            snapshot_id=snapshot.snapshot_id,
            incomplete_action=recovery_info["incomplete_action"],
        )
        return snapshot

    def destroy(self, workspace_id: str) -> None:
        ws = self.get(workspace_id)
        with self._lock:
            if ws.status in (
                WorkspaceStatus.RUNNING,
                WorkspaceStatus.PAUSED,
                WorkspaceStatus.WAITING_FOR_APPROVAL,
            ):
                raise WorkspaceError(
                    "stop the workspace before destroying it"
                )
        self._revoke_temp_credentials(ws)
        root = Path(ws.root_dir)
        # Secure cleanup of temp data first.
        tmp = root / ws.dirs.get("tmp", "tmp")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(root, ignore_errors=True)
        with self._lock:
            self._workspaces.pop(workspace_id, None)
            self._sessions.pop(ws.session_id, None)
            self._guards.pop(workspace_id, None)
            self._file_managers.pop(workspace_id, None)
            self._emergency.pop(workspace_id, None)
        self._save_registry()
        self._emit(ws, "workspace.destroyed")

    def complete(self, workspace_id: str) -> None:
        ws = self.get(workspace_id)
        with self._lock:
            self._transition(ws, WorkspaceStatus.COMPLETED)
        self._release_lock(ws)
        self._save_registry()
        self._emit(ws, "task.completed")
        self._maybe_cleanup(ws)

    def fail(
        self, workspace_id: str, *, reason: str = ""
    ) -> None:
        ws = self.get(workspace_id)
        with self._lock:
            if ws.status != WorkspaceStatus.FAILED:
                try:
                    self._transition(ws, WorkspaceStatus.FAILED)
                except WorkspaceError:
                    ws.status = WorkspaceStatus.FAILED
        self._release_lock(ws)
        self._save_registry()
        self._emit(ws, "task.failed", reason=reason)
        if not ws.config.retain_on_failure:
            self._maybe_cleanup(ws)

    # -- snapshots / checkpoints --------------------------------------

    def take_snapshot(
        self,
        workspace_id: str,
        *,
        agent_state: Any | None = None,
        plan: Any | None = None,
        goal: str = "",
        current_task: str = "",
        completed_steps: list[str] | None = None,
        pending_steps: list[str] | None = None,
        browser_state_ref: str = "",
        artifact_refs: list[str] | None = None,
        connector_refs: list[str] | None = None,
        recovery_info: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot:
        """Create an integrity-protected snapshot.

        Secrets are never stored: any credential-shaped value
        is redacted before persistence.
        """
        ws = self.get(workspace_id)
        fm = self.file_manager(workspace_id)
        try:
            files_metadata = fm.metadata()
        except Exception:
            files_metadata = {}
        # Redact before anything is persisted.
        completed = list(
            redact_value(list(completed_steps or []))
        )
        pending = list(redact_value(list(pending_steps or [])))
        artifacts = list(redact_value(list(artifact_refs or [])))
        connectors = list(
            redact_value(list(connector_refs or []))
        )
        recovery = dict(redact_value(recovery_info or {}))
        agent_ref = ""
        if agent_state is not None:
            to_dict = getattr(agent_state, "to_dict", None)
            raw = to_dict() if callable(to_dict) else {}
            agent_ref = _snapshot_checksum(
                redact_value(raw)
            )
        snapshot = WorkspaceSnapshot(
            workspace_id=workspace_id,
            task_id=ws.task_id,
            agent_state_ref=agent_ref,
            current_goal=goal,
            current_task=current_task,
            completed_steps=completed,
            pending_steps=pending,
            browser_state_ref=browser_state_ref,
            files_metadata=files_metadata,
            artifact_refs=artifacts,
            connector_refs=connectors,
            recovery_info=recovery,
        )
        snapshot.checksum = _snapshot_checksum(
            snapshot.to_dict()
        )
        dest = (
            Path(ws.checkpoint_dir)
            / f"{snapshot.snapshot_id}.json"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(snapshot.to_dict(), indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, dest)
        ws.checkpoint_id = snapshot.snapshot_id
        ws.updated_at = time.time()
        session = self._sessions.get(ws.session_id)
        if session is not None:
            session.checkpoints_taken += 1
        self._save_registry()
        self._emit(
            ws, "checkpoint.created",
            snapshot_id=snapshot.snapshot_id,
            completed_steps=len(snapshot.completed_steps),
        )
        return snapshot

    def latest_snapshot(
        self, workspace_id: str
    ) -> WorkspaceSnapshot | None:
        ws = self.get(workspace_id)
        cdir = Path(ws.checkpoint_dir)
        if not cdir.exists():
            return None
        files = sorted(
            cdir.glob("snap-*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in files:
            try:
                data = json.loads(
                    path.read_text(encoding="utf-8")
                )
                snap = WorkspaceSnapshot(
                    snapshot_id=data.get("snapshot_id", ""),
                    workspace_id=data.get("workspace_id", ""),
                    task_id=data.get("task_id", ""),
                    created_at=data.get("created_at", 0.0),
                    agent_state_ref=data.get(
                        "agent_state_ref", ""),
                    current_goal=data.get("current_goal", ""),
                    current_task=data.get("current_task", ""),
                    completed_steps=data.get(
                        "completed_steps", []),
                    pending_steps=data.get("pending_steps", []),
                    browser_state_ref=data.get(
                        "browser_state_ref", ""),
                    files_metadata=data.get(
                        "files_metadata", {}),
                    artifact_refs=data.get("artifact_refs", []),
                    connector_refs=data.get(
                        "connector_refs", []),
                    recovery_info=data.get("recovery_info", {}),
                    checksum=data.get("checksum", ""),
                )
                if self._verify_snapshot(snap):
                    return snap
            except Exception:
                continue
        return None

    def _verify_snapshot(self, snap: WorkspaceSnapshot) -> bool:
        if not snap.checksum:
            return False
        data = snap.to_dict()
        # Creation hashed with checksum="" — reproduce exactly.
        data["checksum"] = ""
        return _snapshot_checksum(data) == snap.checksum

    # -- emergency stop -------------------------------------------------

    def emergency_stop(
        self, workspace_id: str | None = None, *,
        reason: str = "",
    ) -> None:
        """Global (workspace_id=None) or per-workspace stop.

        Never auto-resumes: an explicit authorized resume is
        required afterwards.
        """
        with self._lock:
            if workspace_id is None:
                self._emergency["global"] = True
                targets = list(self._workspaces.values())
            else:
                self._emergency[workspace_id] = True
                targets = [self.get(workspace_id)]
            for ws in targets:
                if ws.status in (
                    WorkspaceStatus.RUNNING,
                    WorkspaceStatus.PAUSED,
                    WorkspaceStatus.WAITING_FOR_APPROVAL,
                    WorkspaceStatus.RECOVERING,
                ):
                    try:
                        self._transition(
                            ws, WorkspaceStatus.STOPPING
                        )
                        self._transition(
                            ws, WorkspaceStatus.STOPPED
                        )
                    except WorkspaceError:
                        ws.status = WorkspaceStatus.STOPPED
                self._release_lock(ws)
                self._emit(
                    ws, "emergency.stop",
                    reason=reason,
                    scope=(
                        "global"
                        if workspace_id is None
                        else "workspace"
                    ),
                )
        self._save_registry()

    def clear_emergency(
        self, workspace_id: str | None = None, *,
        authorized_by: str = "",
    ) -> None:
        """Explicit authorized reset of an emergency stop."""
        if not authorized_by:
            raise WorkspaceError(
                "emergency reset requires explicit authorization"
            )
        with self._lock:
            if workspace_id is None:
                self._emergency.pop("global", None)
            else:
                self._emergency.pop(workspace_id, None)
        if workspace_id is not None:
            ws = self.get(workspace_id)
            self._emit(
                ws, "emergency.cleared",
                authorized_by=authorized_by,
            )

    def is_stopped(self, workspace_id: str = "") -> bool:
        with self._lock:
            if self._emergency.get("global"):
                return True
            if workspace_id:
                return bool(
                    self._emergency.get(workspace_id)
                )
        return False

    def _assert_not_stopped(self, ws: Workspace) -> None:
        if self.is_stopped(ws.workspace_id):
            raise WorkspaceError(
                "workspace is under emergency stop; "
                "explicit authorized reset required"
            )

    # -- accessors ------------------------------------------------------

    def file_manager(
        self, workspace_id: str
    ) -> WorkspaceFileManager:
        ws = self.get(workspace_id)
        with self._lock:
            fm = self._file_managers.get(workspace_id)
            if fm is None:
                fm = WorkspaceFileManager(
                    ws.root_dir,
                    sensitive_paths=(
                        ws.config.policy.sensitive_paths
                    ),
                )
                self._file_managers[workspace_id] = fm
        return fm

    def network_guard(
        self, workspace_id: str
    ) -> WorkspaceNetworkGuard:
        with self._lock:
            guard = self._guards.get(workspace_id)
        if guard is None:
            raise WorkspaceError(
                f"unknown workspace: {workspace_id}"
            )
        return guard

    def session(
        self, workspace_id: str
    ) -> WorkspaceSession | None:
        ws = self.get(workspace_id)
        with self._lock:
            return self._sessions.get(ws.session_id)

    # -- cross-workspace handoff -----------------------------------------

    def handoff(
        self,
        from_workspace: str,
        to_workspace: str,
        payload: dict[str, Any],
        *,
        authorized_by: str = "",
    ) -> dict[str, Any]:
        """Explicit controlled cross-workspace handoff.

        Allowed only when both workspaces opt in and the
        handoff is explicitly authorized.
        """
        src = self.get(from_workspace)
        dst = self.get(to_workspace)
        if not (
            src.config.policy.allow_cross_workspace_handoff
            and dst.config.policy.allow_cross_workspace_handoff
        ):
            raise WorkspaceError(
                "cross-workspace handoff not enabled by policy"
            )
        if not authorized_by:
            raise WorkspaceError(
                "handoff requires explicit authorization"
            )
        clean = redact_value(payload)
        self._emit(
            src, "workspace.handoff_sent",
            to=to_workspace, authorized_by=authorized_by,
        )
        self._emit(
            dst, "workspace.handoff_received",
            sender=from_workspace, authorized_by=authorized_by,
        )
        return {"ok": True, "payload": clean}

    # -- resource-limit helpers --------------------------------------------

    def check_limits(
        self, workspace_id: str, *, usage: dict[str, float]
    ) -> list[str]:
        """Return names of exceeded limits (empty = within limits)."""
        ws = self.get(workspace_id)
        limits = ws.config.resource_limits
        exceeded: list[str] = []
        checks = {
            "max_execution_time_s": usage.get(
                "execution_time_s", 0.0),
            "max_task_duration_s": usage.get(
                "task_duration_s", 0.0),
            "max_cpu_percent": usage.get("cpu_percent", 0.0),
            "max_memory_mb": usage.get("memory_mb", 0.0),
            "max_disk_mb": usage.get("disk_mb", 0.0),
            "max_download_mb": usage.get("download_mb", 0.0),
            "max_network_requests": usage.get(
                "network_requests", 0.0),
            "max_subprocesses": usage.get("subprocesses", 0.0),
            "max_browser_tabs": usage.get("browser_tabs", 0.0),
            "max_retries": usage.get("retries", 0.0),
            "max_loop_iterations": usage.get(
                "loop_iterations", 0.0),
        }
        for name, value in checks.items():
            limit = getattr(limits, name)
            if value > limit:
                exceeded.append(name)
        if exceeded:
            self._emit(
                ws, "resource.limit_reached",
                exceeded=exceeded, usage=usage,
            )
        return exceeded

    # -- health --------------------------------------------------------------

    def health_check(
        self, workspace_id: str, **kwargs: Any
    ):
        ws = self.get(workspace_id)
        disk_mb = self.adapter.disk_usage_mb(
            Path(ws.root_dir)
        )
        return self.health.check(
            ws, disk_usage_mb=disk_mb, **kwargs
        )

    # -- internal helpers ------------------------------------------------------

    def _release_lock(self, ws: Workspace) -> None:
        try:
            (Path(ws.root_dir) / _LOCK_FILE).unlink(
                missing_ok=True
            )
        except OSError:
            pass

    def _cleanup_stale_locks(self, ws: Workspace) -> None:
        lock = Path(ws.root_dir) / _LOCK_FILE
        if not lock.exists():
            return
        try:
            data = json.loads(lock.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        pid = data.get("pid")
        stale = True
        if pid:
            try:
                os.kill(int(pid), 0)
                stale = False
            except (OSError, ValueError):
                stale = True
        if stale:
            lock.unlink(missing_ok=True)
            self._emit(ws, "lock.stale_cleaned", pid=pid)

    def _revoke_temp_credentials(self, ws: Workspace) -> None:
        # Workspace never stores raw credentials; it only drops
        # references so nothing lingers after destroy.
        refs = ws.metadata.pop("credential_refs", [])
        self._emit(
            ws, "credentials.revoked",
            revoked_refs=len(refs) if isinstance(refs, list) else 0,
        )

    def _maybe_cleanup(self, ws: Workspace) -> None:
        mode = ws.config.cleanup_mode
        if mode == CleanupMode.EPHEMERAL:
            self._cleanup_workspace_files(ws, secure=True)
        elif mode == CleanupMode.UNTIL_TASK_COMPLETE and (
            ws.status == WorkspaceStatus.COMPLETED
        ):
            self._cleanup_workspace_files(ws, secure=True)

    def _cleanup_workspace_files(
        self, ws: Workspace, *, secure: bool
    ) -> None:
        root = Path(ws.root_dir)
        # Preserve artifacts/exports before cleanup.
        preserved: list[str] = []
        artifacts = root / ws.dirs.get("artifacts", "artifacts")
        if artifacts.exists():
            preserved.append(str(artifacts))
        for sub in ("tmp", "logs", "browser"):
            target = root / ws.dirs.get(sub, sub)
            if target.exists() and str(target) not in preserved:
                shutil.rmtree(target, ignore_errors=True)
        if secure:
            # Overwrite-free best-effort: temp data already removed.
            pass
        self._emit(
            ws, "workspace.cleaned",
            preserved=preserved, secure=secure,
        )

    # -- metrics ---------------------------------------------------------------

    def metrics(self, workspace_id: str) -> dict[str, Any]:
        ws = self.get(workspace_id)
        session = self._sessions.get(ws.session_id)
        guard = self._guards.get(workspace_id)
        disk_mb = self.adapter.disk_usage_mb(Path(ws.root_dir))
        return {
            "workspace_id": workspace_id,
            "status": ws.status.value,
            "duration_s": round(
                time.time() - ws.created_at, 1),
            "disk_mb": round(disk_mb, 2),
            "session": (
                session.to_dict() if session else None
            ),
            "network": (
                guard.stats() if guard else None
            ),
            "limits": ws.config.resource_limits.to_dict(),
        }

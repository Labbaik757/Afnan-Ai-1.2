"""Remote command routing.

Every remote command flows through one pipeline:

    validate -> idempotency -> authorize -> rate-limit
        -> dispatch to existing runtime API
        -> verify -> audit -> result

The router never executes privileged work itself and never bypasses
the existing runtime: TaskManager, GoalManager, BrowserController,
ComputerRuntime, ArtifactManager, ApprovalCenter and SecurityCenter
stay authoritative.  Remote commands that would bypass them are
rejected.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from afnan_ai.control import views
from afnan_ai.control.idempotency import IdempotencyStore
from afnan_ai.control.models import (
    CommandStatus,
    CommandType,
    CommandValidationError,
    QueuePolicy,
    RemoteCommand,
    RemoteCommandResult,
    queue_policy_for,
)
from afnan_ai.security.models import Actor, ActorKind


class CommandRouterError(Exception):
    """Routing failed (denied, expired, malformed, ...)."""

    def __init__(
        self, message: str, *, code: str = "ROUTING_FAILED"
    ) -> None:
        super().__init__(message)
        self.code = code


def _remote_actor(
    device_id: str, session_id: str, capabilities: tuple[str, ...]
) -> Actor:
    return Actor(
        kind=ActorKind.USER,
        actor_id=f"remote:{device_id}:{session_id}",
        capabilities=tuple(capabilities),
    )


class RemoteCommandRouter:
    """Validates, authorizes and dispatches remote commands."""

    def __init__(
        self,
        *,
        security_center,
        task_manager=None,
        goal_manager=None,
        scheduler=None,
        activity_center=None,
        approval_center=None,
        artifact_manager=None,
        browser_controller=None,
        computer_runtime=None,
        device_registry=None,
        session_manager=None,
        auth_provider=None,
        idempotency: IdempotencyStore | None = None,
        audit_adapter=None,
        metrics=None,
        command_ttl_s: float = 120.0,
        max_payload_bytes: int = 65536,
        on_command: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.security = security_center
        self.tasks = task_manager
        self.goals = goal_manager
        self.scheduler = scheduler
        self.activity = activity_center
        self.approvals = approval_center
        self.artifacts = artifact_manager
        self.browser = browser_controller
        self.computer = computer_runtime
        self.devices = device_registry
        self.sessions = session_manager
        self.auth = auth_provider
        self.idempotency = idempotency or IdempotencyStore()
        self.audit = audit_adapter
        self.metrics = metrics
        self._command_ttl_s = command_ttl_s
        self._max_payload_bytes = max_payload_bytes
        self._on_command = on_command
        self._lock = threading.RLock()
        self._handlers: dict[
            CommandType, tuple[str, Callable]
        ] = {}
        self._register_handlers()

    # -- handler table ----------------------------------------------------

    def _on(
        self,
        command_type: CommandType,
        capability: str,
        handler: Callable,
    ) -> None:
        self._handlers[command_type] = (capability, handler)

    def _register_handlers(self) -> None:
        on = self._on
        # Monitoring.
        on(CommandType.GET_AGENT_STATUS, "remote.status.read",
           self._h_agent_status)
        on(CommandType.LIST_TASKS, "remote.tasks.read",
           self._h_list_tasks)
        on(CommandType.GET_TASK, "remote.tasks.read",
           self._h_get_task)
        on(CommandType.LIST_GOALS, "remote.goals.read",
           self._h_list_goals)
        on(CommandType.GET_GOAL, "remote.goals.read",
           self._h_get_goal)
        on(CommandType.LIST_ARTIFACTS, "remote.artifacts.read",
           self._h_list_artifacts)
        on(CommandType.GET_ARTIFACT, "remote.artifacts.read",
           self._h_get_artifact)
        on(CommandType.LIST_APPROVALS, "remote.status.read",
           self._h_list_approvals)
        on(CommandType.GET_APPROVAL, "remote.status.read",
           self._h_get_approval)
        on(CommandType.LIST_DEVICES, "remote.devices.read",
           self._h_list_devices)
        on(CommandType.GET_DEVICE, "remote.devices.read",
           self._h_get_device)
        on(CommandType.LIST_SESSIONS, "remote.devices.read",
           self._h_list_sessions)
        on(CommandType.QUERY_ACTIVITY, "remote.activity.read",
           self._h_query_activity)
        on(CommandType.GET_METRICS, "remote.metrics.read",
           self._h_get_metrics)
        on(CommandType.HEALTH_CHECK, "remote.status.read",
           self._h_health)
        on(CommandType.LIST_SCHEDULES, "remote.schedule.manage",
           self._h_list_schedules)
        # Task control.
        on(CommandType.START_TASK, "remote.tasks.control",
           self._h_start_task)
        on(CommandType.PAUSE_TASK, "remote.tasks.control",
           self._h_pause_task)
        on(CommandType.RESUME_TASK, "remote.tasks.control",
           self._h_resume_task)
        on(CommandType.CANCEL_TASK, "remote.tasks.control",
           self._h_cancel_task)
        on(CommandType.RETRY_TASK, "remote.tasks.control",
           self._h_retry_task)
        # Goal control.
        on(CommandType.PAUSE_GOAL, "remote.goals.control",
           self._h_pause_goal)
        on(CommandType.RESUME_GOAL, "remote.goals.control",
           self._h_resume_goal)
        on(CommandType.CANCEL_GOAL, "remote.goals.control",
           self._h_cancel_goal)
        # Approvals.
        on(CommandType.APPROVE_ACTION, "remote.approvals.decide",
           self._h_approve)
        on(CommandType.DENY_ACTION, "remote.approvals.decide",
           self._h_deny)
        # Browser / computer (existing runtimes only).
        on(CommandType.BROWSER_STATE, "remote.browser.read",
           self._h_browser_state)
        on(CommandType.BROWSER_NAVIGATE, "remote.browser.control",
           self._h_browser_navigate)
        on(CommandType.BROWSER_SCREENSHOT, "remote.browser.read",
           self._h_browser_screenshot)
        on(CommandType.COMPUTER_OBSERVE, "remote.computer.observe",
           self._h_computer_observe)
        on(CommandType.COMPUTER_ACT, "remote.computer.control",
           self._h_computer_act)
        # Scheduling.
        on(CommandType.MANAGE_SCHEDULE, "remote.schedule.manage",
           self._h_manage_schedule)
        # Safety.
        on(CommandType.EMERGENCY_STOP,
           "remote.safety.emergency_stop", self._h_emergency_stop)
        # Session / device management.
        on(CommandType.HEARTBEAT, "remote.status.read",
           self._h_heartbeat)
        on(CommandType.REVOKE_DEVICE, "remote.devices.manage",
           self._h_revoke_device)
        on(CommandType.REVOKE_SESSION, "remote.devices.manage",
           self._h_revoke_session)
        on(CommandType.ROTATE_CREDENTIALS, "remote.devices.manage",
           self._h_rotate_credentials)

    # -- main entry ---------------------------------------------------------

    def route(
        self,
        command: RemoteCommand,
        *,
        session_capabilities: tuple[str, ...] = (),
        live_session: bool = True,
    ) -> RemoteCommandResult:
        """Run the full pipeline for one remote command."""
        started = time.time()
        try:
            return self._route_inner(
                command,
                session_capabilities=session_capabilities,
                live_session=live_session,
            )
        finally:
            if self.metrics is not None:
                try:
                    self.metrics.record_command(
                        command.command_type.value,
                        time.time() - started,
                    )
                except Exception:
                    pass

    def _route_inner(
        self,
        command: RemoteCommand,
        *,
        session_capabilities: tuple[str, ...],
        live_session: bool,
    ) -> RemoteCommandResult:
        # 1. Schema validation.
        self._validate(command)
        # 2. Expiry.
        if command.is_expired():
            return self._denied(
                command, "command expired", "EXPIRED",
                CommandStatus.EXPIRED,
            )
        # 3. Offline queue policy.
        policy = queue_policy_for(command.command_type)
        if not live_session and policy in (
            QueuePolicy.REQUIRES_LIVE_SESSION,
            QueuePolicy.NEVER_QUEUEABLE,
        ):
            return self._denied(
                command,
                "command requires a live session",
                "REQUIRES_LIVE_SESSION",
                CommandStatus.DENIED,
            )
        # 4. Idempotency: replay stored result for duplicates.
        if command.idempotency_key:
            existing = self.idempotency.check(
                command.device_id, command.idempotency_key
            )
            if existing is not None and not existing.in_flight:
                result = RemoteCommandResult(
                    command_id=command.command_id,
                    status=CommandStatus.DUPLICATE,
                    result=dict(existing.result),
                    correlation_id=command.correlation_id,
                )
                if self.metrics is not None:
                    self.metrics.record_duplicate_command()
                self._audit(command, result, "duplicate")
                return result
            if not self.idempotency.claim(
                command.device_id,
                command.idempotency_key,
                command_id=command.command_id,
                command_type=command.command_type.value,
                session_id=command.session_id,
            ):
                return self._denied(
                    command,
                    "duplicate command (idempotency key in use)",
                    "DUPLICATE",
                    CommandStatus.DUPLICATE,
                )
        # 5. Authorization via the existing security layer.
        handler = self._handlers.get(command.command_type)
        if handler is None:
            self.idempotency.release(
                command.device_id, command.idempotency_key
            )
            return self._denied(
                command, "unknown command type", "UNKNOWN_COMMAND",
                CommandStatus.FAILED,
            )
        capability, fn = handler
        actor = _remote_actor(
            command.device_id,
            command.session_id,
            session_capabilities,
        )
        allowed = self.security.permissions.check(
            actor, capability
        )
        # Session capabilities are the grant boundary: the actor
        # carries exactly what the session was granted.
        if not allowed:
            self.idempotency.release(
                command.device_id, command.idempotency_key
            )
            if self.metrics is not None:
                try:
                    self.metrics.record_authz_denial()
                except Exception:
                    pass
            return self._denied(
                command,
                f"missing capability: {capability}",
                "FORBIDDEN",
                CommandStatus.DENIED,
            )
        # 6. Rate limit (existing limiter, per-device key).
        rl_key = (
            f"control:{command.device_id}:"
            f"{command.command_type.value}"
        )
        verdict = self.security.rate_limiter.check(rl_key)
        if isinstance(verdict, dict) and not verdict.get(
            "ok", True
        ):
            self.idempotency.release(
                command.device_id, command.idempotency_key
            )
            return self._denied(
                command, "rate limit exceeded", "RATE_LIMITED",
                CommandStatus.DENIED,
            )
        self.security.rate_limiter.record_call(rl_key)
        # 7. Dispatch to the existing runtime API.
        try:
            payload = fn(command, actor)
            status = CommandStatus.COMPLETED
            error, code = "", ""
        except CommandRouterError as e:
            payload, status, error, code = (
                {},
                CommandStatus.FAILED,
                str(e),
                e.code,
            )
        except Exception as e:  # never leak internals
            payload, status, error, code = (
                {},
                CommandStatus.FAILED,
                "command execution failed",
                "EXECUTION_FAILED",
            )
        result = RemoteCommandResult(
            command_id=command.command_id,
            status=status,
            result=payload,
            error=error,
            error_code=code,
            verification={"verified": status
                          == CommandStatus.COMPLETED},
            correlation_id=command.correlation_id,
        )
        # 8. Record idempotency outcome + audit.
        if command.idempotency_key:
            if status == CommandStatus.COMPLETED:
                self.idempotency.complete(
                    command.device_id,
                    command.idempotency_key,
                    result.to_dict(),
                )
            else:
                self.idempotency.release(
                    command.device_id, command.idempotency_key
                )
        self._audit(command, result, capability)
        if self._on_command is not None:
            try:
                self._on_command(
                    {
                        "command_id": command.command_id,
                        "command_type": (
                            command.command_type.value
                        ),
                        "device_id": command.device_id,
                        "status": result.status.value,
                    }
                )
            except Exception:
                pass
        return result

    # -- validation ---------------------------------------------------------

    def _validate(self, command: RemoteCommand) -> None:
        if not command.session_id or not command.device_id:
            raise CommandValidationError(
                "session_id and device_id are required"
            )
        if len(command.session_id) > 128 or len(
            command.device_id
        ) > 128:
            raise CommandValidationError("id too long")
        if len(str(command.payload)) > self._max_payload_bytes:
            raise CommandValidationError("payload too large")
        age = time.time() - command.created_at
        if age > self._command_ttl_s and not command.expires_at:
            raise CommandValidationError("command too old")
        if command.protocol_version != "v1":
            raise CommandValidationError(
                "unsupported protocol version"
            )

    def _denied(
        self,
        command: RemoteCommand,
        message: str,
        code: str,
        status: CommandStatus,
    ) -> RemoteCommandResult:
        result = RemoteCommandResult(
            command_id=command.command_id,
            status=status,
            error=message,
            error_code=code,
            correlation_id=command.correlation_id,
        )
        self._audit(command, result, "")
        return result

    def _audit(
        self,
        command: RemoteCommand,
        result: RemoteCommandResult,
        capability: str,
    ) -> None:
        if self.audit is None:
            return
        try:
            self.audit.command(
                command=command,
                result=result,
                capability=capability,
            )
        except Exception:
            pass

    # -- helpers --------------------------------------------------------------

    def _require(self, service, name: str):
        if service is None:
            raise CommandRouterError(
                f"{name} is not wired into this control plane",
                code="NOT_WIRED",
            )
        return service

    def _payload_str(
        self, command: RemoteCommand, key: str,
        *, required: bool = True, max_len: int = 500
    ) -> str:
        value = command.payload.get(key, "")
        if required and not value:
            raise CommandRouterError(
                f"payload.{key} is required", code="BAD_PAYLOAD"
            )
        text = str(value)
        if len(text) > max_len:
            raise CommandRouterError(
                f"payload.{key} too long", code="BAD_PAYLOAD"
            )
        return text

    # -- monitoring handlers ----------------------------------------------------

    def _h_agent_status(self, command, actor):
        devices = self._require(self.devices, "DeviceRegistry")
        sessions = self._require(self.sessions, "SessionManager")
        security = self.security
        # Current task: prefer the background runner / task manager.
        current_task = None
        waiting = False
        pending_approvals = 0
        if self.tasks is not None:
            try:
                running = self.tasks.list(status="running")
                if running:
                    current_task = views.task_view(running[0])
                waiting = bool(
                    self.tasks.list(
                        status="waiting_for_approval"
                    )
                )
            except Exception:
                pass
        current_goal = None
        if self.goals is not None:
            try:
                active = self.goals.active_goals()
                if active:
                    current_goal = views.goal_view(active[0])
            except Exception:
                pass
        if self.approvals is not None:
            try:
                pending_approvals = len(self.approvals.pending())
            except Exception:
                pass
        return views.agent_view(
            status="running",
            current_task=current_task,
            current_goal=current_goal,
            waiting_for_approval=waiting,
            pending_approvals=pending_approvals,
            connected_devices=len(devices.list()),
            active_sessions=len(
                sessions.list(live_only=True)
            ),
            emergency_tripped=bool(
                security.emergency.is_tripped()
            ),
            runtime_version="1.2",
        )

    def _h_list_tasks(self, command, actor):
        tasks = self._require(self.tasks, "TaskManager")
        status = command.payload.get("status") or None
        return {
            "tasks": [views.task_view(t) for t in tasks.list(
                status=status
            )][:200]
        }

    def _h_get_task(self, command, actor):
        tasks = self._require(self.tasks, "TaskManager")
        task_id = self._payload_str(command, "task_id")
        task = tasks.get(task_id)
        if task is None:
            raise CommandRouterError(
                "task not found", code="NOT_FOUND"
            )
        return {"task": views.task_view(task)}

    def _h_list_goals(self, command, actor):
        goals = self._require(self.goals, "GoalManager")
        status = command.payload.get("status") or None
        return {
            "goals": [views.goal_view(g) for g in goals.list(
                status=status
            )][:200]
        }

    def _h_get_goal(self, command, actor):
        goals = self._require(self.goals, "GoalManager")
        goal_id = self._payload_str(command, "goal_id")
        goal = goals.get(goal_id)
        if goal is None:
            raise CommandRouterError(
                "goal not found", code="NOT_FOUND"
            )
        return {"goal": views.goal_view(goal)}

    def _h_list_artifacts(self, command, actor):
        artifacts = self._require(
            self.artifacts, "ArtifactManager"
        )
        project_id = str(
            command.payload.get("project_id", "default")
        )[:64]
        items = artifacts.list_artifacts(project_id=project_id)
        # Never expose sensitive artifacts without the explicit
        # capability; the read capability alone is not enough.
        sensitive_ok = self.security.permissions.check(
            actor, "remote.admin"
        )
        out = []
        for item in items[:200]:
            if getattr(item, "sensitive", False) and not (
                sensitive_ok
            ):
                continue
            out.append(views.artifact_view(item))
        return {"artifacts": out}

    def _h_get_artifact(self, command, actor):
        artifacts = self._require(
            self.artifacts, "ArtifactManager"
        )
        artifact_id = self._payload_str(command, "artifact_id")
        record = artifacts.get(artifact_id)
        if record is None:
            raise CommandRouterError(
                "artifact not found", code="NOT_FOUND"
            )
        if getattr(record, "sensitive", False):
            if not self.security.permissions.check(
                actor, "remote.admin"
            ):
                raise CommandRouterError(
                    "sensitive artifact requires remote.admin",
                    code="FORBIDDEN",
                )
        return {"artifact": views.artifact_view(record)}

    def _h_list_approvals(self, command, actor):
        approvals = self._require(
            self.approvals, "ApprovalCenter"
        )
        return {
            "approvals": [
                views.approval_view(a)
                for a in approvals.pending()
            ]
        }

    def _h_get_approval(self, command, actor):
        approvals = self._require(
            self.approvals, "ApprovalCenter"
        )
        approval_id = self._payload_str(command, "approval_id")
        tracked = approvals.get(approval_id)
        if tracked is None:
            raise CommandRouterError(
                "approval not found", code="NOT_FOUND"
            )
        return {"approval": views.approval_view(tracked)}

    def _h_list_devices(self, command, actor):
        devices = self._require(self.devices, "DeviceRegistry")
        return {
            "devices": [
                views.device_view(d) for d in devices.list()
            ]
        }

    def _h_get_device(self, command, actor):
        devices = self._require(self.devices, "DeviceRegistry")
        device_id = self._payload_str(command, "device_id")
        info = devices.get(device_id)
        if info is None:
            raise CommandRouterError(
                "device not found", code="NOT_FOUND"
            )
        return {"device": views.device_view(info)}

    def _h_list_sessions(self, command, actor):
        sessions = self._require(
            self.sessions, "SessionManager"
        )
        return {
            "sessions": [
                views.session_view(s)
                for s in sessions.list()
            ]
        }

    def _h_query_activity(self, command, actor):
        activity = self._require(
            self.activity, "ActivityCenter"
        )
        after_seq = int(command.payload.get("after_seq", 0) or 0)
        limit = min(
            int(command.payload.get("limit", 100) or 100), 500
        )
        task_id = str(command.payload.get("task_id", ""))[:64]
        sync_state = activity.sync(
            after_seq=after_seq, task_id=task_id, limit=limit
        )
        return {
            "last_seq": sync_state["last_seq"],
            "events": sync_state["missed_events"],
        }

    def _h_get_metrics(self, command, actor):
        metrics = self._require(self.metrics, "Metrics")
        return {"metrics": metrics.snapshot()}

    def _h_health(self, command, actor):
        return {
            "ok": True,
            "protocol_version": "v1",
            "emergency_tripped": bool(
                self.security.emergency.is_tripped()
            ),
        }

    def _h_list_schedules(self, command, actor):
        scheduler = self._require(
            self.scheduler, "TaskScheduler"
        )
        status = command.payload.get("status") or None
        items = scheduler.list(status=status)
        return {
            "schedules": [
                {
                    "schedule_id": s.schedule_id,
                    "goal_text": s.goal_text[:300],
                    "recurrence": s.recurrence,
                    "next_run_at": s.next_run_at,
                    "status": s.status,
                }
                for s in items[:200]
            ]
        }

    # -- task control handlers ----------------------------------------------------

    def _h_start_task(self, command, actor):
        tasks = self._require(self.tasks, "TaskManager")
        goal_text = self._payload_str(
            command, "goal_text", max_len=2000
        )
        task = tasks.enqueue(
            goal_text,
            goal_id=str(
                command.payload.get("goal_id", "")
            )[:64],
            priority=int(command.payload.get("priority", 3) or 3),
            metadata={
                "remote": True,
                "device_id": command.device_id,
                "session_id": command.session_id,
            },
        )
        return {"task": views.task_view(task)}

    def _h_pause_task(self, command, actor):
        tasks = self._require(self.tasks, "TaskManager")
        task_id = self._payload_str(command, "task_id")
        return {"task": views.task_view(tasks.pause(task_id))}

    def _h_resume_task(self, command, actor):
        tasks = self._require(self.tasks, "TaskManager")
        task_id = self._payload_str(command, "task_id")
        return {"task": views.task_view(tasks.resume(task_id))}

    def _h_cancel_task(self, command, actor):
        tasks = self._require(self.tasks, "TaskManager")
        task_id = self._payload_str(command, "task_id")
        return {"task": views.task_view(tasks.cancel(task_id))}

    def _h_retry_task(self, command, actor):
        tasks = self._require(self.tasks, "TaskManager")
        task_id = self._payload_str(command, "task_id")
        task = tasks.get(task_id)
        if task is None:
            raise CommandRouterError(
                "task not found", code="NOT_FOUND"
            )
        # Retry = resume a failed task; TaskManager validates the
        # transition, so illegal retries fail safely.
        return {"task": views.task_view(tasks.resume(task_id))}

    # -- goal control handlers ------------------------------------------------------

    def _h_pause_goal(self, command, actor):
        goals = self._require(self.goals, "GoalManager")
        goal_id = self._payload_str(command, "goal_id")
        return {"goal": views.goal_view(goals.pause(goal_id))}

    def _h_resume_goal(self, command, actor):
        goals = self._require(self.goals, "GoalManager")
        goal_id = self._payload_str(command, "goal_id")
        return {"goal": views.goal_view(goals.resume(goal_id))}

    def _h_cancel_goal(self, command, actor):
        goals = self._require(self.goals, "GoalManager")
        goal_id = self._payload_str(command, "goal_id")
        return {"goal": views.goal_view(goals.cancel(goal_id))}

    # -- approval handlers ------------------------------------------------------------

    def _h_approve(self, command, actor):
        return self._decide_approval(command, actor, True)

    def _h_deny(self, command, actor):
        return self._decide_approval(command, actor, False)

    def _decide_approval(
        self, command, actor, granted: bool
    ):
        approvals = self._require(
            self.approvals, "ApprovalCenter"
        )
        approval_id = self._payload_str(command, "approval_id")
        tracked = approvals.get(approval_id)
        if tracked is None:
            raise CommandRouterError(
                "approval not found", code="NOT_FOUND"
            )
        # The approval must still be pending and unexpired; the
        # ApprovalCenter enforces single-decision semantics.
        outcome = approvals.decide(
            approval_id,
            granted=granted,
            actor=f"remote:{command.device_id}",
        )
        return {
            "approval_id": approval_id,
            "granted": granted,
            "outcome": outcome,
        }

    # -- browser handlers (existing runtime only) ----------------------------------------

    def _h_browser_state(self, command, actor):
        browser = self._require(
            self.browser, "BrowserController"
        )
        try:
            page = browser.current_page()
        except Exception:
            page = {}
        return {
            "state": {
                "url": str(page.get("url", ""))[:500],
                "title": str(page.get("title", ""))[:200],
            }
        }

    def _h_browser_navigate(self, command, actor):
        browser = self._require(
            self.browser, "BrowserController"
        )
        url = self._payload_str(
            command, "url", max_len=2000
        )
        # URL policy: reuse the existing resource policy so remote
        # navigation cannot bypass browser domain controls.
        try:
            self.security.resource_policy.check_browser_url(url)
        except Exception as e:
            raise CommandRouterError(
                f"url blocked by resource policy: {e}",
                code="URL_BLOCKED",
            )
        result = browser.navigate(url)
        return {
            "url": str(result.get("url", url))[:500],
            "title": str(result.get("title", ""))[:200],
        }

    def _h_browser_screenshot(self, command, actor):
        browser = self._require(
            self.browser, "BrowserController"
        )
        result = browser.screenshot()
        # Screenshots stay server-side; the client gets a
        # reference it can fetch through the artifact channel.
        response = {
            "path": str(result.get("path", "")),
            "size_bytes": result.get("size_bytes", 0),
            "captured_at": result.get("captured_at", ""),
        }
        # Optional inline delivery for thin clients (e.g. the Android
        # app) that cannot fetch the server-side path directly.
        # Backward compatible: omitted unless explicitly requested.
        if command.payload.get("inline"):
            import base64
            import os
            path = response["path"]
            if path and os.path.isfile(path):
                with open(path, "rb") as f:
                    raw = f.read(5 * 1024 * 1024)
                response["data_base64"] = base64.b64encode(raw).decode(
                    "ascii"
                )
        return response

    # -- computer handlers (existing runtime only) ------------------------------------------

    def _h_computer_observe(self, command, actor):
        computer = self._require(
            self.computer, "ComputerRuntime"
        )
        observe = getattr(computer, "observe", None)
        if observe is None:
            raise CommandRouterError(
                "computer runtime has no observe()",
                code="NOT_WIRED",
            )
        observation = observe()
        # The runtime already redacts sensitive content; project a
        # minimal shape anyway.
        if isinstance(observation, dict):
            return {
                "observation": {
                    k: str(v)[:500]
                    for k, v in observation.items()
                    if k in ("windows", "active_window",
                             "displays", "summary")
                }
            }
        return {"observation": str(observation)[:1000]}

    def _h_computer_act(self, command, actor):
        computer = self._require(
            self.computer, "ComputerRuntime"
        )
        action = command.payload.get("action")
        if not isinstance(action, dict):
            raise CommandRouterError(
                "payload.action must be an object",
                code="BAD_PAYLOAD",
            )
        # The computer runtime's own pipeline (observe -> resolve ->
        # validate -> permission -> execute -> re-observe -> verify)
        # stays authoritative; remote input is just the action.
        # ComputerRuntime.execute() is preferred; ComputerController
        # exposes the same pipeline through act().
        execute = getattr(computer, "execute", None)
        if execute is None:
            execute = getattr(computer, "act", None)
        if execute is None:
            raise CommandRouterError(
                "computer runtime has no execute()/act()",
                code="NOT_WIRED",
            )
        result = execute(action)
        to_dict = getattr(result, "to_dict", None)
        data = to_dict() if callable(to_dict) else dict(
            result if isinstance(result, dict) else {}
        )
        return {"result": {k: str(v)[:500] for k, v in
                           data.items()}}

    # -- scheduling --------------------------------------------------------------------------

    def _h_manage_schedule(self, command, actor):
        scheduler = self._require(
            self.scheduler, "TaskScheduler"
        )
        op = self._payload_str(command, "op", max_len=32)
        if op == "create":
            goal_text = self._payload_str(
                command, "goal_text", max_len=2000
            )
            scheduled = scheduler.schedule_task(
                goal_text,
                delay_s=float(
                    command.payload.get("delay_s", 60) or 60
                ),
                recurrence=str(
                    command.payload.get("recurrence", "once")
                )[:16],
            )
            return {"schedule_id": scheduled.schedule_id}
        schedule_id = self._payload_str(command, "schedule_id")
        if op == "pause":
            scheduler.pause(schedule_id)
        elif op == "resume":
            scheduler.resume(schedule_id)
        elif op == "cancel":
            scheduler.cancel(schedule_id)
        else:
            raise CommandRouterError(
                f"unknown schedule op: {op}", code="BAD_PAYLOAD"
            )
        return {"schedule_id": schedule_id, "op": op}

    # -- safety ---------------------------------------------------------------------------------

    def _h_emergency_stop(self, command, actor):
        reason = self._payload_str(
            command, "reason", required=False, max_len=500
        ) or "remote emergency stop"
        record = self.security.trip_emergency(
            reason, actor=f"remote:{command.device_id}"
        )
        return {
            "tripped": True,
            "at": getattr(record, "at", ""),
            "reason": reason,
        }

    # -- session / device management ------------------------------------------------------------------

    def _h_heartbeat(self, command, actor):
        sessions = self._require(
            self.sessions, "SessionManager"
        )
        session = sessions.heartbeat(command.session_id)
        return {
            "session_id": session.session_id,
            "state": session.state.value,
            "last_event_seq": session.last_event_seq,
        }

    def _h_revoke_device(self, command, actor):
        devices = self._require(self.devices, "DeviceRegistry")
        sessions = self._require(
            self.sessions, "SessionManager"
        )
        device_id = self._payload_str(command, "device_id")
        if device_id == command.device_id:
            raise CommandRouterError(
                "a device cannot revoke itself; use session revoke",
                code="FORBIDDEN",
            )
        devices.revoke(device_id)
        revoked_sessions = sessions.revoke_device_sessions(
            device_id
        )
        tokens = 0
        if self.auth is not None:
            tokens = self.auth.revoke_device_tokens(device_id)
        return {
            "device_id": device_id,
            "revoked_sessions": revoked_sessions,
            "revoked_tokens": tokens,
        }

    def _h_revoke_session(self, command, actor):
        sessions = self._require(
            self.sessions, "SessionManager"
        )
        session_id = self._payload_str(command, "session_id")
        if session_id == command.session_id:
            raise CommandRouterError(
                "a session cannot revoke itself",
                code="FORBIDDEN",
            )
        sessions.revoke(session_id)
        tokens = 0
        if self.auth is not None:
            tokens = self.auth.revoke_session_tokens(session_id)
        return {
            "session_id": session_id,
            "revoked_tokens": tokens,
        }

    def _h_rotate_credentials(self, command, actor):
        if self.auth is None:
            raise CommandRouterError(
                "auth provider not wired", code="NOT_WIRED"
            )
        sessions = self._require(
            self.sessions, "SessionManager"
        )
        session = sessions.get(command.session_id)
        if session is None:
            raise CommandRouterError(
                "session not found", code="NOT_FOUND"
            )
        old_token = self._payload_str(
            command, "token", max_len=256
        )
        new_token = self.auth.rotate(session, old_token)
        # The new token value is returned once, over the already
        # authenticated channel; it is never logged or stored.
        return {"token": new_token}

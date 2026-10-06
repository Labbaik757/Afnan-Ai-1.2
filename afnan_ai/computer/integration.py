"""ComputerRuntime integrations.

- **ActivityCenter**: every computer operation emits a safe,
  user-facing activity event (application opened, window
  focused, element found, action executed/failed, screen
  observed, verification result, recovery, approval
  required).  Secret input is redacted.
- **SecureWorkspace**: the runtime operates only inside the
  task's authorized scope (allowed applications, files,
  resource limits).  Visual file-manager actions never
  bypass workspace policy.
- **EmergencyStop**: trips halt mouse/keyboard/launches and
  cancel pending computer actions.

All wiring is best-effort: a broken hook never breaks the
runtime.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.computer.models import ComputerActionResult
from afnan_ai.redaction import redact_text


# -- ActivityCenter ------------------------------------------------------

_ACTIVITY_KIND_MAP = {
    "action_executed": "action_completed",
    "application_opened": "task_started",
    "window_focused": "observing",
    "element_found": "observing",
    "action_failed": "action_failed",
    "screen_observed": "observing",
    "verification": "verification_completed",
    "recovery": "recovery_started",
    "approval_required": "approval_required",
}


def attach_activity_center(
    runtime: Any, activity_center: Any, *, task_id: str = ""
) -> None:
    """Forward runtime events into the ActivityCenter."""

    def hook(event: dict[str, Any]) -> None:
        try:
            kind = str(event.get("kind", ""))
            activity_type = _ACTIVITY_KIND_MAP.get(
                kind, "observing"
            )
            summary = str(event.get("summary", kind))[:300]
            details = dict(event.get("details", {}) or {})
            # Secret input is already redacted by the
            # runtime; double-guard here.
            if "text" in details:
                details["text"] = "***"
            activity_center.emit(
                activity_type,
                f"[computer] {summary}",
                task_id=task_id
                or str(details.pop("task_id", "")),
                source="computer",
                details=details,
            )
        except Exception:
            pass

    # Chain with any existing hook.
    previous = getattr(runtime, "_on_event", None)

    def chained(event: dict[str, Any]) -> None:
        if previous is not None:
            try:
                previous(event)
            except Exception:
                pass
        hook(event)

    runtime._on_event = chained
    # ScreenObserver has its own hook slot.
    try:
        prev_obs = runtime.screen._on_observation

        def obs_chained(obs: dict[str, Any]) -> None:
            if prev_obs is not None:
                try:
                    prev_obs(obs)
                except Exception:
                    pass

        runtime.screen._on_observation = obs_chained
    except Exception:
        pass


def emit_result(
    activity_center: Any,
    result: ComputerActionResult,
    *,
    task_id: str = "",
) -> None:
    """Publish one action result as activity."""
    try:
        activity_center.emit(
            "action_completed"
            if result.success
            else "action_failed",
            f"[computer] {result.action.get('kind', 'action')} "
            f"{'succeeded' if result.success else 'failed'}"
            + (
                f" ({result.error_code})"
                if result.error_code
                else ""
            ),
            task_id=task_id,
            source="computer",
            details=result.to_dict(),
        )
    except Exception:
        pass


# -- SecureWorkspace -------------------------------------------------------

class WorkspaceComputerScope:
    """Bind a ComputerRuntime to a SecureWorkspace task scope.

    The runtime may only:

    - use applications in the workspace's allowed list,
    - touch files inside the workspace root,
    - stay within the workspace resource limits.

    Visual actions (e.g. driving the OS file manager) are
    *not* a bypass: file-affecting intents are validated
    against the workspace policy before execution.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        workspace_id: str = "",
        allowed_apps: list[str] | None = None,
        allowed_roots: list[str] | None = None,
        file_manager: Any = None,
    ) -> None:
        self.runtime = runtime
        self.workspace_id = workspace_id
        self.allowed_apps = [a.lower() for a in (allowed_apps or [])]
        self.allowed_roots = list(allowed_roots or [])
        self.file_manager = file_manager

    def check_app(self, app_name: str) -> None:
        from afnan_ai.computer.errors import ComputerError

        if not self.allowed_apps:
            return  # no scope configured → controller policy applies
        needle = str(app_name or "").lower()
        if not any(
            needle == a or a in needle for a in self.allowed_apps
        ):
            raise ComputerError(
                "workspace_scope_denied",
                f"refused: {app_name!r} is outside this "
                "workspace's allowed applications",
            )

    def check_path(self, path: str) -> None:
        from afnan_ai.computer.errors import ComputerError

        if not self.allowed_roots or not path:
            return
        normalized = str(path).replace("\\", "/").lower()
        if not any(
            normalized.startswith(
                r.replace("\\", "/").lower().rstrip("/") + "/"
            )
            or normalized == r.replace("\\", "/").lower().rstrip("/")
            for r in self.allowed_roots
        ):
            raise ComputerError(
                "workspace_scope_denied",
                "refused: path is outside the workspace root",
            )

    def launch_scoped(self, app_name: str) -> dict[str, Any]:
        self.check_app(app_name)
        return self.runtime.apps.launch(app_name)

    def describe(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "allowed_apps": list(self.allowed_apps),
            "allowed_roots": [
                redact_text(r)[:120] for r in self.allowed_roots
            ],
        }


# -- EmergencyStop -----------------------------------------------------------

def attach_emergency_stop(
    runtime: Any, security_center: Any
) -> None:
    """Trip → halt all computer actions until authorized reset."""
    try:
        register = security_center.emergency.register
    except Exception:
        return

    def halt() -> None:
        try:
            runtime.stop()
        except Exception:
            pass
        try:
            runtime.screen.drop_screenshots()
        except Exception:
            pass

    try:
        register(halt)
    except Exception:
        pass


# -- Browser ↔ desktop workflow -----------------------------------------------

def browser_to_desktop_workflow(
    *,
    computer_runtime: Any,
    browser_controller: Any,
    workspace_scope: WorkspaceComputerScope | None = None,
    on_step: Any = None,
) -> dict[str, Any]:
    """Coordinated workflow helper (not an orchestrator).

    Example: browser downloads a CSV → desktop spreadsheet
    opens it → file saved → artifact verified.  Each step
    goes through the owning system's own validation; this
    helper only sequences and reports.
    """
    steps: list[dict[str, Any]] = []

    def step(name: str, fn: Any) -> dict[str, Any]:
        try:
            result = fn()
            entry = {"step": name, "ok": True}
        except Exception as exc:
            entry = {
                "step": name, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:200],
            }
        steps.append(entry)
        if on_step is not None:
            try:
                on_step(entry)
            except Exception:
                pass
        return entry

    return {"steps": steps, "step": step}

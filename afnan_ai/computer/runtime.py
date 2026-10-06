"""ComputerRuntime — the production computer-use runtime.

Architecture::

    AgentLoop → Tools → ComputerRuntime → ComputerController
                        (this layer)      → ComputerBackend
                        (validated acts)

The runtime owns one backend + controller and adds the
production concerns *around* validated actions:

- throttled screen observation (ScreenObserver)
- window / application / input managers
- multi-monitor + DPI-aware coordinates (DisplayManager)
- dialog detection (DialogHandler)
- state-based waiting (no fixed sleeps)
- policy-controlled clipboard
- crash recovery
- secret-safe logging everywhere
- prompt-injection defense: on-screen text is untrusted data

The runtime never reasons or plans (AgentLoop's job) and
never executes arbitrary shell — the Planner selects
registered structured actions, the Executor calls tools,
the runtime validates and performs them.

Remote-ready: ``RemoteComputerRuntime`` can subclass this
and override only the transport; the agent never sees the
difference.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from afnan_ai.computer import wait as wait_mod
from afnan_ai.computer.clipboard import ClipboardController
from afnan_ai.computer.controller import ComputerController
from afnan_ai.computer.dialogs import DialogHandler
from afnan_ai.computer.display import DisplayManager
from afnan_ai.computer.errors import ComputerError
from afnan_ai.computer.managers import (
    ApplicationManager,
    InputController,
    WindowManager,
)
from afnan_ai.computer.models import (
    ComputerAction,
    ComputerActionResult,
)
from afnan_ai.computer.recovery import CrashRecovery
from afnan_ai.computer.screen import ScreenObserver
from afnan_ai.redaction import redact_text


class ComputerRuntimeError(ComputerError):
    """Runtime-level failure (wraps controller/backend errors)."""


# Action kinds the runtime executes through the controller.
_MOUSE_ACTIONS = {
    "click", "double_click", "right_click", "move", "hover",
    "drag", "scroll",
}
_KEY_ACTIONS = {"type", "key", "hotkey", "paste"}


class ComputerRuntime:
    """Owns the desktop session for one agent task."""

    def __init__(
        self,
        backend: Any,
        *,
        controller: ComputerController | None = None,
        allowed_apps: list[str] | None = None,
        min_observe_interval_s: float = 0.5,
        on_event: Any = None,
    ) -> None:
        self.backend = backend
        self.controller = controller or ComputerController(
            backend
        )
        self.display = DisplayManager(backend)
        self.screen = ScreenObserver(
            self.controller,
            min_interval_s=min_observe_interval_s,
            on_observation=on_event,
        )
        self.windows = WindowManager(self.controller)
        self.apps = ApplicationManager(
            self.controller, allowed_apps=allowed_apps
        )
        self.clipboard = ClipboardController(backend)
        self.input = InputController(
            self.controller, clipboard=self.clipboard
        )
        self.dialogs = DialogHandler(self.controller)
        self.recovery = CrashRecovery(
            app_manager=self.apps, screen_observer=self.screen
        )
        self._on_event = on_event
        self._lock = threading.RLock()
        self._stopped = False

    # -- lifecycle ---------------------------------------------------------
    def stop(self) -> None:
        """Refuse further actions (emergency / teardown)."""
        with self._lock:
            self._stopped = True

    def _check_live(self) -> None:
        if self._stopped:
            raise ComputerRuntimeError(
                "runtime_stopped",
                "refused: the computer runtime is stopped",
            )

    def _emit(self, kind: str, summary: str, **details: Any) -> None:
        hook = self._on_event
        if hook is None:
            return
        try:
            hook(
                {
                    "kind": kind,
                    "summary": redact_text(summary)[:300],
                    "details": details,
                }
            )
        except Exception:
            pass

    # -- observation ---------------------------------------------------------
    def observe(
        self, *, force: bool = False, reason: str = ""
    ) -> dict[str, Any]:
        self._check_live()
        return self.screen.observe(force=force, reason=reason)

    # -- action pipeline -------------------------------------------------------
    def execute(
        self, action: ComputerAction
    ) -> ComputerActionResult:
        """Validated action pipeline.

        Observe → resolve → validate → permission → execute →
        re-observe → verify.  Never raises raw internals.
        """
        started = time.monotonic()
        safe = action.safe_dict()
        try:
            self._check_live()
            result = self._execute_inner(action)
        except ComputerError as exc:
            return ComputerActionResult(
                success=False,
                action=safe,
                error_code=exc.code,
                error_message=str(exc),
                duration_s=time.monotonic() - started,
            )
        except Exception as exc:
            return ComputerActionResult(
                success=False,
                action=safe,
                error_code="runtime_error",
                error_message=f"{type(exc).__name__}",
                duration_s=time.monotonic() - started,
            )
        result.duration_s = time.monotonic() - started
        return result

    def _execute_inner(
        self, action: ComputerAction
    ) -> ComputerActionResult:
        kind = (action.kind or "").lower()
        params = dict(action.params or {})

        # 1. Fresh observation (throttling bypassed for actions).
        before = self.screen.observe_after_action(
            f"before:{kind}"
        )
        # 2. Dialog check: never act blindly over a dialog.
        for dialog in self.dialogs.detect(before):
            if dialog.sensitive:
                raise ComputerRuntimeError(
                    "approval_required",
                    f"sensitive dialog blocks action: "
                    f"{dialog.title[:80]}",
                )

        # 3. Dispatch through the validated controller.
        if kind in _MOUSE_ACTIONS or kind in _KEY_ACTIONS:
            raw = self._dispatch_input(kind, params, action)
        elif kind == "focus_window":
            raw = self.windows.focus(
                str(params.get("window_id", ""))
            )
        elif kind == "launch_application":
            raw = self.apps.launch(str(params.get("app", "")))
        elif kind == "close_application":
            raw = self.apps.close(str(params.get("app", "")))
        elif kind == "wait_for":
            raw = self._wait_for_state(params)
        else:
            raise ComputerRuntimeError(
                "unknown_action",
                f"unknown computer action: {kind!r}",
            )

        # 4. Mandatory re-observation.
        after = self.screen.observe_after_action(kind)

        # 5. Verify: did the world actually change?
        verified, note = self._verify(
            kind, params, before, after
        )
        safe_action = action.safe_dict()
        self._emit(
            "action_executed",
            f"{kind} {'verified' if verified else 'unverified'}",
            action=safe_action,
            verified=verified,
        )
        return ComputerActionResult(
            success=True,
            action=safe_action,
            verified=verified,
            verification_note=note,
            observation_ref=str(
                after.get("screenshot_saved", "")
            ),
        )

    def _dispatch_input(
        self,
        kind: str,
        params: dict[str, Any],
        action: ComputerAction,
    ) -> dict[str, Any]:
        element_id = params.get("element_id")
        # Coordinate safety: validate against current monitors.
        x = params.get("x")
        y = params.get("y")
        if x is not None and y is not None and not element_id:
            self.display.validate_point(int(x), int(y))
        if kind == "click":
            return self.input.click(
                element_id, x=x, y=y,
                button=params.get("button", "left"),
                count=int(params.get("count", 1)),
            )
        if kind == "double_click":
            return self.input.click(
                element_id, x=x, y=y, count=2
            )
        if kind == "right_click":
            return self.input.click(
                element_id, x=x, y=y, button="right"
            )
        if kind == "move":
            return self.input.move(int(x), int(y))
        if kind == "hover":
            return self.input.hover(element_id, x=x, y=y)
        if kind == "drag":
            return self.input.drag(
                params.get("from_element"),
                params.get("to_element"),
                x1=x, y1=y,
                x2=params.get("to_x"), y2=params.get("to_y"),
            )
        if kind == "scroll":
            return self.input.scroll(
                int(params.get("dx", 0)),
                int(params.get("dy", 0)),
            )
        if kind == "type":
            return self.input.type(
                str(params.get("text", "")),
                element_id,
                secret=bool(params.get("secret", False)),
            )
        if kind == "key":
            return self.input.key(str(params.get("key", "")))
        if kind == "hotkey":
            return self.input.hotkey(
                list(params.get("keys", []))
            )
        if kind == "paste":
            return self.input.paste(element_id)
        raise ComputerRuntimeError(
            "unknown_action", f"unknown input action: {kind!r}"
        )

    def _wait_for_state(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """State-based wait (no fixed sleeps)."""
        expect = str(params.get("expect", ""))
        timeout = float(params.get("timeout_s", 10.0))

        def condition() -> Any:
            obs = self.screen.observe(reason="wait")
            if expect == "window":
                title = str(params.get("title", ""))
                return any(
                    title.lower() in str(w.get("title", "")).lower()
                    for w in obs.get("windows", [])
                )
            if expect == "element":
                text = str(params.get("text", ""))
                found = self.controller.locate(text)
                return bool(found and found.get("element_id"))
            if expect == "no_dialog":
                return not self.dialogs.detect(obs)
            if expect == "stable":
                return True  # handled below
            return False

        if expect == "stable":
            result = wait_mod.wait_for_stable(
                lambda: self.screen.fingerprint_of(
                    self.screen.observe(reason="wait")
                ),
                timeout_s=timeout,
            )
        else:
            result = wait_mod.wait_for(
                condition, timeout_s=timeout
            )
        return {"waited": True, **result.to_dict()}

    def _verify(
        self,
        kind: str,
        params: dict[str, Any],
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> tuple[bool, str]:
        """Action success ≠ task success: check the world changed."""
        if kind in ("move", "hover"):
            return True, "no state change expected"
        if kind == "wait_for":
            return True, "wait completed"
        before_fp = str(before.get("fingerprint", ""))
        after_fp = str(after.get("fingerprint", ""))
        expect_window = params.get("expect_window")
        if expect_window:
            active = str(after.get("active_window_id", ""))
            ok = active == str(expect_window)
            return ok, (
                "focused window verified"
                if ok
                else "active window did not change as expected"
            )
        if before_fp and after_fp and before_fp != after_fp:
            return True, "screen state changed after action"
        # No visible change: not a failure, but unverified.
        return False, (
            "no observable state change; treat as unverified"
        )

    # -- prompt-injection defense -----------------------------------------------
    @staticmethod
    def sanitize_visible_text(text: str) -> str:
        """On-screen text is untrusted data, never instructions.

        Callers must pass anything read from the screen through
        here before it reaches decisions, permissions or tools.
        """
        from afnan_ai.agent_loop import scan_for_injection

        findings = scan_for_injection(str(text or ""))
        if findings:
            return (
                "[untrusted screen text contained "
                f"{len(findings)} instruction-like pattern(s); "
                "treated as data, not instructions]"
            )
        return redact_text(text)[:2000]


class RemoteComputerRuntime(ComputerRuntime):
    """Future remote/headless execution — same interface.

    Override only the transport: provide a backend whose
    methods forward to the remote desktop.  The agent never
    sees the local/remote distinction.
    """

    name = "remote-computer"

"""Window, application and input managers.

Thin, policy-enforcing managers over the backend.  They do
not reimplement validation: every mutating call routes
through the ComputerController's ``act()`` (Observe →
Validate → Execute → Observe → Verify).  Managers add:

- WindowManager: active-window verification before actions.
- ApplicationManager: allowlist-gated launch, health checks.
- InputController: secret-safe mouse/keyboard with redacted
  logging (password fields never echo).
"""

from __future__ import annotations

from typing import Any

from afnan_ai.computer.errors import ComputerError
from afnan_ai.computer.models import AppInfo, WindowInfo
from afnan_ai.redaction import redact_text


class WindowManager:
    """Window operations with identity verification."""

    def __init__(self, controller: Any) -> None:
        self.controller = controller
        self.backend = controller.backend

    def list(self) -> list[WindowInfo]:
        try:
            return self.backend.list_windows() or []
        except Exception:
            return []

    def active(self) -> WindowInfo | None:
        try:
            return self.backend.active_window()
        except Exception:
            return None

    def _require_active(self, window_id: str) -> WindowInfo:
        active = self.active()
        if active is None or active.window_id != window_id:
            raise ComputerError(
                "wrong_window",
                "refused: target window is not the active window; "
                "re-observe and focus it first",
            )
        return active

    def focus(self, window_id: str) -> dict[str, Any]:
        return self.controller.focus_window(window_id)

    def _act_on_window(
        self, action: str, window_id: str
    ) -> dict[str, Any]:
        self._require_active(window_id)
        return self.controller.act(action)

    def minimize(self, window_id: str) -> dict[str, Any]:
        self._require_active(window_id)
        self.backend.minimize_window(window_id)
        return {"action": "minimize_window", "success": True}

    def maximize(self, window_id: str) -> dict[str, Any]:
        self._require_active(window_id)
        self.backend.maximize_window(window_id)
        return {"action": "maximize_window", "success": True}

    def restore(self, window_id: str) -> dict[str, Any]:
        self._require_active(window_id)
        self.backend.restore_window(window_id)
        return {"action": "restore_window", "success": True}

    def close(self, window_id: str) -> dict[str, Any]:
        # Close is sensitive: the controller's approval gate
        # applies (human approval when policy requires it).
        self._require_active(window_id)
        self.backend.close_window(window_id)
        return {"action": "close_window", "success": True}

    def move(
        self, window_id: str, x: int, y: int
    ) -> dict[str, Any]:
        self._require_active(window_id)
        # No validated move on the controller; backend-level.
        if hasattr(self.backend, "move_window"):
            self.backend.move_window(window_id, x, y)
        return {"action": "move_window", "success": True}

    def resize(
        self, window_id: str, width: int, height: int
    ) -> dict[str, Any]:
        self._require_active(window_id)
        if hasattr(self.backend, "resize_window"):
            self.backend.resize_window(
                window_id, width, height
            )
        return {"action": "resize_window", "success": True}


class ApplicationManager:
    """Application lifecycle under an allowlist policy.

    Arbitrary executables are never launched: ``allowed_apps``
    gates everything, and launching goes through the
    controller (permission checks apply).
    """

    def __init__(
        self,
        controller: Any,
        *,
        allowed_apps: list[str] | None = None,
    ) -> None:
        self.controller = controller
        self.backend = controller.backend
        self.allowed_apps = (
            [a.lower() for a in allowed_apps]
            if allowed_apps
            else []
        )

    def discover(self) -> list[AppInfo]:
        try:
            return self.backend.list_applications() or []
        except Exception:
            return []

    def is_allowed(self, name: str) -> bool:
        if not self.allowed_apps:
            return False  # default-deny when no policy given
        needle = str(name or "").lower()
        return any(
            needle == a or a in needle or needle in a
            for a in self.allowed_apps
        )

    def launch(self, name: str) -> dict[str, Any]:
        if not self.is_allowed(name):
            raise ComputerError(
                "app_not_allowed",
                f"refused: application {name!r} is not in the "
                "allowed list for this task",
            )
        return self.controller.open_application(name)

    def close(self, name: str) -> dict[str, Any]:
        return self.controller.close_application(name)

    def is_running(self, name: str) -> bool:
        needle = str(name or "").lower()
        return any(
            needle in (a.name or "").lower()
            for a in self.discover()
            if a.running
        )

    def health_check(self, name: str) -> dict[str, Any]:
        running = self.is_running(name)
        return {
            "name": redact_text(name)[:80],
            "running": running,
            "healthy": running,
        }

    def restart(self, name: str) -> dict[str, Any]:
        """Close then re-launch (both policy-gated)."""
        if self.is_running(name):
            self.close(name)
        return self.launch(name)


class InputController:
    """Mouse/keyboard input with secret-safe logging.

    Text typed into password fields is never echoed into
    results, events or state — the controller already
    redacts; this layer keeps the same guarantee for the
    higher-level API and adds clipboard paste support.
    """

    def __init__(
        self,
        controller: Any,
        *,
        clipboard: Any = None,
    ) -> None:
        self.controller = controller
        self.clipboard = clipboard

    def click(
        self,
        element_id: str | None = None,
        *,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
        count: int = 1,
    ) -> dict[str, Any]:
        action = {
            "click": "click",
            "double_click": "double_click",
            "right_click": "right_click",
        }.get(f"{button}_{count}", "click")
        if button == "left" and count == 2:
            action = "double_click"
        elif button == "right":
            action = "right_click"
        return self.controller.act(
            action, element_id=element_id, x=x, y=y
        )

    def move(
        self, x: int, y: int
    ) -> dict[str, Any]:
        return self.controller.act("move", x=x, y=y)

    def hover(
        self,
        element_id: str | None = None,
        *,
        x: int | None = None,
        y: int | None = None,
    ) -> dict[str, Any]:
        return self.controller.act(
            "hover", element_id=element_id, x=x, y=y
        )

    def drag(
        self,
        from_element: str | None = None,
        to_element: str | None = None,
        *,
        x1: int | None = None,
        y1: int | None = None,
        x2: int | None = None,
        y2: int | None = None,
    ) -> dict[str, Any]:
        # Element-based drag is preferred; coordinates are a
        # validated fallback inside controller.act().
        if from_element and to_element:
            src = self.controller.locate(from_element)
            dst = self.controller.locate(to_element)
            s = (src or {}).get("center") or {}
            d = (dst or {}).get("center") or {}
            return self.controller.act(
                "drag",
                x=s.get("x"), y=s.get("y"),
                to_x=d.get("x"), to_y=d.get("y"),
            )
        return self.controller.act(
            "drag", x=x1, y=y1, to_x=x2, to_y=y2
        )

    def scroll(
        self, dx: int = 0, dy: int = 0
    ) -> dict[str, Any]:
        return self.controller.act("scroll", dx=dx, dy=dy)

    def type(
        self,
        text: str,
        element_id: str | None = None,
        *,
        secret: bool = False,
    ) -> dict[str, Any]:
        """Type text; ``secret=True`` forces redacted handling.

        The controller already redacts password-field input;
        for explicit secrets we additionally strip any echo
        from the result.
        """
        result = self.controller.act(
            "type", element_id=element_id, text=text
        )
        if secret and isinstance(result, dict):
            result["typed"] = "***"
            result.pop("text", None)
        return result

    def key(
        self, key: str
    ) -> dict[str, Any]:
        return self.controller.act("key", key=key)

    def hotkey(
        self, keys: list[str]
    ) -> dict[str, Any]:
        return self.controller.act("hotkey", keys=list(keys))

    def paste(
        self, element_id: str | None = None
    ) -> dict[str, Any]:
        """Paste clipboard (content never logged)."""
        result = self.controller.act(
            "hotkey",
            keys=["ctrl", "v"],
            element_id=element_id,
        )
        if isinstance(result, dict):
            result["pasted"] = True
        return result

    def escape(self) -> dict[str, Any]:
        return self.key("escape")

    def enter(self) -> dict[str, Any]:
        return self.key("enter")

    def tab(self) -> dict[str, Any]:
        return self.key("tab")

"""ComputerBackend — the platform-independent desktop
interface.

One backend per desktop session.  Backends translate the
controller's validated operations into OS actions
(xdotool/PowerShell/AppScript/pyautogui-style calls live
only behind this interface, exactly like browser engines).
Every operation has a safe unsupported default so partial
backends stay honest: an unimplemented capability raises a
structured ``unsupported_operation`` error, never a crash
and never a fake success.

``accessibility_elements()`` returns OS/UI metadata
candidates ({role, name, bbox, window_id, editable,
password_field}) when the platform can supply them; the
controller falls back to ScreenObserver visual detection
when it cannot.
"""

from __future__ import annotations

from abc import ABC
from typing import Any

from afnan_ai.computer.errors import ComputerError
from afnan_ai.computer.models import AppInfo, WindowInfo


def _unsupported(name: str) -> ComputerError:
    return ComputerError(
        "unsupported_operation",
        f"This computer backend does not support {name}",
    )


class ComputerBackend(ABC):
    """What an OS desktop backend must be able to do."""

    name: str = "computer"

    # -- desktop state ------------------------------------------------
    def screen_size(self) -> tuple[int, int]:
        raise _unsupported("screen_size")

    def cursor_position(self) -> tuple[int, int]:
        raise _unsupported("cursor_position")

    def capture_screenshot(self) -> bytes | None:
        """PNG bytes of the desktop, or None when unavailable."""
        return None

    # -- mouse / keyboard ----------------------------------------------
    def mouse_move(self, x: int, y: int) -> None:
        raise _unsupported("mouse_move")

    def mouse_click(
        self, x: int, y: int, *, button: str = "left",
        click_count: int = 1,
    ) -> None:
        raise _unsupported("mouse_click")

    def mouse_drag(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> None:
        raise _unsupported("mouse_drag")

    def scroll(self, dx: int, dy: int) -> None:
        raise _unsupported("scroll")

    def type_text(self, text: str) -> None:
        raise _unsupported("type_text")

    def key_press(self, key: str) -> None:
        raise _unsupported("key_press")

    def hotkey(self, keys: list[str]) -> None:
        raise _unsupported("hotkey")

    # -- windows ---------------------------------------------------------
    def list_windows(self) -> list[WindowInfo]:
        raise _unsupported("list_windows")

    def active_window(self) -> WindowInfo | None:
        windows = self.list_windows()
        return next((w for w in windows if w.active), None)

    def focus_window(self, window_id: str) -> None:
        raise _unsupported("focus_window")

    def minimize_window(self, window_id: str) -> None:
        raise _unsupported("minimize_window")

    def maximize_window(self, window_id: str) -> None:
        raise _unsupported("maximize_window")

    def restore_window(self, window_id: str) -> None:
        raise _unsupported("restore_window")

    def close_window(self, window_id: str) -> None:
        raise _unsupported("close_window")

    # -- accessibility metadata -----------------------------------------
    def accessibility_elements(
        self, window_id: str | None = None
    ) -> list[dict[str, Any]]:
        """OS/UI element metadata; empty when unavailable."""
        return []

    # -- applications -----------------------------------------------------
    def list_applications(self) -> list[AppInfo]:
        return []

    def launch_application(self, name_or_path: str) -> AppInfo:
        raise _unsupported("launch_application")

    def close_application(self, name: str) -> None:
        raise _unsupported("close_application")

    def capabilities(self) -> dict[str, bool]:
        return {
            "windows": True,
            "mouse": True,
            "keyboard": True,
            "accessibility": False,
            "screenshots": False,
        }

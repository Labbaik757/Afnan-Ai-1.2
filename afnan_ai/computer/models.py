"""Serializable desktop models — no OS objects cross here."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from afnan_ai.redaction import redact_text


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class WindowInfo:
    window_id: str
    title: str
    app: str = ""
    active: bool = False
    minimized: bool = False
    maximized: bool = False
    bbox: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "title": redact_text(self.title)[:160],
            "app": self.app[:80],
            "active": self.active,
            "minimized": self.minimized,
            "maximized": self.maximized,
            "bbox": self.bbox,
        }


@dataclass
class AppInfo:
    name: str
    executable: str = ""
    running: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "executable": self.executable,
            "running": self.running,
        }


@dataclass
class ComputerElement:
    """One semantic desktop target.

    ``source`` is ``accessibility`` (OS/UI metadata — identity
    known), ``window`` (a window itself) or ``visual`` (pixel
    detection fallback).  ``element_id`` stays empty for
    low-confidence candidates so a guess can never be acted
    on.
    """

    element_id: str
    role: str
    name: str
    source: str = "accessibility"
    confidence: float = 0.9
    bbox: dict[str, int] | None = None
    center: dict[str, int] | None = None
    window_id: str = ""
    app: str = ""
    editable: bool = False
    password_field: bool = False
    value: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "role": self.role,
            "name": redact_text(self.name)[:120],
            "source": self.source,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox,
            "center": self.center,
            "window_id": self.window_id,
            "app": self.app,
            "editable": self.editable,
            "password_field": self.password_field,
            # Never echo a field's value back into state/logs.
            "value": "***" if self.password_field else self.value[:80],
        }


@dataclass
class ComputerObservation:
    """One structured snapshot of the whole desktop."""

    screen_width: int = 0
    screen_height: int = 0
    cursor: dict[str, int] | None = None
    windows: list[WindowInfo] = field(default_factory=list)
    active_window_id: str = ""
    active_app: str = ""
    elements: list[ComputerElement] = field(default_factory=list)
    fingerprint: str = ""
    screenshot_saved: str = ""
    captured_at: str = field(default_factory=_now)

    def to_dict(self, *, max_elements: int = 50) -> dict[str, Any]:
        active = next(
            (w for w in self.windows
             if w.window_id == self.active_window_id),
            None,
        )
        return {
            "screen": {
                "width": self.screen_width,
                "height": self.screen_height,
            },
            "cursor": self.cursor,
            "active_app": self.active_app,
            "active_window": (
                active.to_dict() if active else None
            ),
            "windows": [w.to_dict() for w in self.windows],
            "elements": [
                e.to_dict() for e in self.elements[:max_elements]
            ],
            "fingerprint": self.fingerprint,
            "screenshot_saved": self.screenshot_saved,
            "captured_at": self.captured_at,
        }


# ------------------------------------------------------------------
# ComputerRuntime models (production layer)
# ------------------------------------------------------------------

@dataclass
class MonitorInfo:
    """One physical display."""

    monitor_id: str
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    scale: float = 1.0  # DPI/display scaling factor
    primary: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "monitor_id": self.monitor_id,
            "bounds": {
                "x": self.x, "y": self.y,
                "width": self.width, "height": self.height,
            },
            "scale": self.scale,
            "primary": self.primary,
        }

    def contains(self, x: int, y: int) -> bool:
        return (
            self.x <= x < self.x + self.width
            and self.y <= y < self.y + self.height
        )


@dataclass
class DialogInfo:
    """A detected desktop dialog / popup."""

    dialog_id: str
    kind: str = "unknown"  # confirmation|save|file_picker|auth|
    # permission|error|info|unknown
    title: str = ""
    text: str = ""
    window_id: str = ""
    app: str = ""
    buttons: list[str] = field(default_factory=list)
    sensitive: bool = False  # irreversible/sensitive confirmation

    def to_dict(self) -> dict[str, Any]:
        return {
            "dialog_id": self.dialog_id,
            "kind": self.kind,
            "title": redact_text(self.title)[:160],
            "text": redact_text(self.text)[:300],
            "window_id": self.window_id,
            "app": self.app[:80],
            "buttons": [redact_text(b)[:60] for b in self.buttons],
            "sensitive": self.sensitive,
        }


@dataclass
class ComputerAction:
    """One validated desktop action request."""

    kind: str  # click|double_click|right_click|drag|scroll|
    # move|hover|type|key|hotkey|focus_window|launch_app|...
    target: str = ""  # element_id or description
    params: dict[str, Any] = field(default_factory=dict)
    risk: str = "LOW_RISK"  # READ_ONLY|LOW_RISK|WRITE|
    # SENSITIVE|IRREVERSIBLE
    reason: str = ""

    def safe_dict(self) -> dict[str, Any]:
        """Redacted representation for logs/events."""
        params = dict(self.params or {})
        # Never echo typed secrets.
        if self.kind in ("type", "paste") or params.get(
            "secret"
        ):
            params = {
                k: ("***" if k in ("text", "secret", "value") else v)
                for k, v in params.items()
            }
        return {
            "kind": self.kind,
            "target": redact_text(self.target)[:160],
            "risk": self.risk,
            "reason": redact_text(self.reason)[:200],
            "params": params,
        }


@dataclass
class ComputerActionResult:
    """Outcome of one runtime-executed action."""

    success: bool
    action: dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    verification_note: str = ""
    observation_ref: str = ""
    error_code: str = ""
    error_message: str = ""
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "action": self.action,
            "verified": self.verified,
            "verification_note": self.verification_note[:300],
            "observation_ref": self.observation_ref,
            "error_code": self.error_code,
            "error_message": redact_text(
                self.error_message
            )[:300],
            "duration_s": round(self.duration_s, 3),
        }

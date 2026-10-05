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

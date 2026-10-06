"""Display management: monitors, DPI/scaling, coordinates.

All coordinate conversion lives here, in one place.  Raw
coordinates are never treated as permanent element
identity: every conversion is validated against the current
monitor layout, and callers must re-observe before acting.

- Logical coordinates: what the agent reasons in.
- Physical coordinates: what the OS backend needs
  (logical × scale, per-monitor offset).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from afnan_ai.computer.models import MonitorInfo


@dataclass
class Point:
    x: int
    y: int


class DisplayManager:
    """Monitor discovery + centralized coordinate conversion."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self._monitors: list[MonitorInfo] = []
        self._refreshed = False

    def refresh(self) -> list[MonitorInfo]:
        """(Re)discover monitors from the backend."""
        raw = []
        try:
            raw = self.backend.list_monitors() or []
        except Exception:
            raw = []
        monitors: list[MonitorInfo] = []
        for i, m in enumerate(raw):
            if isinstance(m, MonitorInfo):
                monitors.append(m)
            elif isinstance(m, dict):
                monitors.append(
                    MonitorInfo(
                        monitor_id=str(
                            m.get(
                                "monitor_id", f"monitor-{i}"
                            )
                        ),
                        x=int(m.get("x", 0)),
                        y=int(m.get("y", 0)),
                        width=int(m.get("width", 0)),
                        height=int(m.get("height", 0)),
                        scale=float(m.get("scale", 1.0) or 1.0),
                        primary=bool(m.get("primary", i == 0)),
                    )
                )
        if not monitors:
            # Fallback: single virtual monitor from screen size.
            try:
                w, h = self.backend.screen_size()
            except Exception:
                w, h = 0, 0
            monitors = [
                MonitorInfo(
                    monitor_id="primary", x=0, y=0,
                    width=w, height=h, scale=1.0,
                    primary=True,
                )
            ]
        self._monitors = monitors
        self._refreshed = True
        return monitors

    @property
    def monitors(self) -> list[MonitorInfo]:
        if not self._refreshed:
            self.refresh()
        return list(self._monitors)

    def primary(self) -> MonitorInfo:
        for m in self.monitors:
            if m.primary:
                return m
        return self.monitors[0]

    def monitor_at(
        self, x: int, y: int
    ) -> MonitorInfo | None:
        for m in self.monitors:
            if m.contains(x, y):
                return m
        return None

    def virtual_bounds(self) -> dict[str, int]:
        ms = self.monitors
        if not ms:
            return {"x": 0, "y": 0, "width": 0, "height": 0}
        x0 = min(m.x for m in ms)
        y0 = min(m.y for m in ms)
        x1 = max(m.x + m.width for m in ms)
        y1 = max(m.y + m.height for m in ms)
        return {
            "x": x0, "y": y0,
            "width": x1 - x0, "height": y1 - y0,
        }

    # -- coordinate conversion -------------------------------------------
    def to_physical(self, x: int, y: int) -> Point:
        """Logical → physical (OS) coordinates."""
        m = self.monitor_at(x, y) or self.primary()
        px = m.x + round((x - m.x) * m.scale)
        py = m.y + round((y - m.y) * m.scale)
        return Point(px, py)

    def to_logical(self, px: int, py: int) -> Point:
        """Physical → logical coordinates."""
        m = self.monitor_at(px, py) or self.primary()
        x = m.x + round((px - m.x) / m.scale) if m.scale else px
        y = m.y + round((py - m.y) / m.scale) if m.scale else py
        return Point(x, y)

    def validate_point(self, x: int, y: int) -> Point:
        """Clamp/verify a logical point is on a real monitor.

        Raises ValueError when no monitor contains it — the
        caller must re-observe instead of clicking blindly.
        """
        m = self.monitor_at(x, y)
        if m is None:
            raise ValueError(
                f"point ({x}, {y}) is not on any known monitor; "
                "re-observe before acting"
            )
        return Point(x, y)

    def validate_bbox(
        self, bbox: dict[str, int] | None
    ) -> dict[str, int]:
        if not bbox:
            raise ValueError("missing bounds for target")
        for key in ("x", "y", "width", "height"):
            if key not in bbox:
                raise ValueError(
                    f"bounds missing {key!r}"
                )
        cx = int(bbox["x"] + bbox["width"] // 2)
        cy = int(bbox["y"] + bbox["height"] // 2)
        self.validate_point(cx, cy)
        return bbox

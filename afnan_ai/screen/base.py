"""ScreenObserver core types — structured visual information.

The ScreenObserver turns pixels (a desktop capture or a browser
page screenshot) into *structured* observations: screen
dimensions, prominent visual regions and candidate UI elements,
each with a confidence score.  The Planner never receives raw
image bytes; it receives these records, which are also what gets
recorded in AgentState.

Confidence tiers (see :class:`ScreenObserver.assess_action`):
elements at or above ``HIGH_CONFIDENCE`` may be acted on,
elements in the middle band should be confirmed by the Verifier,
and anything below ``MEDIUM_CONFIDENCE`` must go through human
approval — a low-confidence guess is never clicked blindly.

Failures are structured (:class:`ScreenException` with a
serializable :class:`ScreenError`), mirroring the browser
layer's error model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

#: confidence at/above which a visual element may be acted on
HIGH_CONFIDENCE = 0.8
#: confidence below which an action needs human approval
MEDIUM_CONFIDENCE = 0.5


class ScreenErrorCode(str, Enum):
    SCREEN_UNAVAILABLE = "screen_unavailable"
    CAPTURE_FAILED = "capture_failed"
    INVALID_IMAGE = "invalid_image"
    DETECTION_FAILED = "detection_failed"
    ELEMENT_NOT_FOUND = "element_not_found"


@dataclass
class ScreenError:
    """Structured, serializable description of a screen failure."""

    code: ScreenErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": dict(self.details),
        }


class ScreenException(Exception):
    """A screen operation failed in a structured, expected way."""

    def __init__(
        self,
        message: str,
        *,
        code: ScreenErrorCode = ScreenErrorCode.CAPTURE_FAILED,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.error = ScreenError(
            code=code, message=message, details=dict(details or {})
        )

    @property
    def code(self) -> ScreenErrorCode:
        return self.error.code


@dataclass
class VisualRegion:
    """A rectangle on the screen, in pixels (origin top-left)."""

    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        cx, cy = self.center
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "center": {"x": cx, "y": cy},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisualRegion":
        return cls(
            x=int(data.get("x", 0)),
            y=int(data.get("y", 0)),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
        )


@dataclass
class VisualElement:
    """One candidate UI element found on the screen.

    ``source`` is ``"dom"`` when the element came from the
    browser's DOM/accessibility tree (confidence 1.0 — its
    identity is known) and ``"visual"`` when it was detected
    from pixels alone (the fallback for canvas/custom-drawn UI,
    where DOM information does not exist).
    """

    ref: str
    kind: str
    confidence: float
    source: str = "visual"
    label: str = ""
    region: VisualRegion | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "kind": self.kind,
            "label": self.label,
            "confidence": round(float(self.confidence), 3),
            "source": self.source,
            "region": self.region.to_dict() if self.region else None,
            **{k: v for k, v in self.extra.items()},
        }


@dataclass
class ScreenObservation:
    """A structured snapshot of what is on the screen."""

    source: str
    width: int
    height: int
    elements: list[VisualElement] = field(default_factory=list)
    screen_changed: bool = False
    screenshot_path: str | None = None
    captured_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    dom_available: bool = False

    @property
    def dom_elements(self) -> list[VisualElement]:
        return [e for e in self.elements if e.source == "dom"]

    @property
    def visual_elements(self) -> list[VisualElement]:
        return [e for e in self.elements if e.source == "visual"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "screen_observation",
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "captured_at": self.captured_at,
            "screen_changed": self.screen_changed,
            "screenshot_path": self.screenshot_path,
            "dom_available": self.dom_available,
            "element_count": len(self.elements),
            "dom_element_count": len(self.dom_elements),
            "visual_element_count": len(self.visual_elements),
            "elements": [e.to_dict() for e in self.elements],
        }

    def summary(self, limit: int = 8) -> dict[str, Any]:
        """Compact form recorded in AgentState (no image data)."""
        top = sorted(
            self.elements, key=lambda e: e.confidence, reverse=True
        )[:limit]
        return {
            "kind": "screen_observation",
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "screen_changed": self.screen_changed,
            "screenshot_path": self.screenshot_path,
            "element_count": len(self.elements),
            "dom_element_count": len(self.dom_elements),
            "visual_element_count": len(self.visual_elements),
            "top_elements": [
                {
                    "ref": e.ref,
                    "kind": e.kind,
                    "label": e.label,
                    "confidence": round(float(e.confidence), 3),
                    "source": e.source,
                }
                for e in top
            ],
        }

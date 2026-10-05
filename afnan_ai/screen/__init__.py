"""ScreenObserver — structured visual observation for Afnan.

Captures the desktop (via the agent's existing capture) or a
browser page (via the BrowserController) and describes what is
on screen: dimensions, visible UI elements, regions and
confidence scores.  DOM/accessibility data wins where it
exists; pixel detection is the fallback where it does not.
Low-confidence detections are gated to the Verifier or human
approval — never acted on blindly.
"""
from afnan_ai.screen.base import (
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    ScreenError,
    ScreenErrorCode,
    ScreenException,
    ScreenObservation,
    VisualElement,
    VisualRegion,
)
from afnan_ai.screen.detector import ElementDetector, RegionDetector
from afnan_ai.screen.observer import ActionAssessment, ScreenObserver
from afnan_ai.screen.sources import (
    BrowserCaptureSource,
    CaptureSource,
    FunctionCaptureSource,
)
from afnan_ai.screen.tools import create_screen_tools, register_screen_tools

__all__ = [
    "HIGH_CONFIDENCE",
    "MEDIUM_CONFIDENCE",
    "ActionAssessment",
    "BrowserCaptureSource",
    "CaptureSource",
    "ElementDetector",
    "FunctionCaptureSource",
    "RegionDetector",
    "ScreenError",
    "ScreenErrorCode",
    "ScreenException",
    "ScreenObservation",
    "ScreenObserver",
    "VisualElement",
    "VisualRegion",
    "create_screen_tools",
    "register_screen_tools",
]

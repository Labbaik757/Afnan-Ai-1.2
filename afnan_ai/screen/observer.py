"""ScreenObserver — structured visual observation of the screen.

Pipeline: capture (desktop via the agent's capture, or a
browser page via the BrowserController) → decode → detect UI
elements from pixels → structured :class:`ScreenObservation`
with dimensions, regions, elements and confidence scores.

Design rules, mirroring the rest of the agent:

* **Structured output only.**  Raw image bytes never reach the
  Planner or AgentState; observations carry dimensions, element
  refs, regions and confidences (a screenshot *file path* may
  be included when the caller asks to save one).
* **DOM first, pixels as fallback.**  Browser observations fuse
  the controller's DOM elements (confidence 1.0 — identity is
  known) with visual detections; when the DOM is empty or
  unavailable (canvas apps, custom-drawn UI) the visual
  elements are all there is, clearly marked ``source="visual"``.
* **Never act on a guess.**  The observer clicks nothing.  It
  *assesses* proposed actions by confidence: high confidence may
  proceed, medium must be confirmed by the Verifier, and low
  confidence must go through human approval.  Any actuator
  (browser or future computer control) is expected to consult
  :meth:`assess_action` first.
* **Failures are structured**, never crashes and never fake
  observations.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from afnan_ai.log_config import get_logger
from afnan_ai.screen.base import (
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    ScreenErrorCode,
    ScreenException,
    ScreenObservation,
    VisualElement,
)
from afnan_ai.screen.detector import ElementDetector, RegionDetector
from afnan_ai.screen.image import decode_png
from afnan_ai.screen.sources import BrowserCaptureSource, CaptureSource

logger = get_logger(__name__)


@dataclass
class ActionAssessment:
    """May an actuator act on this element without a human?

    ``tier`` is ``"ok"`` (high confidence), ``"verify"``
    (medium — the Verifier should confirm against fresh state)
    or ``"approval_required"`` (low — a human decides).
    """

    ref: str
    confidence: float
    tier: str
    allowed_automatically: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "confidence": round(float(self.confidence), 3),
            "tier": self.tier,
            "allowed_automatically": self.allowed_automatically,
            "reason": self.reason,
        }


class ScreenObserver:
    """Observes a screen (or browser page) and describes it."""

    def __init__(
        self,
        source: CaptureSource | None = None,
        *,
        detector: ElementDetector | None = None,
        browser_controller: Any = None,
        high_confidence: float = HIGH_CONFIDENCE,
        medium_confidence: float = MEDIUM_CONFIDENCE,
    ):
        self._source = source
        self._detector = detector or RegionDetector()
        self._browser = browser_controller
        self.high_confidence = float(high_confidence)
        self.medium_confidence = float(medium_confidence)
        self._counter = 0
        self._last: ScreenObservation | None = None
        self._last_fingerprint: str | None = None
        self._by_ref: dict[str, VisualElement] = {}

    # -- configuration -------------------------------------------------
    def set_source(self, source: CaptureSource | None) -> None:
        self._source = source

    @property
    def last_observation(self) -> ScreenObservation | None:
        return self._last

    # -- observation ---------------------------------------------------
    def observe(
        self,
        *,
        origin: str = "desktop",
        save_dir: str | None = None,
        filename: str | None = None,
    ) -> ScreenObservation:
        """Capture the screen and return a structured observation.

        ``origin="browser"`` fuses the BrowserController's page
        screenshot and DOM elements (when a controller is
        attached); anything else uses the configured desktop
        capture source.
        """
        dom_elements: list[dict[str, Any]] = []
        dom_available = False
        if origin == "browser":
            if self._browser is None:
                raise ScreenException(
                    "No BrowserController is attached to this "
                    "ScreenObserver",
                    code=ScreenErrorCode.SCREEN_UNAVAILABLE,
                )
            source: CaptureSource = BrowserCaptureSource(self._browser)
            dom_elements, dom_available = self._dom_elements()
        else:
            if self._source is None:
                raise ScreenException(
                    "No screen capture source is configured",
                    code=ScreenErrorCode.SCREEN_UNAVAILABLE,
                )
            source = self._source

        frame = source.capture()  # ScreenException propagates
        image = decode_png(frame.image_bytes)  # invalid_image too
        try:
            detections = self._detector.detect(image)
        except ScreenException:
            raise
        except Exception as e:
            raise ScreenException(
                f"Visual detection failed: {e}",
                code=ScreenErrorCode.DETECTION_FAILED,
            ) from e

        elements: list[VisualElement] = []
        for dom in dom_elements:
            elements.append(self._next_element(
                kind=str(dom.get("tag", "element")),
                label=str(dom.get("text", "")),
                confidence=1.0,
                source="dom",
                extra={"dom_ref": dom.get("ref")},
            ))
        for det in detections:
            elements.append(self._next_element(
                kind=det.kind,
                label="",
                confidence=det.confidence,
                source="visual",
                region=det.region,
            ))

        fingerprint = hashlib.sha256(image.fingerprint_bytes()).hexdigest()
        observation = ScreenObservation(
            source=frame.origin,
            width=image.width,
            height=image.height,
            elements=elements,
            screen_changed=(
                self._last_fingerprint is not None
                and fingerprint != self._last_fingerprint
            ),
            dom_available=dom_available,
        )
        if save_dir:
            observation.screenshot_path = self._save(
                frame.image_bytes, save_dir, filename
            )
        # refs survive a re-observe of an unchanged screen; a
        # changed screen makes older refs stale (element_not_found)
        carried = (
            dict(self._by_ref)
            if fingerprint == self._last_fingerprint
            else {}
        )
        self._last = observation
        self._last_fingerprint = fingerprint
        carried.update({e.ref: e for e in elements})
        self._by_ref = carried
        logger.info(
            "screen observed: %s %sx%s elements=%s dom=%s visual=%s",
            observation.source, observation.width, observation.height,
            len(elements), len(observation.dom_elements),
            len(observation.visual_elements),
        )
        return observation

    def _dom_elements(self) -> tuple[list[dict[str, Any]], bool]:
        """DOM elements via the controller; ([], False) on any
        failure — the visual fallback then carries the page."""
        try:
            observed = self._browser.observe()
            if isinstance(observed, dict):
                return list(observed.get("elements") or []), True
        except Exception as e:
            logger.info("browser DOM unavailable, visual fallback: %s", e)
        return [], False

    def _next_element(self, **kwargs) -> VisualElement:
        self._counter += 1
        return VisualElement(ref=f"vis_{self._counter}", **kwargs)

    @staticmethod
    def _save(image_bytes: bytes, save_dir: str, filename: str | None) -> str:
        directory = Path(save_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timezone
        name = filename or (
            "screen_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            + ".png"
        )
        path = directory / name
        path.write_bytes(image_bytes)
        return str(path)

    # -- queries ---------------------------------------------------------
    def find_elements(
        self,
        *,
        min_confidence: float = 0.0,
        kind: str | None = None,
        source: str | None = None,
    ) -> list[VisualElement]:
        """Elements of the latest observation (observing first
        when there is none), filtered by confidence/kind/source."""
        observation = self._last or self.observe()
        return [
            e for e in observation.elements
            if e.confidence >= min_confidence
            and (kind is None or e.kind == kind)
            and (source is None or e.source == source)
        ]

    def get_element(self, ref: str) -> VisualElement:
        element = self._by_ref.get(str(ref))
        if element is None:
            raise ScreenException(
                f"Unknown or stale visual element reference: {ref!r} "
                "(observe again and use a fresh ref)",
                code=ScreenErrorCode.ELEMENT_NOT_FOUND,
                details={"ref": str(ref)},
            )
        return element

    # -- action gating ---------------------------------------------------
    def assess_action(self, ref: str) -> ActionAssessment:
        """Judge whether acting on *ref* may proceed automatically.

        DOM-identified elements are known (confidence 1.0);
        pixel-detected ones are judged by their score.  Low
        confidence never proceeds: it needs human approval.
        """
        element = self.get_element(ref)
        confidence = element.confidence
        if confidence >= self.high_confidence:
            return ActionAssessment(
                ref=ref, confidence=confidence, tier="ok",
                allowed_automatically=True,
                reason=(
                    f"Confidence {confidence:.2f} is high "
                    f"({element.source} element); action may proceed"
                ),
            )
        if confidence >= self.medium_confidence:
            return ActionAssessment(
                ref=ref, confidence=confidence, tier="verify",
                allowed_automatically=False,
                reason=(
                    f"Confidence {confidence:.2f} is medium; confirm "
                    "with the Verifier against fresh state before acting"
                ),
            )
        return ActionAssessment(
            ref=ref, confidence=confidence, tier="approval_required",
            allowed_automatically=False,
            reason=(
                f"Confidence {confidence:.2f} is low; human approval "
                "is required before acting on this element"
            ),
        )

    # -- Verifier hook ---------------------------------------------------
    def verifier_observation(self, step: Any = None) -> dict[str, Any] | None:
        """Fresh screen observation for ``screen_*`` steps.

        Plugs into the Verifier's generic observation-provider
        hook (the same one browser reliability uses): the
        Verifier then judges a screen step's expected result
        against what is actually on the screen now.  Returns
        None for non-screen steps or when the screen cannot be
        observed, so verification falls back to the step's own
        reported output.
        """
        tool_name = str(getattr(step, "tool_name", "")) if step else ""
        if step is not None and not tool_name.startswith("screen_"):
            return None
        try:
            origin = "desktop" if self._source is not None else "browser"
            return self.observe(origin=origin).to_dict()
        except ScreenException:
            return None

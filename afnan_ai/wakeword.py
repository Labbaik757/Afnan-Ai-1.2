"""On-device wake word detection for Afnan AI.

The default cloud loop (sending room audio to Google just to
string-match the wake word) is replaced, when configured,
by a fully local detector.  No account, no API key, no
network needed.

Design mirrors ``LLMProvider``/``PlatformAdapter``: a small
``WakeWordDetector`` interface, lazy optional dependency,
and automatic fallback to the previous behaviour when the
model or library is unavailable.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any


class WakeWordDetector(ABC):
    """Scores 16-bit mono 16 kHz PCM audio frames."""

    #: Samples per frame this detector consumes.
    frame_samples: int = 1280  # 80 ms @ 16 kHz

    @abstractmethod
    def score(self, frame: bytes) -> float:
        """Return 0.0–1.0 confidence for one frame."""

    def is_wake(self, frame: bytes, threshold: float) -> bool:
        try:
            return self.score(frame) >= threshold
        except Exception:
            return False


class FakeWakeWordDetector(WakeWordDetector):
    """Deterministic detector for tests and dry runs."""

    def __init__(self, scores: list[float] | None = None) -> None:
        self._scores = list(scores or [0.0])
        self.calls = 0

    def score(self, frame: bytes) -> float:
        self.calls += 1
        if len(self._scores) > 1:
            return self._scores.pop(0)
        return self._scores[0]


class OpenWakeWordDetector(WakeWordDetector):
    """openWakeWord-backed detector (Apache-2.0, offline).

    Needs ``pip install openwakeword`` and a trained
    ``.onnx`` model (see README: train the "Afnan" model
    with your own voice recordings for Urdu-accented
    pronunciation).
    """

    def __init__(
        self,
        model_path: str,
        *,
        threshold: float = 0.5,
    ) -> None:
        try:
            from openwakeword.model import Model
        except ImportError as exc:
            raise RuntimeError(
                "openwakeword is not installed "
                "(pip install openwakeword)"
            ) from exc
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"wake word model not found: {model_path}"
            )
        self._model = Model(
            wakeword_models=[model_path],
            inference_framework="onnx",
        )
        self.threshold = threshold

    def score(self, frame: bytes) -> float:
        import numpy as np

        audio = np.frombuffer(frame, dtype=np.int16)
        result = self._model.predict(audio)
        if not result:
            return 0.0
        return float(max(result.values()))


def create_detector(
    model_path: str | None,
    *,
    threshold: float = 0.5,
    fake_scores: list[float] | None = None,
) -> WakeWordDetector | None:
    """Build a detector, or return None when unavailable.

    Returns None (→ caller falls back to the cloud loop)
    when no model is configured or the library/import fails.
    """
    if fake_scores is not None:
        return FakeWakeWordDetector(fake_scores)
    if not model_path:
        return None
    try:
        return OpenWakeWordDetector(
            model_path, threshold=threshold
        )
    except Exception:
        return None


def detector_status(
    model_path: str | None,
) -> dict[str, Any]:
    """Describe wake-word readiness without side effects."""
    try:
        import openwakeword  # noqa: F401
        library = True
    except ImportError:
        library = False
    return {
        "model_configured": bool(model_path),
        "model_exists": bool(
            model_path and os.path.exists(model_path)
        ),
        "library_installed": library,
        "on_device": bool(
            model_path
            and os.path.exists(model_path)
            and library
        ),
    }

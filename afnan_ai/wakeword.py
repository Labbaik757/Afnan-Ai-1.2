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
    # Prefer the lightweight numpy JSON classifier, then a
    # custom ONNX classifier, then openWakeWord-format models.
    for cls in (
        NumpyWakeWordDetector,
        CustomONNXDetector,
        OpenWakeWordDetector,
    ):
        try:
            return cls(model_path, threshold=threshold)
        except Exception:
            continue
    return None


def detector_status(
    model_path: str | None,
) -> dict[str, Any]:
    """Describe wake-word readiness without side effects."""
    try:
        import numpy  # noqa: F401

        has_numpy = True
    except ImportError:
        has_numpy = False
    try:
        import onnxruntime  # noqa: F401

        ort = True
    except ImportError:
        ort = False
    try:
        import openwakeword  # noqa: F401

        oww = True
    except ImportError:
        oww = False
    model_ok = bool(model_path and os.path.exists(model_path))
    is_json = bool(model_path and model_path.endswith(".json"))
    on_device = bool(
        model_ok
        and (
            (is_json and has_numpy)
            or ((not is_json) and (ort or oww))
        )
    )
    return {
        "model_configured": bool(model_path),
        "model_exists": model_ok,
        "numpy_installed": has_numpy,
        "onnxruntime_installed": ort,
        "openwakeword_installed": oww,
        "library_installed": has_numpy or ort or oww,
        "on_device": on_device,
    }


# -- numpy MFCC (must match /tmp/wwtrain/train.py exactly) -----------

_MEL_FB: Any = None


def _mel_filterbank(
    sr: int = 16000, n_fft: int = 512, n_mels: int = 26
) -> Any:
    import numpy as np

    def hz_to_mel(hz: float) -> float:
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel: float) -> float:
        return 700 * (10 ** (mel / 2595) - 1)

    low, high = hz_to_mel(0), hz_to_mel(sr // 2)
    points = mel_to_hz(np.linspace(low, high, n_mels + 2))
    bins = np.floor((n_fft + 1) * points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f0, f1, f2 = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f0, f1):
            fb[m - 1, k] = (k - f0) / max(1, f1 - f0)
        for k in range(f1, f2):
            fb[m - 1, k] = (f2 - k) / max(1, f2 - f1)
    return fb


def _mfcc_sequence(wav: Any) -> Any:
    """Full MFCC sequence: (frames, 39). Temporal order kept.

    Must match /tmp/wwtrain/train.py exactly (numpy only).
    """
    import numpy as np

    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = _mel_filterbank()
    sr = 16000
    wav = np.append(wav[0], wav[1:] - 0.97 * wav[:-1])
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    frames = []
    for start in range(0, len(wav) - frame_len + 1, hop):
        frame = wav[start:start + frame_len] * np.hamming(
            frame_len
        )
        spec = np.abs(np.fft.rfft(frame, n=512)) ** 2
        mel = np.dot(_MEL_FB, spec)
        log_mel = np.log(mel + 1e-10)
        dct = np.fft.rfft(
            np.concatenate([log_mel, log_mel[::-1]]), n=64
        ).real[:13]
        frames.append(dct)
    frames = np.array(frames)
    if len(frames) < 3:
        frames = np.pad(frames, ((0, 3 - len(frames)), (0, 0)))
    delta = np.gradient(frames, axis=0)
    delta2 = np.gradient(delta, axis=0)
    return np.concatenate([frames, delta, delta2], axis=1)


def _mfcc_features(wav: Any) -> Any:
    """Downsampled temporal features: 20 frames x 39 = 780."""
    import numpy as np

    seq = _mfcc_sequence(wav)
    idx = np.linspace(0, len(seq) - 1, 20).astype(int)
    return seq[idx].ravel()


class NumpyWakeWordDetector(WakeWordDetector):
    """Pure-numpy wake-word classifier (no onnxruntime needed).

    Loads ``afnan.json`` (scaler + logistic-regression weights
    exported from training) and scores a sliding 1-second
    window.  Needs only ``numpy`` (lazy optional).
    """

    window_samples: int = 16000

    def __init__(
        self,
        model_path: str,
        *,
        threshold: float = 0.6,
        silence_rms: float = 0.02,
    ) -> None:
        import json as _json

        import numpy as np

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"wake word model not found: {model_path}"
            )
        with open(model_path, encoding="utf-8") as fh:
            params = _json.load(fh)
        if params.get("format") != "afnan-wakeword-v1":
            raise RuntimeError(
                "not an Afnan wake-word JSON model"
            )
        self._np = np
        self._mean = np.array(
            params["scaler_mean"], dtype=np.float64
        )
        self._scale = np.array(
            params["scaler_scale"], dtype=np.float64
        )
        self._weights = np.array(
            params["weights"], dtype=np.float64
        )
        self._intercept = float(params["intercept"])
        if int(params.get("n_features", 0)) != 780:
            raise RuntimeError("bad feature count in model")
        self.threshold = threshold
        self.silence_rms = silence_rms
        self._buffer = np.zeros(
            self.window_samples, dtype=np.float64
        )

    def score(self, frame: bytes) -> float:
        np = self._np
        audio = (
            np.frombuffer(frame, dtype=np.int16).astype(
                np.float64
            )
            / 32768.0
        )
        n = len(audio)
        if n >= self.window_samples:
            self._buffer = audio[-self.window_samples:]
        else:
            self._buffer = np.concatenate(
                [self._buffer[n:], audio]
            )
        window = self._buffer
        rms = float(np.sqrt(np.mean(window ** 2)))
        if rms < self.silence_rms:
            return 0.0
        peak = np.abs(window).max()
        if peak > 1e-6:
            window = window / peak * 0.9
        feats = _mfcc_features(window)
        z = (
            np.sum(
                ((feats - self._mean) / self._scale)
                * self._weights
            )
            + self._intercept
        )
        return float(1.0 / (1.0 + np.exp(-z)))


class CustomONNXDetector(WakeWordDetector):
    """ONNX classifier trained on the user's own voice.

    Sliding 1-second window over 80 ms frames; numpy MFCC
    features identical to the training pipeline.  Needs
    ``pip install numpy onnxruntime`` (both lazy optional).
    """

    window_samples: int = 16000

    def __init__(
        self,
        model_path: str,
        *,
        threshold: float = 0.6,
        silence_rms: float = 0.02,
    ) -> None:
        import numpy as np
        import onnxruntime

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"wake word model not found: {model_path}"
            )
        self._np = np
        self.session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        # Custom classifier expects 780 temporal MFCC features;
        # anything else is probably an openWakeWord-format model.
        if len(shape) < 2 or (
            isinstance(shape[1], int) and shape[1] != 780
        ):
            raise RuntimeError(
                "not a custom wake-word classifier model"
            )
        self.threshold = threshold
        self.silence_rms = silence_rms
        self._buffer = np.zeros(
            self.window_samples, dtype=np.float64
        )

    def score(self, frame: bytes) -> float:
        np = self._np
        audio = (
            np.frombuffer(frame, dtype=np.int16).astype(
                np.float64
            )
            / 32768.0
        )
        n = len(audio)
        if n >= self.window_samples:
            self._buffer = audio[-self.window_samples:]
        else:
            self._buffer = np.concatenate(
                [self._buffer[n:], audio]
            )
        window = self._buffer
        # Energy gate: silence stays silent (never amplified
        # into a false trigger by peak normalization).
        rms = float(np.sqrt(np.mean(window ** 2)))
        if rms < self.silence_rms:
            return 0.0
        peak = np.abs(window).max()
        if peak > 1e-6:
            window = window / peak * 0.9
        feats = _mfcc_features(window).astype(np.float32)
        outputs = self.session.run(
            None, {self._input_name: feats.reshape(1, -1)}
        )
        # skl2onnx zipmap=False → [labels, probabilities]
        proba = np.asarray(outputs[1]).ravel()
        return float(proba[-1]) if len(proba) else 0.0

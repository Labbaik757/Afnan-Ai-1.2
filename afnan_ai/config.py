"""Central configuration for Afnan AI.

One small, explicit place for the tunables that used to be
scattered as literals: the wake word, the orchestration limits
and the default model.  Components keep their own defaults (e.g.
``Agent.DEFAULT_MAX_ITERATIONS`` mirrors ``max_iterations`` here);
this dataclass is what the *application* passes in when it wants
to change them, optionally from environment variables.

Environment overrides (``AgentConfig.from_env()``)::

    AFNAN_WAKE_WORD, AFNAN_MAX_ITERATIONS,
    AFNAN_MAX_RECOVERY_ATTEMPTS, AFNAN_LLM_PROVIDER,
    AFNAN_LLM_MODEL, AFNAN_GIF_PATH, AFNAN_BROWSER_RUNTIME_DIR
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bundled_wakeword_model() -> str | None:
    """Path to the repo-bundled wake-word model, if present."""
    base = Path(__file__).resolve().parent / "wakeword_models"
    for name in ("afnan.json", "afnan.onnx"):
        p = base / name
        if p.exists():
            return str(p)
    return None


def _int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _float_from_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


@dataclass(frozen=True)
class AgentConfig:
    """Runtime configuration for the assistant + Agent pipeline."""

    wake_word: str = "afnan"
    #: Speech-to-text language for Google recognition
    #: ("ur-PK" for Urdu, "en-IN" for English).
    stt_language: str = "ur-PK"
    #: Path to an on-device wake-word .onnx model.
    #: Defaults to the bundled model trained on the user's
    #: own voice (afnan_ai/wakeword_models/afnan.onnx).
    #: None → cloud fallback loop.
    wakeword_model: str | None = _bundled_wakeword_model()  # type: ignore[assignment]
    #: Confidence threshold for the on-device wake detector.
    wakeword_threshold: float = 0.5
    #: Neural voice for Urdu TTS (edge-tts, no account).
    tts_urdu_voice: str = "ur-PK-GulNawazNeural"
    max_iterations: int = 10
    max_recovery_attempts: int = 2
    llm_provider: str = "ollama"
    llm_model: str = "llama3"
    #: Remote Ollama server URL (e.g. a GPU machine or Google Colab
    #: tunnel).  None → local Ollama at localhost:11434.
    llm_host: str | None = None
    gif_path: str = "afnan_animation.gif"
    #: Where the Afnan Browser Runtime persists profiles and
    #: session state.  None → ~/.afnan-ai/browser-runtime.
    browser_runtime_dir: str | None = None

    @classmethod
    def from_env(cls, prefix: str = "AFNAN_") -> "AgentConfig":
        """Build a config from environment variables (falling back
        to the defaults for anything unset or invalid)."""
        defaults = cls()
        return cls(
            wake_word=os.environ.get(
                f"{prefix}WAKE_WORD", defaults.wake_word
            ),
            stt_language=os.environ.get(
                f"{prefix}STT_LANGUAGE", defaults.stt_language
            ),
            wakeword_model=os.environ.get(
                f"{prefix}WAKEWORD_MODEL", defaults.wakeword_model
            ),
            wakeword_threshold=_float_from_env(
                f"{prefix}WAKEWORD_THRESHOLD",
                defaults.wakeword_threshold,
            ),
            tts_urdu_voice=os.environ.get(
                f"{prefix}TTS_URDU_VOICE", defaults.tts_urdu_voice
            ),
            max_iterations=_int_from_env(
                f"{prefix}MAX_ITERATIONS", defaults.max_iterations
            ),
            max_recovery_attempts=_int_from_env(
                f"{prefix}MAX_RECOVERY_ATTEMPTS",
                defaults.max_recovery_attempts,
            ),
            llm_provider=os.environ.get(
                f"{prefix}LLM_PROVIDER", defaults.llm_provider
            ),
            llm_model=os.environ.get(
                f"{prefix}LLM_MODEL", defaults.llm_model
            ),
            llm_host=os.environ.get(
                f"{prefix}LLM_HOST", defaults.llm_host
            ) or None,
            gif_path=os.environ.get(
                f"{prefix}GIF_PATH", defaults.gif_path
            ),
            browser_runtime_dir=os.environ.get(
                f"{prefix}BROWSER_RUNTIME_DIR",
                defaults.browser_runtime_dir,
            ),
        )

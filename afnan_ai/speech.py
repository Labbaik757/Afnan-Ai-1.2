"""Cross-platform text-to-speech for Afnan AI.

Tries the cross-platform ``pyttsx3`` engine first (SAPI on Windows,
NSSpeechSynthesizer on macOS, espeak on Linux) and falls back to the
platform adapter's native speech command.  No OS-specific command
lives in this module — that is the adapter's job.
"""

from __future__ import annotations

from afnan_ai.log_config import get_logger
from afnan_ai.platform.base import PlatformAdapter

logger = get_logger(__name__)

_engine = None


def speak(text: str, adapter: PlatformAdapter) -> None:
    print("afnan:", text)
    try:
        _speak_with_pyttsx3(text)
    except Exception:
        try:
            adapter.speak_system(text)
        except Exception as e:
            logger.warning("speech failed: %s", e)


def _speak_with_pyttsx3(text: str) -> None:
    global _engine
    import pyttsx3  # type: ignore

    if _engine is None:
        _engine = pyttsx3.init()
    _engine.say(text)
    _engine.runAndWait()

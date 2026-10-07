"""Cross-platform text-to-speech for Afnan AI.

Urdu-first: when the text contains Urdu (Arabic script),
the Urdu chain is tried first — edge-tts neural Urdu
voices (no account needed) → gTTS (lang='ur') — before
falling back to the system engine.  English text keeps the
previous behaviour (pyttsx3 → platform adapter).

Every backend is a lazy optional import; nothing here
crashes when a library is missing.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile

from afnan_ai.log_config import get_logger
from afnan_ai.platform.base import PlatformAdapter

logger = get_logger(__name__)

_engine = None

DEFAULT_URDU_VOICE = "ur-PK-GulNawazNeural"


def contains_urdu(text: str) -> bool:
    """True when *text* has Arabic-script (Urdu) characters."""
    return any("\u0600" <= ch <= "\u06ff" for ch in text)


def speak(
    text: str,
    adapter: PlatformAdapter,
    *,
    urdu_voice: str | None = None,
) -> None:
    print("afnan:", text)
    if contains_urdu(text):
        if _speak_urdu(text, voice=urdu_voice or DEFAULT_URDU_VOICE):
            return
    try:
        _speak_with_pyttsx3(text)
    except Exception:
        try:
            adapter.speak_system(text)
        except Exception as e:
            logger.warning("speech failed: %s", e)


def _speak_urdu(text: str, *, voice: str) -> bool:
    """Urdu TTS chain.  Returns True when audio played."""
    if _speak_with_edge_tts(text, voice=voice):
        return True
    if _speak_with_gtts(text):
        return True
    return False


def _speak_with_edge_tts(text: str, *, voice: str) -> bool:
    try:
        import asyncio

        import edge_tts
    except ImportError:
        return False
    try:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False
        )
        tmp.close()
        asyncio.run(
            edge_tts.Communicate(text, voice).save(tmp.name)
        )
        return _play_audio(tmp.name)
    except Exception as e:
        logger.warning("edge-tts failed: %s", e)
        return False


def _speak_with_gtts(text: str) -> bool:
    try:
        from gtts import gTTS
    except ImportError:
        return False
    try:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False
        )
        tmp.close()
        gTTS(text=text, lang="ur").save(tmp.name)
        return _play_audio(tmp.name)
    except Exception as e:
        logger.warning("gTTS failed: %s", e)
        return False


def _play_audio(path: str) -> bool:
    """Best-effort cross-platform audio playback."""
    try:
        if shutil.which("afplay"):  # macOS
            subprocess.run(
                ["afplay", path],
                capture_output=True,
                timeout=60,
            )
            return True
        for player in ("mpg123", "ffplay", "play"):
            if shutil.which(player):
                args = (
                    [player, "-nodisp", "-autoexit", "-loglevel",
                     "quiet", path]
                    if player == "ffplay"
                    else [player, "-q", path]
                )
                subprocess.run(
                    args, capture_output=True, timeout=60
                )
                return True
        # Windows: .NET MediaPlayer via PowerShell.
        import platform as _platform

        if _platform.system() == "Windows":
            ps = (
                "Add-Type -AssemblyName presentationCore;"
                f"$p=New-Object System.Windows.Media.MediaPlayer;"
                f"$p.Open('{path}');$p.Play();"
                "Start-Sleep -Milliseconds "
                "([int]($p.NaturalDuration.TimeSpan.TotalMilliseconds)+500)"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                capture_output=True,
                timeout=60,
            )
            return True
    except Exception as e:
        logger.warning("audio playback failed: %s", e)
    return False


def _get_engine():
    """Return the shared pyttsx3 engine, initializing it once."""
    global _engine
    if _engine is None:
        import pyttsx3  # type: ignore

        _engine = pyttsx3.init()
        # Some systems start pyttsx3 nearly muted; force full volume.
        try:
            _engine.setProperty("volume", 1.0)
        except Exception:
            pass
    return _engine


def warm_up() -> None:
    """Pre-initialize the TTS engine.

    Call once at startup so the first speak() doesn't pay the
    engine-initialization delay.
    """
    try:
        _get_engine()
    except Exception as e:
        logger.warning("tts warm-up failed: %s", e)


def _speak_with_pyttsx3(text: str) -> None:
    engine = _get_engine()
    engine.say(text)
    engine.runAndWait()

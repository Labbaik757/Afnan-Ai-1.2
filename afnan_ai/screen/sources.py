"""Screenshot capture sources for the ScreenObserver.

A capture source answers one question — "give me the current
screen as PNG bytes" — and nothing else.  Platform specifics
live behind the injected callable (the agent's existing desktop
capture) and behind the BrowserController (browser pages), so
this package contains no OS or driver code at all.
"""
from __future__ import annotations

import io
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from afnan_ai.screen.base import ScreenErrorCode, ScreenException


@dataclass
class CapturedFrame:
    image_bytes: bytes
    origin: str


class CaptureSource(ABC):
    """Anything that can produce a screenshot of something."""

    name = "source"

    @abstractmethod
    def capture(self) -> CapturedFrame:
        """Capture and return PNG bytes (or raise ScreenException)."""


def _normalize_image(result: Any) -> bytes:
    """Normalize a capture callable's return value to PNG bytes.

    Accepts raw PNG bytes, a path to an image file, or an
    image-like object with ``save`` (e.g. a PIL image, which is
    what the agent's desktop capture returns).
    """
    if isinstance(result, (bytes, bytearray)) and result:
        return bytes(result)
    if isinstance(result, (str, Path)):
        try:
            data = Path(result).read_bytes()
        except OSError as e:
            raise ScreenException(
                f"Could not read captured image at {result}: {e}",
                code=ScreenErrorCode.CAPTURE_FAILED,
            ) from e
        if data:
            return data
    save = getattr(result, "save", None)
    if callable(save):
        buffer = io.BytesIO()
        try:
            save(buffer, "PNG")
        except TypeError:
            save(buffer)
        data = buffer.getvalue()
        if data:
            return data
    raise ScreenException(
        "Capture returned no usable image data",
        code=ScreenErrorCode.CAPTURE_FAILED,
    )


class FunctionCaptureSource(CaptureSource):
    """Wraps a zero-argument callable that captures the desktop.

    The agent injects its existing ``_capture_screenshot`` (the
    pyautogui-backed capture), so desktop screenshots come from
    the same place they always have.
    """

    def __init__(self, capture_fn: Callable[[], Any], *, name: str = "desktop"):
        self._capture_fn = capture_fn
        self.name = name

    def capture(self) -> CapturedFrame:
        if self._capture_fn is None:
            raise ScreenException(
                "No desktop capture is available on this machine",
                code=ScreenErrorCode.SCREEN_UNAVAILABLE,
            )
        try:
            result = self._capture_fn()
        except ScreenException:
            raise
        except Exception as e:
            raise ScreenException(
                f"Desktop capture failed: {e}",
                code=ScreenErrorCode.CAPTURE_FAILED,
            ) from e
        if result is None:
            raise ScreenException(
                "Desktop capture is unavailable (no screen or the "
                "capture backend is missing)",
                code=ScreenErrorCode.SCREEN_UNAVAILABLE,
            )
        return CapturedFrame(_normalize_image(result), origin=self.name)


class BrowserCaptureSource(CaptureSource):
    """Screenshots of the current browser page, via the existing
    BrowserController (which stays the only browser driver)."""

    name = "browser"

    def __init__(self, controller: Any, *, tab_id: str | None = None):
        self._controller = controller
        self._tab_id = tab_id

    def capture(self) -> CapturedFrame:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = self._controller.screenshot(
                    output_dir=tmp, tab_id=self._tab_id
                )
                path = result.get("path") if isinstance(result, dict) else None
                if not path:
                    raise ScreenException(
                        "Browser screenshot returned no file",
                        code=ScreenErrorCode.CAPTURE_FAILED,
                    )
                data = Path(path).read_bytes()
        except ScreenException:
            raise
        except Exception as e:
            raise ScreenException(
                f"Browser screenshot failed: {e}",
                code=ScreenErrorCode.CAPTURE_FAILED,
            ) from e
        if not data:
            raise ScreenException(
                "Browser screenshot was empty",
                code=ScreenErrorCode.CAPTURE_FAILED,
            )
        return CapturedFrame(data, origin="browser")

"""Controlled clipboard abstraction.

Read / write / clear under policy.  Sensitive clipboard
data is never logged, never persisted unnecessarily, and a
cleanup policy runs on task completion.
"""

from __future__ import annotations

import threading
from typing import Any

from afnan_ai.computer.errors import ComputerError
from afnan_ai.redaction import redact_text


class ClipboardController:
    """Policy-gated clipboard for the agent."""

    def __init__(
        self,
        backend: Any,
        *,
        allow_read: bool = True,
        allow_write: bool = True,
        max_chars: int = 100_000,
    ) -> None:
        self.backend = backend
        self.allow_read = allow_read
        self.allow_write = allow_write
        self.max_chars = max_chars
        self._lock = threading.RLock()
        self._wrote_this_task = False

    def _available(self) -> bool:
        try:
            caps = self.backend.capabilities() or {}
        except Exception:
            caps = {}
        return bool(caps.get("clipboard", False))

    def read(self) -> str:
        """Read clipboard (content is the caller's responsibility)."""
        if not self.allow_read:
            raise ComputerError(
                "clipboard_denied",
                "refused: clipboard read is disabled by policy",
            )
        if not self._available():
            raise ComputerError(
                "unsupported_operation",
                "clipboard is not available on this backend",
            )
        with self._lock:
            try:
                text = self.backend.clipboard_read() or ""
            except Exception as exc:
                raise ComputerError(
                    "clipboard_failed",
                    f"clipboard read failed: {exc}",
                ) from exc
        return str(text)[: self.max_chars]

    def write(self, text: str) -> dict[str, Any]:
        if not self.allow_write:
            raise ComputerError(
                "clipboard_denied",
                "refused: clipboard write is disabled by policy",
            )
        if not self._available():
            raise ComputerError(
                "unsupported_operation",
                "clipboard is not available on this backend",
            )
        text = str(text or "")[: self.max_chars]
        with self._lock:
            try:
                self.backend.clipboard_write(text)
            except Exception as exc:
                raise ComputerError(
                    "clipboard_failed",
                    f"clipboard write failed: {exc}",
                ) from exc
            self._wrote_this_task = True
        # Log length only — never the content.
        return {"written": True, "chars": len(text)}

    def clear(self) -> dict[str, Any]:
        with self._lock:
            try:
                self.backend.clipboard_clear()
            except Exception:
                # Best-effort: overwrite with empty string.
                try:
                    self.backend.clipboard_write("")
                except Exception:
                    pass
            self._wrote_this_task = False
        return {"cleared": True}

    def cleanup_after_task(self) -> dict[str, Any]:
        """Policy: clear the clipboard when the task is done."""
        if self._wrote_this_task:
            return self.clear()
        return {"cleared": False, "reason": "nothing written"}

    def safe_preview(self, text: str, *, limit: int = 60) -> str:
        """Redacted preview for logs/events."""
        return redact_text(str(text or ""))[:limit]

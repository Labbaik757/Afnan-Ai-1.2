"""Download tracking for the BrowserController.

The :class:`DownloadManager` centralises everything the driver reports
about downloads — start, progress, completion, failure, filename, file
type and destination — and verifies finished files (existence, size and
a basic magic-bytes integrity check for common formats).

Safety rules:

* Downloads are only ever *saved* — never opened or executed.
* Executable / installer payloads are flagged ``unsafe`` and marked as
  requiring human approval; nothing runs them automatically.
* URLs pass through redaction before they leave this module, so query
  tokens never reach logs or LLM context.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..redaction import redact_text
from .base import BrowserErrorCode, BrowserException

__all__ = ["DownloadManager", "UNSAFE_EXTENSIONS"]

#: Extensions that are never auto-opened and always need approval.
UNSAFE_EXTENSIONS = frozenset({
    "exe", "msi", "dmg", "pkg", "bat", "cmd", "com", "scr",
    "sh", "bash", "ps1", "jar", "apk", "appimage", "deb", "rpm",
})

#: Magic-byte prefixes for a basic integrity check by extension.
_MAGIC = {
    "png": (b"\x89PNG",),
    "jpg": (b"\xff\xd8\xff",),
    "jpeg": (b"\xff\xd8\xff",),
    "gif": (b"GIF8",),
    "pdf": (b"%PDF",),
    "zip": (b"PK\x03\x04", b"PK\x05\x06"),
}


class DownloadManager:
    """Tracks browser downloads reported by the backend."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    # -- queries -----------------------------------------------------------

    def list_downloads(self) -> list[dict[str, Any]]:
        return [self._describe(raw) for raw in self._raw()]

    def get(self, download_id: str) -> dict[str, Any] | None:
        for record in self.list_downloads():
            if record["download_id"] == download_id:
                return record
        return None

    def latest(self) -> dict[str, Any] | None:
        records = self.list_downloads()
        return records[-1] if records else None

    # -- waiting -----------------------------------------------------------

    def wait_for(
        self,
        download_id: str | None = None,
        *,
        timeout_ms: int = 30000,
    ) -> dict[str, Any]:
        """Wait until a download completes or fails.

        Raises a structured ``timeout`` error when the download does
        not finish in time; an interrupted/failed download is returned
        with its real ``failed`` state, never faked as success.
        """
        if timeout_ms is None or timeout_ms <= 0:
            raise BrowserException(
                "timeout_ms must be a positive number of milliseconds.",
                code=BrowserErrorCode.TIMEOUT,
            )
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            record = (
                self.get(download_id)
                if download_id
                else self.latest()
            )
            if record is None:
                raise BrowserException(
                    "No download is being tracked yet.",
                    code=BrowserErrorCode.OPERATION_FAILED,
                )
            if record["state"] in ("completed", "failed"):
                return record
            if time.monotonic() >= deadline:
                raise BrowserException(
                    f"Download {record['download_id']} did not finish "
                    f"within {timeout_ms}ms (state: {record['state']}).",
                    code=BrowserErrorCode.TIMEOUT,
                    details={"download": record},
                )
            time.sleep(0.05)

    # -- internals ---------------------------------------------------------

    def _raw(self) -> list[dict[str, Any]]:
        try:
            records = self._backend.downloads()
        except Exception:
            return []
        return [dict(r) for r in (records or [])]

    def _describe(self, raw: dict[str, Any]) -> dict[str, Any]:
        path_value = raw.get("path") or ""
        path = Path(path_value) if path_value else None
        exists = bool(path and path.is_file())
        size = path.stat().st_size if exists and path else 0
        filename = raw.get("filename") or (
            path.name if path else ""
        )
        extension = (
            filename.rsplit(".", 1)[-1].lower()
            if "." in filename
            else ""
        )
        state = str(raw.get("state") or "started")
        record: dict[str, Any] = {
            "download_id": str(raw.get("id") or filename),
            "url": redact_text(str(raw.get("url") or "")),
            "filename": filename,
            "file_type": extension,
            "state": state,
            "destination": str(path) if path else None,
            "size_bytes": size,
            "failure": raw.get("failure"),
            "unsafe": extension in UNSAFE_EXTENSIONS,
            "requires_approval": extension in UNSAFE_EXTENSIONS,
            "integrity": "not_checked",
        }
        if state == "completed":
            record["integrity"] = self._verify(path, extension, size)
        elif state == "failed":
            record["integrity"] = "failed"
        return record

    @staticmethod
    def _verify(
        path: Path | None, extension: str, size: int
    ) -> str:
        """Basic integrity: exists, non-empty, magic bytes match."""
        if path is None or not path.is_file():
            return "missing_file"
        if size <= 0:
            return "empty_file"
        magics = _MAGIC.get(extension)
        if not magics:
            return "ok"
        try:
            with path.open("rb") as handle:
                head = handle.read(8)
        except OSError:
            return "unreadable"
        return "ok" if any(head.startswith(m) for m in magics) else "corrupt"

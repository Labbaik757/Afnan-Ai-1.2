"""Safe file-system operations for the Computer Use layer.

Runs through the existing Tool system like every other
capability.  Read operations (list, find downloads) are
free; destructive ones (move, rename, overwrite) go through
the same human-approval gate as sensitive desktop actions —
no approver means they do not run.  File *contents* are
never returned or logged here; results carry names, paths
and sizes only.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable

from afnan_ai.computer.errors import ComputerError
from afnan_ai.computer.policy import ComputerApprovalGate


def _downloads_candidates() -> list[Path]:
    home = Path.home()
    return [home / "Downloads", home / "Desktop"]


class FileService:
    def __init__(
        self,
        *,
        gate: ComputerApprovalGate | None = None,
        approver: Any = None,
        open_path: Callable[[str], Any] | None = None,
        downloads_dirs: list[str] | None = None,
    ):
        self.gate = gate or ComputerApprovalGate(approver)
        self.open_path = open_path
        self.downloads_dirs = [
            Path(d).expanduser()
            for d in (downloads_dirs or [])
        ] or _downloads_candidates()

    # -- helpers ------------------------------------------------------
    def _require_approval(self, action: str, summary: str) -> None:
        decision = self.gate.check(
            action, level="destructive", target_summary=summary
        )
        if decision.outcome not in ("approved", "not_required"):
            raise ComputerError(
                "approval_required"
                if decision.outcome == "no_approver"
                else "approval_denied",
                f"File operation {action} requires human "
                f"approval: {summary}",
                decision=decision.to_dict(),
            )

    @staticmethod
    def _entry(path: Path) -> dict[str, Any]:
        try:
            stat = path.stat()
            return {
                "name": path.name,
                "path": str(path),
                "is_dir": path.is_dir(),
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }
        except OSError as exc:
            raise ComputerError(
                "file_error", f"Cannot read {path}: {exc}"
            )

    # -- operations -----------------------------------------------------
    def list_dir(self, path: str) -> dict[str, Any]:
        folder = Path(path).expanduser()
        if not folder.is_dir():
            raise ComputerError(
                "file_error", f"Not a directory: {folder}"
            )
        entries = []
        for child in sorted(folder.iterdir())[:200]:
            try:
                entries.append(self._entry(child))
            except ComputerError:
                continue
        return {"path": str(folder), "entries": entries}

    def create_folder(self, path: str) -> dict[str, Any]:
        folder = Path(path).expanduser()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ComputerError(
                "file_error", f"Cannot create {folder}: {exc}"
            )
        return {"created": str(folder)}

    def copy_file(self, source: str, destination: str) -> dict[str, Any]:
        src = Path(source).expanduser()
        dst = Path(destination).expanduser()
        if not src.is_file():
            raise ComputerError(
                "file_error", f"Source file not found: {src}"
            )
        if dst.exists():
            self._require_approval(
                "file_copy_overwrite",
                f"Overwrite {dst} with {src}",
            )
        try:
            if dst.is_dir():
                dst = dst / src.name
            shutil.copy2(src, dst)
        except OSError as exc:
            raise ComputerError(
                "file_error", f"Copy failed: {exc}"
            )
        return {"copied_to": str(dst), "size": dst.stat().st_size}

    def move_file(self, source: str, destination: str) -> dict[str, Any]:
        src = Path(source).expanduser()
        dst = Path(destination).expanduser()
        if not src.exists():
            raise ComputerError(
                "file_error", f"Source not found: {src}"
            )
        self._require_approval(
            "file_move", f"Move {src} -> {dst}"
        )
        try:
            if dst.is_dir():
                dst = dst / src.name
            shutil.move(str(src), str(dst))
        except OSError as exc:
            raise ComputerError(
                "file_error", f"Move failed: {exc}"
            )
        return {"moved_to": str(dst)}

    def rename_file(self, path: str, new_name: str) -> dict[str, Any]:
        src = Path(path).expanduser()
        if not src.exists():
            raise ComputerError(
                "file_error", f"File not found: {src}"
            )
        if "/" in new_name or "\\" in new_name:
            raise ComputerError(
                "invalid_arguments",
                "new_name must be a plain file name",
            )
        dst = src.with_name(new_name)
        self._require_approval(
            "file_rename", f"Rename {src} -> {dst}"
        )
        try:
            src.rename(dst)
        except OSError as exc:
            raise ComputerError(
                "file_error", f"Rename failed: {exc}"
            )
        return {"renamed_to": str(dst)}

    def save_text(self, path: str, content: str) -> dict[str, Any]:
        target = Path(path).expanduser()
        if target.exists():
            self._require_approval(
                "file_overwrite", f"Overwrite {target}"
            )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ComputerError(
                "file_error", f"Save failed: {exc}"
            )
        return {
            "saved_to": str(target),
            "bytes": len(content.encode("utf-8")),
        }

    def open_file(self, path: str) -> dict[str, Any]:
        target = Path(path).expanduser()
        if not target.exists():
            raise ComputerError(
                "file_error", f"File not found: {target}"
            )
        if self.open_path is None:
            raise ComputerError(
                "unsupported_operation",
                "No platform file opener is configured",
            )
        self.open_path(str(target))
        return {"opened": str(target)}

    def find_downloads(
        self, *, max_age_s: float = 3600.0, limit: int = 20
    ) -> dict[str, Any]:
        cutoff = time.time() - max(0.0, float(max_age_s))
        found: list[dict[str, Any]] = []
        for folder in self.downloads_dirs:
            if not folder.is_dir():
                continue
            for child in folder.iterdir():
                try:
                    if child.is_file() and child.stat().st_mtime >= cutoff:
                        found.append(self._entry(child))
                except OSError:
                    continue
        found.sort(key=lambda e: e["modified"], reverse=True)
        return {"downloads": found[: int(limit)]}

"""WorkspaceFileManager — scoped filesystem isolation.

Every operation is confined to a single workspace root.
Defends against ``..`` traversal, symlink escape,
absolute-path escape, cross-workspace access and access
to sensitive system directories.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class WorkspacePathError(Exception):
    """Raised when a path violates workspace isolation."""


@dataclass(frozen=True)
class FileOpResult:
    ok: bool
    path: str = ""
    detail: str = ""
    bytes_written: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "path": self.path,
            "detail": self.detail,
            "bytes_written": self.bytes_written,
        }


# Credential-shaped filenames that must never be written
# into a workspace by file operations.
_CREDENTIAL_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "secrets.json",
    ".netrc",
    "_netrc",
    "id_rsa",
    "id_ed25519",
    ".aws",
    ".ssh",
}

# System roots that are always off-limits.
_SENSITIVE_PREFIXES = (
    "/etc",
    "/proc",
    "/sys",
    "/dev",
    "/boot",
    "/root",
    "/var/run",
    "C:\\Windows",
    "C:\\Program Files",
)


class WorkspaceFileManager:
    """Scoped file operations for one workspace."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        allowed_subdirs: tuple[str, ...] | None = None,
        sensitive_paths: tuple[str, ...] = (),
        max_file_mb: float = 256.0,
    ) -> None:
        self.root = Path(workspace_root).resolve()
        self.allowed_subdirs = allowed_subdirs
        self.sensitive_paths = tuple(sensitive_paths)
        self.max_file_bytes = int(max_file_mb * 1024 * 1024)

    # -- path resolution ---------------------------------------------

    def resolve(self, rel: str | Path) -> Path:
        """Resolve *rel* inside the workspace root.

        Raises :class:`WorkspacePathError` on traversal,
        absolute-path escape, symlink escape or sensitive
        path access.
        """
        raw = str(rel)
        candidate = (self.root / raw).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise WorkspacePathError(
                f"path escapes workspace root: {raw!r}"
            )
        # Symlink escape: re-check after resolve already done
        # above; also reject if any parent is a symlink that
        # points outside the root.
        for parent in [candidate, *candidate.parents]:
            if parent == self.root:
                break
            if parent.is_symlink():
                raise WorkspacePathError(
                    f"symlink escape rejected: {raw!r}"
                )
        lowered = os.path.normcase(str(candidate)).lower()
        for prefix in _SENSITIVE_PREFIXES:
            if lowered == prefix.lower() or lowered.startswith(
                prefix.lower().rstrip(os.sep) + os.sep
            ):
                raise WorkspacePathError(
                    f"sensitive system path denied: {raw!r}"
                )
        for sp in self.sensitive_paths:
            spath = (self.root / sp).resolve()
            try:
                candidate.relative_to(spath)
                raise WorkspacePathError(
                    f"policy-denied path: {raw!r}"
                )
            except ValueError:
                pass
        return candidate

    def _guard_credential_name(self, path: Path) -> None:
        name = path.name.lower()
        if name in _CREDENTIAL_NAMES or name.endswith(
            (".pem", ".key")
        ):
            raise WorkspacePathError(
                f"credential-shaped file denied: {path.name!r}"
            )

    def _guard_size(self, data: bytes) -> None:
        if len(data) > self.max_file_bytes:
            raise WorkspacePathError(
                f"file exceeds {self.max_file_bytes} bytes"
            )

    # -- operations ----------------------------------------------------

    def create(
        self, rel: str | Path, data: bytes | str = b""
    ) -> FileOpResult:
        path = self.resolve(rel)
        self._guard_credential_name(path)
        raw = (
            data.encode("utf-8")
            if isinstance(data, str)
            else bytes(data)
        )
        self._guard_size(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise WorkspacePathError(
                f"already exists: {rel!r}"
            )
        path.write_bytes(raw)
        return FileOpResult(
            True, str(path), "created",
            bytes_written=len(raw),
        )

    def write(
        self, rel: str | Path, data: bytes | str
    ) -> FileOpResult:
        path = self.resolve(rel)
        self._guard_credential_name(path)
        raw = (
            data.encode("utf-8")
            if isinstance(data, str)
            else bytes(data)
        )
        self._guard_size(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return FileOpResult(
            True, str(path), "written",
            bytes_written=len(raw),
        )

    def read(
        self, rel: str | Path, max_bytes: int = 4 * 1024 * 1024
    ) -> bytes:
        path = self.resolve(rel)
        if not path.is_file():
            raise WorkspacePathError(
                f"not a file: {rel!r}"
            )
        with open(path, "rb") as fh:
            return fh.read(max_bytes + 1)[:max_bytes]

    def move(
        self, src: str | Path, dst: str | Path
    ) -> FileOpResult:
        spath = self.resolve(src)
        dpath = self.resolve(dst)
        self._guard_credential_name(dpath)
        dpath.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(spath), str(dpath))
        return FileOpResult(True, str(dpath), "moved")

    def copy(
        self, src: str | Path, dst: str | Path
    ) -> FileOpResult:
        spath = self.resolve(src)
        dpath = self.resolve(dst)
        self._guard_credential_name(dpath)
        dpath.parent.mkdir(parents=True, exist_ok=True)
        if spath.is_dir():
            shutil.copytree(
                str(spath), str(dpath), dirs_exist_ok=True
            )
        else:
            shutil.copy2(str(spath), str(dpath))
        return FileOpResult(True, str(dpath), "copied")

    def delete(self, rel: str | Path) -> FileOpResult:
        path = self.resolve(rel)
        if path == self.root:
            raise WorkspacePathError(
                "refusing to delete workspace root"
            )
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
        else:
            raise WorkspacePathError(
                f"not found: {rel!r}"
            )
        return FileOpResult(True, str(path), "deleted")

    def list(
        self, rel: str | Path = ".", recursive: bool = False
    ) -> list[dict[str, Any]]:
        path = self.resolve(rel)
        if not path.is_dir():
            raise WorkspacePathError(
                f"not a directory: {rel!r}"
            )
        entries: list[dict[str, Any]] = []
        if recursive:
            for dirpath, dirnames, filenames in os.walk(path):
                for name in dirnames + filenames:
                    full = Path(dirpath) / name
                    entries.append(self._stat(full))
        else:
            for child in sorted(path.iterdir()):
                entries.append(self._stat(child))
        return entries

    def _stat(self, path: Path) -> dict[str, Any]:
        try:
            st = path.stat()
            rel = str(path.relative_to(self.root))
        except OSError:
            rel = path.name
            st = None
        return {
            "name": path.name,
            "rel": rel,
            "is_dir": path.is_dir(),
            "is_symlink": path.is_symlink(),
            "size": st.st_size if st else 0,
            "mtime": st.st_mtime if st else 0.0,
        }

    def search(
        self,
        pattern: str,
        rel: str | Path = ".",
        max_results: int = 100,
    ) -> list[str]:
        """Filename substring search inside the workspace."""
        path = self.resolve(rel)
        needle = pattern.lower()
        hits: list[str] = []
        for dirpath, _, filenames in os.walk(path):
            for fn in filenames:
                if needle in fn.lower():
                    hits.append(
                        str(
                            (Path(dirpath) / fn).relative_to(
                                self.root
                            )
                        )
                    )
                    if len(hits) >= max_results:
                        return hits
        return hits

    def archive(
        self, rel: str | Path, dest: str | Path,
        fmt: str = "zip",
    ) -> FileOpResult:
        src = self.resolve(rel)
        out = self.resolve(dest)
        self._guard_credential_name(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "zip":
            with zipfile.ZipFile(
                out, "w", zipfile.ZIP_DEFLATED
            ) as zf:
                if src.is_dir():
                    for dirpath, _, filenames in os.walk(src):
                        for fn in filenames:
                            full = Path(dirpath) / fn
                            zf.write(
                                full,
                                full.relative_to(src.parent),
                            )
                else:
                    zf.write(src, src.name)
        elif fmt == "tar":
            with tarfile.open(out, "w:gz") as tf:
                tf.add(src, arcname=src.name)
        else:
            raise WorkspacePathError(
                f"unsupported archive format: {fmt!r}"
            )
        return FileOpResult(True, str(out), f"archived:{fmt}")

    def extract(
        self, rel: str | Path, dest: str | Path
    ) -> FileOpResult:
        """Extract an archive safely (no zip-slip)."""
        src = self.resolve(rel)
        outdir = self.resolve(dest)
        outdir.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(src):
            with zipfile.ZipFile(src) as zf:
                for member in zf.namelist():
                    target = (outdir / member).resolve()
                    try:
                        target.relative_to(outdir)
                    except ValueError:
                        raise WorkspacePathError(
                            f"zip-slip rejected: {member!r}"
                        )
                    if member.endswith("/"):
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(
                            parents=True, exist_ok=True
                        )
                        target.write_bytes(
                            zf.read(member)
                        )
        elif tarfile.is_tarfile(src):
            with tarfile.open(src) as tf:
                for member in tf.getmembers():
                    target = (outdir / member.name).resolve()
                    try:
                        target.relative_to(outdir)
                    except ValueError:
                        raise WorkspacePathError(
                            f"tar-slip rejected: {member.name!r}"
                        )
                tf.extractall(outdir, filter="data")
        else:
            raise WorkspacePathError(
                f"not an archive: {rel!r}"
            )
        return FileOpResult(True, str(outdir), "extracted")

    def metadata(self) -> dict[str, Any]:
        """Redacted files metadata for checkpoints."""
        files: dict[str, Any] = {}
        for entry in self.list(".", recursive=True):
            if not entry["is_dir"]:
                files[entry["rel"]] = {
                    "size": entry["size"],
                    "mtime": entry["mtime"],
                }
        return {"file_count": len(files), "files": files}

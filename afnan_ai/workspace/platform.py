"""OS-specific workspace adapters.

Core WorkspaceManager stays platform-independent; each
adapter only knows how to build paths, enforce platform
process/resource primitives and pick safe temp locations.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from abc import ABC, abstractmethod
from pathlib import Path


class BaseWorkspaceAdapter(ABC):
    """Platform-specific primitives for workspaces."""

    name: str = "base"

    @abstractmethod
    def default_base_dir(self) -> Path:
        """Default root under which workspaces live."""

    def prepare_root(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)

    def secure_temp_dir(self, workspace_tmp: Path) -> Path:
        """Return (creating) the workspace temp dir."""
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        return workspace_tmp

    def terminate_process_tree(self, pid: int) -> bool:
        """Best-effort termination of a process tree."""
        try:
            if os.name == "nt":
                import subprocess

                subprocess.run(
                    [
                        "taskkill",
                        "/PID",
                        str(pid),
                        "/T",
                        "/F",
                    ],
                    capture_output=True,
                    timeout=10,
                )
                return True
            os.kill(pid, 15)
            return True
        except Exception:
            return False

    def disk_usage_mb(self, path: Path) -> float:
        total = 0
        try:
            for dirpath, _, filenames in os.walk(path):
                for fn in filenames:
                    try:
                        total += os.path.getsize(
                            os.path.join(dirpath, fn)
                        )
                    except OSError:
                        pass
        except OSError:
            pass
        return total / (1024 * 1024)

    def free_disk_mb(self, path: Path) -> float:
        try:
            usage = shutil.disk_usage(str(path))
            return usage.free / (1024 * 1024)
        except OSError:
            return 0.0


class LinuxWorkspaceAdapter(BaseWorkspaceAdapter):
    name = "linux"

    def default_base_dir(self) -> Path:
        return (
            Path(
                os.environ.get(
                    "AFNAN_WORKSPACE_DIR", ""
                )
                or Path.home() / ".afnan-ai" / "workspaces"
            ).expanduser()
        )


class MacOSWorkspaceAdapter(BaseWorkspaceAdapter):
    name = "macos"

    def default_base_dir(self) -> Path:
        return (
            Path(
                os.environ.get(
                    "AFNAN_WORKSPACE_DIR", ""
                )
                or Path.home() / ".afnan-ai" / "workspaces"
            ).expanduser()
        )


class WindowsWorkspaceAdapter(BaseWorkspaceAdapter):
    name = "windows"

    def default_base_dir(self) -> Path:
        base = os.environ.get("AFNAN_WORKSPACE_DIR", "")
        if base:
            return Path(base).expanduser()
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            return Path(local) / "AfnanAI" / "workspaces"
        return Path.home() / ".afnan-ai" / "workspaces"

    def terminate_process_tree(self, pid: int) -> bool:
        return super().terminate_process_tree(pid)


def get_adapter(
    platform_name: str | None = None,
) -> BaseWorkspaceAdapter:
    """Return the adapter for *platform_name* (or current OS)."""
    name = (platform_name or sys.platform).lower()
    if name.startswith("win"):
        return WindowsWorkspaceAdapter()
    if name == "darwin":
        return MacOSWorkspaceAdapter()
    if name.startswith("linux"):
        return LinuxWorkspaceAdapter()
    # Unknown POSIX-ish platform → linux behaviour.
    return LinuxWorkspaceAdapter()


def current_platform() -> str:
    sysname = platform.system().lower()
    if sysname == "windows":
        return "windows"
    if sysname == "darwin":
        return "macos"
    return "linux"

"""Base platform adapter — the contract every OS adapter implements.

Core agent code must only use this interface.  It must never import
``os.startfile``, call ``open -a``, ``mdfind``, ``xdg-open`` or a
PowerShell speech command directly; those belong in the concrete
Windows / macOS / Linux adapters.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path


KNOWN_FOLDERS = {
    "downloads": "Downloads",
    "download": "Downloads",
    "desktop": "Desktop",
    "documents": "Documents",
    "document": "Documents",
    "pictures": "Pictures",
    "picture": "Pictures",
    "music": "Music",
    "videos": "Videos",
    "video": "Videos",
}


class PlatformAdapter(ABC):
    """Abstract operating-system adapter."""

    #: Short platform key: "windows", "macos" or "linux"
    name: str = "base"

    # -- speech ------------------------------------------------------
    @abstractmethod
    def speak_system(self, text: str) -> None:
        """Speak *text* with the OS-native speech engine (fallback TTS)."""

    # -- opening files / folders -------------------------------------
    @abstractmethod
    def open_path(self, path: str) -> None:
        """Open a file or folder with the default OS handler."""

    # -- launching applications ---------------------------------------
    @abstractmethod
    def launch_app(self, app_key: str) -> bool:
        """Launch a known application.

        ``app_key`` is one of: ``vscode``, ``chrome``, ``edge``,
        ``whatsapp``, ``safari``.  Returns True when a launch was
        attempted/succeeded, False when the app is not available on
        this platform (e.g. Safari on Windows).
        """

    def open_app_window(self, url: str) -> None:
        """Open *url* in a standalone app-like window.

        Default implementation opens the URL in the default
        browser.  Platforms override this when they can do better
        (e.g. Chrome's ``--app`` mode: no tabs or address bar, so
        the page feels like its own desktop app).
        """
        import webbrowser

        webbrowser.open(url)

    # -- finding folders ----------------------------------------------
    def known_folder_path(self, foldername: str) -> Path | None:
        """Return a well-known user folder (Downloads, Desktop, ...)."""
        key = foldername.strip().lower()
        folder = KNOWN_FOLDERS.get(key)
        if not folder:
            return None
        candidate = Path.home() / folder
        return candidate if candidate.is_dir() else None

    def find_folder_in_home(self, foldername: str, max_depth: int = 3) -> str | None:
        """Shallow home-directory search — shared by Windows/Linux."""
        target = foldername.strip().lower()
        known = self.known_folder_path(foldername)
        if known is not None:
            return str(known)

        home = Path.home()
        for root, dirs, _files in os.walk(home):
            try:
                depth = len(Path(root).relative_to(home).parts)
            except ValueError:
                depth = max_depth + 1
            if depth > max_depth:
                dirs[:] = []
                continue
            for dirname in dirs:
                if dirname.lower() == target:
                    return os.path.join(root, dirname)
        return None

    @abstractmethod
    def find_folder(self, foldername: str) -> str | None:
        """Find a folder by name and return its path, or None."""

    # -- capabilities --------------------------------------------------
    def supports_app(self, app_key: str) -> bool:
        """Whether *app_key* can be launched natively on this OS."""
        if app_key == "safari":
            return self.name == "macos"
        return app_key in {"vscode", "chrome", "edge", "whatsapp"}

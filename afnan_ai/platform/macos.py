"""macOS platform adapter.

Isolated macOS-only behaviour: ``say``, ``open`` / ``open -a``
and Spotlight (``mdfind``) folder search.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from afnan_ai.platform.base import PlatformAdapter


class MacOSAdapter(PlatformAdapter):
    name = "macos"

    def speak_system(self, text: str) -> None:
        subprocess.run(["say", text], check=False)

    def open_path(self, path: str) -> None:
        subprocess.run(["open", path], check=False)

    def launch_app(self, app_key: str) -> bool:
        macos_apps = {
            "vscode": "Visual Studio Code",
            "chrome": "Google Chrome",
            "edge": "Microsoft Edge",
            "whatsapp": "WhatsApp",
            "safari": "Safari",
        }
        app_name = macos_apps.get(app_key)
        if not app_name:
            return False
        subprocess.run(["open", "-a", app_name], check=False)
        return True

    def find_folder(self, foldername: str) -> str | None:
        known = self.known_folder_path(foldername)
        if known is not None:
            return str(known)

        target = foldername.strip().lower()
        try:
            result = subprocess.run(
                ["mdfind", "-name", foldername],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.strip().split("\n"):
                if line and os.path.isdir(line) and Path(line).name.lower() == target:
                    return line
        except Exception:
            pass
        return None

"""Linux platform adapter.

Isolated Linux-only behaviour: ``spd-say`` / ``espeak`` speech,
``xdg-open`` and executable lookup for browsers/editors.
"""

from __future__ import annotations

import shutil
import subprocess

from afnan_ai.platform.base import PlatformAdapter


class LinuxAdapter(PlatformAdapter):
    name = "linux"

    def speak_system(self, text: str) -> None:
        if shutil.which("spd-say"):
            subprocess.run(["spd-say", text], check=False)
        elif shutil.which("espeak"):
            subprocess.run(["espeak", text], check=False)
        else:
            print("(no Linux TTS engine found — install espeak or pyttsx3)")

    def open_path(self, path: str) -> None:
        subprocess.run(["xdg-open", path], check=False)

    def launch_app(self, app_key: str) -> bool:
        linux_apps = {
            "vscode": ["code"],
            "chrome": ["google-chrome", "chromium-browser", "chromium"],
            "edge": ["microsoft-edge"],
            "whatsapp": [],  # the agent falls back to WhatsApp Web
            "safari": [],  # Safari is not available on Linux
        }
        for candidate in linux_apps.get(app_key, []):
            if shutil.which(candidate):
                subprocess.run([candidate], check=False)
                return True
        return False

    def find_folder(self, foldername: str) -> str | None:
        return self.find_folder_in_home(foldername)

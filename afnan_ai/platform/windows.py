"""Windows platform adapter.

Isolated Windows-only behaviour: PowerShell SAPI speech,
``os.startfile`` and ``cmd /c start`` app launching.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from afnan_ai.platform.base import PlatformAdapter


class WindowsAdapter(PlatformAdapter):
    name = "windows"

    def speak_system(self, text: str) -> None:
        safe_text = text.replace("'", "''")
        ps_command = (
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe_text}')"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_command],
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def open_path(self, path: str) -> None:
        os.startfile(path)  # type: ignore[attr-defined]  # Windows only

    def launch_app(self, app_key: str) -> bool:
        windows_apps = {
            "vscode": [["code"], ["cmd", "/c", "start", "", "code"]],
            "chrome": [["cmd", "/c", "start", "", "chrome"]],
            "edge": [["cmd", "/c", "start", "", "msedge"]],
            "whatsapp": [
                ["cmd", "/c", "start", "", "whatsapp:"],
                ["cmd", "/c", "start", "", "https://web.whatsapp.com"],
            ],
            "safari": [],  # Safari is not available on Windows
        }
        for cmd in windows_apps.get(app_key, []):
            try:
                if cmd and cmd[0] == "code" and not shutil.which("code"):
                    continue
                subprocess.run(cmd, check=False)
                return True
            except Exception:
                continue
        return False

    def find_folder(self, foldername: str) -> str | None:
        return self.find_folder_in_home(foldername)

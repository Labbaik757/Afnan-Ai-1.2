"""CommandComputerBackend — desktop control through each
OS's own command-line tools (no third-party dependency).

- Linux: ``xdotool`` / ``wmctrl`` for input and windows,
  ``.desktop`` files for application discovery.
- macOS: ``osascript`` (System Events) for windows and
  keyboard, ``cliclick`` when present for the mouse.
- Windows: PowerShell with user32/System.Windows.Forms for
  input, cursor, screen and window management.

Missing tools surface as structured ``backend_unavailable``
errors — never crashes, never fake success.  The command
runner is injectable so behaviour is fully testable on any
host.  A production deployment may replace this backend with
a pyautogui/native-accessibility implementation behind the
same :class:`ComputerBackend` interface.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from afnan_ai.computer.backend import ComputerBackend
from afnan_ai.computer.errors import ComputerError
from afnan_ai.computer.models import AppInfo, WindowInfo

Runner = Callable[[list[str]], str]
Spawner = Callable[[list[str]], Any]


def _default_runner(argv: list[str]) -> str:
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=15,
            check=False,
        )
    except FileNotFoundError:
        raise ComputerError(
            "backend_unavailable",
            f"Required desktop tool not found: {argv[0]}",
        )
    except subprocess.TimeoutExpired:
        raise ComputerError(
            "timeout", f"Desktop command timed out: {argv[0]}"
        )
    if proc.returncode != 0:
        raise ComputerError(
            "backend_unavailable",
            f"Desktop command failed ({argv[0]}): "
            f"{(proc.stderr or proc.stdout or '').strip()[:200]}",
        )
    return proc.stdout


def _default_spawner(argv: list[str]) -> Any:
    try:
        return subprocess.Popen(
            argv, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        raise ComputerError(
            "backend_unavailable",
            f"Required desktop tool not found: {argv[0]}",
        )


class CommandComputerBackend(ComputerBackend):
    name = "command"

    def __init__(
        self,
        system: str | None = None,
        *,
        runner: Runner | None = None,
        spawner: Spawner | None = None,
    ):
        import platform as _platform

        self.system = (system or _platform.system()).lower()
        self._run = runner or _default_runner
        self._spawn = spawner or _default_spawner

    # -- desktop state -------------------------------------------------
    def screen_size(self) -> tuple[int, int]:
        if self.system == "linux":
            out = self._run(["xdotool", "getdisplaygeometry"])
            parts = out.split()
            return int(parts[0]), int(parts[1])
        if self.system == "darwin":
            out = self._run([
                "osascript", "-e",
                'tell application "Finder" to get bounds of '
                "window of desktop",
            ])
            parts = [p.strip() for p in out.split(",")]
            return int(parts[2]), int(parts[3])
        out = self._run([
            "powershell", "-NoProfile", "-Command",
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$s=[System.Windows.Forms.Screen]::PrimaryScreen."
            "Bounds; \"$($s.Width) $($s.Height)\"",
        ])
        parts = out.split()
        return int(parts[0]), int(parts[1])

    def cursor_position(self) -> tuple[int, int]:
        if self.system == "linux":
            out = self._run(
                ["xdotool", "getmouselocation", "--shell"]
            )
            values = dict(
                line.split("=", 1)
                for line in out.splitlines() if "=" in line
            )
            return int(values["X"]), int(values["Y"])
        if self.system == "darwin":
            out = self._run(["cliclick", "p"])
            parts = out.replace(",", " ").split()
            return int(parts[0]), int(parts[1])
        out = self._run([
            "powershell", "-NoProfile", "-Command",
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$p=[System.Windows.Forms.Cursor]::Position; "
            "\"$($p.X) $($p.Y)\"",
        ])
        parts = out.split()
        return int(parts[0]), int(parts[1])

    # -- mouse / keyboard ------------------------------------------------
    def mouse_move(self, x: int, y: int) -> None:
        if self.system == "linux":
            self._run(["xdotool", "mousemove", str(x), str(y)])
        elif self.system == "darwin":
            self._run(["cliclick", f"m:{x},{y}"])
        else:
            self._run([
                "powershell", "-NoProfile", "-Command",
                "Add-Type -AssemblyName System.Windows.Forms; "
                f"[System.Windows.Forms.Cursor]::Position = "
                f"New-Object System.Drawing.Point({x}, {y})",
            ])

    def mouse_click(
        self, x: int, y: int, *, button: str = "left",
        click_count: int = 1,
    ) -> None:
        if self.system == "linux":
            code = {"left": "1", "middle": "2", "right": "3"}[
                button
            ]
            argv = [
                "xdotool", "mousemove", str(x), str(y), "click",
            ]
            if click_count > 1:
                argv += ["--repeat", str(click_count)]
            argv.append(code)
            self._run(argv)
        elif self.system == "darwin":
            verb = {
                "left": "c", "right": "rc",
            }.get(button, "c")
            argv = ["cliclick", f"{verb}:{x},{y}"]
            if click_count > 1 and button == "left":
                argv = ["cliclick", f"dc:{x},{y}"]
            self._run(argv)
        else:
            flag = {
                "left": "0x0002,0x0004",
                "right": "0x0008,0x0010",
                "middle": "0x0020,0x0040",
            }[button]
            down, up = flag.split(",")
            calls = "; ".join(
                f"[M]::mouse_event({f},0,0,0,[UIntPtr]::Zero)"
                for _ in range(click_count) for f in (down, up)
            )
            self.mouse_move(x, y)
            self._run([
                "powershell", "-NoProfile", "-Command",
                "Add-Type -AssemblyName System.Windows.Forms; "
                "Add-Type -Name M -Namespace W -MemberDefinition "
                "'[DllImport(\"user32.dll\")] public static "
                "extern void mouse_event(uint f,uint x,uint y,"
                "uint d,UIntPtr e);'; " + calls,
            ])

    def mouse_drag(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if self.system == "linux":
            self._run([
                "xdotool", "mousemove", str(x1), str(y1),
                "mousedown", "1", "mousemove", str(x2), str(y2),
                "mouseup", "1",
            ])
        elif self.system == "darwin":
            self._run([
                "cliclick", f"dd:{x1},{y1}", f"dm:{x2},{y2}",
                f"du:{x2},{y2}",
            ])
        else:
            self.mouse_move(x1, y1)
            self._win_mouse_event("0x0002")
            self.mouse_move(x2, y2)
            self._win_mouse_event("0x0004")

    def _win_mouse_event(self, flag: str) -> None:
        self._run([
            "powershell", "-NoProfile", "-Command",
            "Add-Type -Name M -Namespace W -MemberDefinition "
            "'[DllImport(\"user32.dll\")] public static extern "
            "void mouse_event(uint f,uint x,uint y,uint d,"
            "UIntPtr e);'; "
            f"[M]::mouse_event({flag},0,0,0,[UIntPtr]::Zero)",
        ])

    def scroll(self, dx: int, dy: int) -> None:
        if self.system == "linux":
            argv = ["xdotool"]
            if dy:
                argv += [
                    "click", "--repeat", str(abs(dy)),
                    "4" if dy > 0 else "5",
                ]
            if dx:
                argv += [
                    "click", "--repeat", str(abs(dx)),
                    "7" if dx > 0 else "6",
                ]
            if len(argv) > 1:
                self._run(argv)
        elif self.system == "darwin":
            raise ComputerError(
                "unsupported_operation",
                "scroll needs a native backend on macOS",
            )
        else:
            self._run([
                "powershell", "-NoProfile", "-Command",
                "Add-Type -Name M -Namespace W -MemberDefinition "
                "'[DllImport(\"user32.dll\")] public static "
                "extern void mouse_event(uint f,uint x,uint y,"
                "int d,UIntPtr e);'; "
                f"[M]::mouse_event(0x0800,0,0,{dy * 120},"
                "[UIntPtr]::Zero)",
            ])

    def type_text(self, text: str) -> None:
        if self.system == "linux":
            self._run(["xdotool", "type", "--", text])
        elif self.system == "darwin":
            escaped = text.replace("\\", "\\\\").replace(
                '"', '\\"'
            )
            self._run([
                "osascript", "-e",
                f'tell application "System Events" to '
                f'keystroke "{escaped}"',
            ])
        else:
            escaped = text.replace("'", "''")
            self._run([
                "powershell", "-NoProfile", "-Command",
                "Add-Type -AssemblyName System.Windows.Forms; "
                f"[System.Windows.Forms.SendKeys]::SendWait("
                f"'{escaped}')",
            ])

    def key_press(self, key: str) -> None:
        if self.system == "linux":
            self._run(["xdotool", "key", key])
        elif self.system == "darwin":
            self._run([
                "osascript", "-e",
                f'tell application "System Events" to '
                f'keystroke "{key}"',
            ])
        else:
            self._run([
                "powershell", "-NoProfile", "-Command",
                "Add-Type -AssemblyName System.Windows.Forms; "
                f"[System.Windows.Forms.SendKeys]::SendWait("
                f"'{{{key.upper()}}}')",
            ])

    def hotkey(self, keys: list[str]) -> None:
        combo = "+".join(keys)
        if self.system == "linux":
            self._run(["xdotool", "key", combo.lower()])
        elif self.system == "darwin":
            mods = {
                "cmd": "command down", "command": "command down",
                "ctrl": "control down", "control": "control down",
                "alt": "option down", "option": "option down",
                "shift": "shift down",
            }
            using = ", ".join(
                mods[k.lower()] for k in keys[:-1]
                if k.lower() in mods
            )
            script = (
                f'tell application "System Events" to keystroke '
                f'"{keys[-1]}"'
            )
            if using:
                script += f" using {{{using}}}"
            self._run(["osascript", "-e", script])
        else:
            joined = "".join(
                {"ctrl": "^", "alt": "%", "shift": "+"}.get(
                    k.lower(), ""
                )
                for k in keys[:-1]
            )
            self._run([
                "powershell", "-NoProfile", "-Command",
                "Add-Type -AssemblyName System.Windows.Forms; "
                "[System.Windows.Forms.SendKeys]::SendWait("
                f"'{joined}{{{keys[-1].upper()}}}')",
            ])

    # -- windows ------------------------------------------------------------
    def list_windows(self) -> list[WindowInfo]:
        if self.system == "linux":
            out = self._run(["wmctrl", "-l"])
            windows = []
            for line in out.splitlines():
                parts = line.split(None, 3)
                if len(parts) < 4:
                    continue
                windows.append(WindowInfo(
                    window_id=parts[0], title=parts[3],
                ))
            try:
                active_int = int(
                    self._run(
                        ["xdotool", "getactivewindow"]
                    ).strip()
                )
                for window in windows:
                    try:
                        if int(window.window_id, 16) == active_int:
                            window.active = True
                    except ValueError:
                        continue
            except (ComputerError, ValueError):
                pass
            return windows
        if self.system == "darwin":
            script = (
                'tell application "System Events" to repeat with '
                "p in (every process whose visible is true)\n"
                "repeat with w in (every window of p)\n"
                'log ((name of p) & "|" & (name of w))\n'
                "end repeat\nend repeat"
            )
            out = self._run(["osascript", "-e", script])
            windows = []
            for idx, line in enumerate(out.splitlines()):
                if "|" not in line:
                    continue
                app, title = line.split("|", 1)
                windows.append(WindowInfo(
                    window_id=f"win_{idx}", title=title, app=app,
                    active=idx == 0,
                ))
            return windows
        out = self._run([
            "powershell", "-NoProfile", "-Command",
            "Get-Process | Where-Object {$_.MainWindowTitle} | "
            "ForEach-Object { \"$($_.Id)|$($_.ProcessName)|"
            "$($_.MainWindowTitle)\" }",
        ])
        windows = []
        for line in out.splitlines():
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            windows.append(WindowInfo(
                window_id=parts[0], app=parts[1], title=parts[2],
            ))
        if windows:
            windows[0].active = True
        return windows

    def focus_window(self, window_id: str) -> None:
        if self.system == "linux":
            self._run(["xdotool", "windowactivate", window_id])
        elif self.system == "darwin":
            self._run([
                "osascript", "-e",
                f'tell application "System Events" to set '
                f"frontmost of (first process whose unix id is "
                f"{window_id}) to true",
            ])
        else:
            self._run([
                "powershell", "-NoProfile", "-Command",
                "Add-Type -Name W -Namespace U -MemberDefinition "
                "'[DllImport(\"user32.dll\")] public static "
                "extern bool SetForegroundWindow(IntPtr h);'; "
                f"$p=Get-Process -Id {window_id}; "
                "[U.W]::SetForegroundWindow($p.MainWindowHandle)",
            ])

    def minimize_window(self, window_id: str) -> None:
        if self.system == "linux":
            self._run(["xdotool", "windowminimize", window_id])
        else:
            raise ComputerError(
                "unsupported_operation",
                f"minimize_window not scripted on {self.system}",
            )

    def maximize_window(self, window_id: str) -> None:
        if self.system == "linux":
            self._run([
                "wmctrl", "-i", "-r", window_id, "-b",
                "add,maximized_vert,maximized_horz",
            ])
        else:
            raise ComputerError(
                "unsupported_operation",
                f"maximize_window not scripted on {self.system}",
            )

    def restore_window(self, window_id: str) -> None:
        if self.system == "linux":
            self._run([
                "wmctrl", "-i", "-r", window_id, "-b",
                "remove,maximized_vert,maximized_horz,hidden",
            ])
        else:
            raise ComputerError(
                "unsupported_operation",
                f"restore_window not scripted on {self.system}",
            )

    def close_window(self, window_id: str) -> None:
        if self.system == "linux":
            self._run(["xdotool", "windowclose", window_id])
        elif self.system == "darwin":
            raise ComputerError(
                "unsupported_operation",
                "close_window needs a native backend on macOS",
            )
        else:
            self._run([
                "powershell", "-NoProfile", "-Command",
                f"(Get-Process -Id {window_id}).CloseMainWindow()",
            ])

    # -- applications ---------------------------------------------------------
    def list_applications(self) -> list[AppInfo]:
        apps: list[AppInfo] = []
        if self.system == "linux":
            roots = [
                Path("/usr/share/applications"),
                Path.home() / ".local/share/applications",
            ]
            seen: set[str] = set()
            for root in roots:
                if not root.is_dir():
                    continue
                for entry in sorted(root.glob("*.desktop")):
                    try:
                        text = entry.read_text(
                            encoding="utf-8", errors="ignore"
                        )
                    except OSError:
                        continue
                    name = ""
                    exe = ""
                    for line in text.splitlines():
                        if line.startswith("Name=") and not name:
                            name = line[5:].strip()
                        if line.startswith("Exec=") and not exe:
                            exe = line[5:].split()[0]
                    if name and name not in seen:
                        seen.add(name)
                        apps.append(AppInfo(name=name, executable=exe))
            return apps
        if self.system == "darwin":
            root = Path("/Applications")
            if root.is_dir():
                for entry in sorted(root.iterdir()):
                    if entry.suffix == ".app":
                        apps.append(AppInfo(
                            name=entry.stem, executable=str(entry)
                        ))
            return apps
        root = (
            Path.home() / "AppData/Roaming/Microsoft/Windows"
            "/Start Menu/Programs"
        )
        if root.is_dir():
            for entry in sorted(root.rglob("*.lnk")):
                apps.append(AppInfo(
                    name=entry.stem, executable=str(entry)
                ))
        return apps

    def launch_application(self, name_or_path: str) -> AppInfo:
        target = str(name_or_path)
        if self.system == "linux":
            self._spawn(["gtk-launch", target])
        elif self.system == "darwin":
            self._spawn(["open", "-a", target])
        else:
            self._spawn([
                "powershell", "-NoProfile", "-Command",
                f"Start-Process '{target}'",
            ])
        return AppInfo(name=target, running=True)

    def close_application(self, name: str) -> None:
        if self.system == "linux":
            self._run(["pkill", "-x", name])
        elif self.system == "darwin":
            self._run([
                "osascript", "-e",
                f'tell application "{name}" to quit',
            ])
        else:
            self._run([
                "powershell", "-NoProfile", "-Command",
                f"Stop-Process -Name '{name}' -ErrorAction "
                "SilentlyContinue",
            ])

    def capabilities(self) -> dict[str, bool]:
        return {
            "windows": True,
            "mouse": True,
            "keyboard": True,
            "accessibility": False,
            "screenshots": False,
            "applications": True,
        }

import speech_recognition as sr
import subprocess
import webbrowser
import os
import sys
import platform
import shutil
import urllib.parse
from pathlib import Path
from datetime import datetime

try:
    import pywhatkit
except Exception:
    pywhatkit = None

try:
    import ollama
except Exception:
    ollama = None

try:
    import pyautogui
except Exception:
    pyautogui = None

# ------------------------------------------------------------------
# Platform detection
# ------------------------------------------------------------------
SYSTEM = platform.system()  # "Windows", "Darwin" (macOS), "Linux"
IS_WINDOWS = SYSTEM == "Windows"
IS_MACOS = SYSTEM == "Darwin"
IS_LINUX = SYSTEM == "Linux"

recognizer = sr.Recognizer()

# GIF Animation Configuration
GIF_PATH = "afnan_animation.gif"  # Change this to your GIF filename


def show_startup_gif():
    """Show GIF animation in browser (Windows / macOS / Linux)"""
    try:
        gif_absolute_path = os.path.abspath(GIF_PATH)

        if os.path.exists(gif_absolute_path):
            # On Windows, webbrowser needs a proper file:// URL
            file_url = Path(gif_absolute_path).as_uri()

            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Afnan AI</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: black;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            overflow: hidden;
        }}
        .afnan-gif {{
            max-width: 90vw;
            max-height: 90vh;
        }}
    </style>
</head>
<body>
    <div class="afnan-container">
        <img src="{file_url}" alt="Afnan AI Animation" class="afnan-gif">
    </div>
</body>
</html>
            """

            html_file = "afnan_animation.html"
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html_content)

            html_path = os.path.abspath(html_file)
            webbrowser.open(Path(html_path).as_uri())
            print("✅ Afnan AI animation opened in browser")

        else:
            print(f"❌ GIF file not found: {gif_absolute_path}")
            print("💡 Continuing without animation...")

    except Exception as e:
        print(f"❌ GIF Error: {e}")
        print("💡 Continuing without animation...")


# ------------------------------------------------------------------
# Text to speech — cross platform
# ------------------------------------------------------------------
_tts_engine = None


def _speak_with_pyttsx3(text):
    """Speak with pyttsx3 (works offline on Windows, macOS and Linux)."""
    global _tts_engine
    import pyttsx3  # type: ignore

    if _tts_engine is None:
        _tts_engine = pyttsx3.init()
    _tts_engine.say(text)
    _tts_engine.runAndWait()


def _speak_with_system(text):
    """Fallback to the operating system's native speech command."""
    if IS_MACOS:
        subprocess.run(["say", text], check=False)
    elif IS_WINDOWS:
        # Windows built-in SAPI via PowerShell — no extra install needed
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
    elif IS_LINUX:
        if shutil.which("spd-say"):
            subprocess.run(["spd-say", text], check=False)
        elif shutil.which("espeak"):
            subprocess.run(["espeak", text], check=False)
        else:
            print("(no Linux TTS engine found — install espeak or pyttsx3)")


def speak(text):
    try:
        print("afnan:", text)
        try:
            _speak_with_pyttsx3(text)
        except Exception:
            _speak_with_system(text)
    except Exception as e:
        print("Speech Error:", e)


# ------------------------------------------------------------------
# Open files / folders / apps — cross platform
# ------------------------------------------------------------------
def open_path(path):
    """Open a file or folder with the default OS handler."""
    try:
        if IS_WINDOWS:
            os.startfile(path)  # type: ignore[attr-defined]
        elif IS_MACOS:
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:
        print("Open Path Error:", e)


def open_app_windows(app_key):
    """Launch a common app on Windows. Returns True if something launched."""
    # command, url-protocol or executable to try, in order
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


def open_app_macos(app_key):
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


def open_app_linux(app_key):
    linux_apps = {
        "vscode": ["code"],
        "chrome": ["google-chrome", "chromium-browser", "chromium"],
        "edge": ["microsoft-edge"],
        "whatsapp": [],  # falls back to WhatsApp Web below
        "safari": [],
    }
    for candidate in linux_apps.get(app_key, []):
        if shutil.which(candidate):
            subprocess.run([candidate], check=False)
            return True
    return False


def launch_app(app_key):
    if IS_WINDOWS:
        return open_app_windows(app_key)
    if IS_MACOS:
        return open_app_macos(app_key)
    return open_app_linux(app_key)


# -------------------- INTRODUCTION -------------------- #
def introduce_yourself():
    speak("""
Hello! I am Afnan.

Created by Afnan.

I am not just a simple assistant — I am smart, fast, and always ready to help.

I can control your system, search anything, play music and write code,
and assist you like a real AI companion.

What do you want me to do?
""")


# -------------------- FOLDER SEARCH -------------------- #
_KNOWN_FOLDERS = {
    "downloads": "Downloads",
    "download": "Downloads",
    "desktop": "Desktop",
    "documents": "Documents",
    "document": "Documents",
    "pictures": "Pictures",
    "music": "Music",
    "videos": "Videos",
}


def _find_folder_windows(foldername):
    """Search the user's home folder (limited depth) on Windows."""
    target = foldername.strip().lower()
    if target in _KNOWN_FOLDERS:
        candidate = Path.home() / _KNOWN_FOLDERS[target]
        if candidate.is_dir():
            return str(candidate)

    home = Path.home()
    # shallow search first, then a limited walk
    for parent in [home, home / "Desktop", home / "Documents", home / "Downloads"]:
        if not parent.exists():
            continue
        for child in parent.iterdir():
            if child.is_dir() and child.name.lower() == target:
                return str(child)

    for root, dirs, _files in os.walk(home):
        # keep it fast — do not dig too deep
        depth = Path(root).relative_to(home).parts
        if len(depth) > 3:
            dirs[:] = []
            continue
        for d in dirs:
            if d.lower() == target:
                return os.path.join(root, d)
    return None


def _find_folder_macos(foldername):
    target = foldername.strip().lower()
    if target in _KNOWN_FOLDERS:
        candidate = Path.home() / _KNOWN_FOLDERS[target]
        if candidate.is_dir():
            return str(candidate)

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


def _find_folder_linux(foldername):
    target = foldername.strip().lower()
    if target in _KNOWN_FOLDERS:
        candidate = Path.home() / _KNOWN_FOLDERS[target]
        if candidate.is_dir():
            return str(candidate)

    home = Path.home()
    for root, dirs, _files in os.walk(home):
        depth = Path(root).relative_to(home).parts
        if len(depth) > 3:
            dirs[:] = []
            continue
        for d in dirs:
            if d.lower() == target:
                return os.path.join(root, d)
    return None


def open_folder_anywhere(foldername):
    try:
        foldername = foldername.strip()
        if not foldername:
            speak("Please tell me the folder name boss")
            return

        if IS_WINDOWS:
            path = _find_folder_windows(foldername)
        elif IS_MACOS:
            path = _find_folder_macos(foldername)
        else:
            path = _find_folder_linux(foldername)

        if path:
            speak("Opening folder")
            open_path(path)
        else:
            speak("Folder not found boss")

    except Exception as e:
        print("Folder Search Error:", e)
        speak("Error while opening folder")


def ask_local_ai(prompt):
    try:
        if ollama is None:
            return "Sorry boss, AI is not available. Ollama is not installed."
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
    except Exception as e:
        print("Ollama Error:", e)
        return "Sorry boss, AI is not responding. Make sure Ollama is running."


def listen_command(timeout=5, phrase_time=6):
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("Listening...")
            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time,
            )

        return recognizer.recognize_google(audio, language="en-IN")

    except sr.WaitTimeoutError:
        return ""
    except Exception:
        return ""


def play_song(command):
    try:
        song = command.lower().replace("play", "", 1).strip()

        if not song:
            speak("Please tell me the song name.")
            return

        speak(f"Playing {song} on YouTube")
        if pywhatkit is not None:
            pywhatkit.playonyt(song)
        else:
            query = urllib.parse.quote(song)
            webbrowser.open(f"https://www.youtube.com/results?search_query={query}")

    except Exception:
        speak("Sorry boss")


def take_screenshot():
    if pyautogui is None:
        speak("Sorry boss, screenshot is not available")
        return None

    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = screenshots_dir / f"screenshot_{timestamp}.png"

    screenshot = pyautogui.screenshot()
    screenshot.save(str(file_path))

    open_path(str(file_path))
    return str(file_path)


# -------------------- COMMAND PROCESSOR -------------------- #
def process_command(command):
    command = command.lower().strip()

    try:
        if "open visual studio code" in command or "open vs code" in command:
            speak("Opening Visual Studio Code")
            if not launch_app("vscode"):
                speak("Visual Studio Code not found boss")

        elif "open safari" in command:
            if IS_WINDOWS or IS_LINUX:
                speak("Safari is not available on this system, opening your default browser")
                webbrowser.open("https://www.apple.com/safari/")
            else:
                speak("Opening Safari")
                launch_app("safari")

        elif "open chrome" in command or "open google chrome" in command:
            speak("Opening Chrome")
            if not launch_app("chrome"):
                webbrowser.open("https://www.google.com")

        elif "open edge" in command or "open microsoft edge" in command:
            speak("Opening Microsoft Edge")
            if not launch_app("edge"):
                speak("Microsoft Edge not found boss")

        elif "open youtube" in command:
            speak("Opening YouTube")
            webbrowser.open("https://youtube.com")

        elif "open whatsapp" in command:
            speak("Opening WhatsApp")
            if not launch_app("whatsapp"):
                webbrowser.open("https://web.whatsapp.com")

        elif (
            "tell me about yourself" in command
            or "introduce yourself" in command
            or "who are you" in command
        ):
            introduce_yourself()

        elif "folder" in command and command.startswith("open"):
            foldername = (
                command.replace("open", "").replace("folder", "").strip()
            )
            open_folder_anywhere(foldername)

        elif "search youtube for" in command:
            query = command.replace("search youtube for", "").strip()
            webbrowser.open(
                f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
            )

        elif "search google for" in command:
            query = command.replace("search google for", "").strip()
            webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(query)}")

        elif command.startswith("play "):
            play_song(command)

        elif "screenshot" in command:
            take_screenshot()
            speak("Screenshot taken")

        elif "stop afnan" in command:
            speak("Goodbye boss")
            raise SystemExit

        else:
            speak("Thinking boss")
            reply = ask_local_ai(command)
            speak(reply)

    except SystemExit:
        raise
    except Exception as e:
        print("Command Error:", e)
        speak("Error boss")


# -------------------- MAIN LOOP -------------------- #
def start_afnan():
    # Show GIF in browser first (Windows / macOS / Linux)
    show_startup_gif()

    speak("Afnan is activated")

    while True:
        try:
            word = listen_command(timeout=5, phrase_time=3)

            if not word:
                continue

            if "afnan" in word.lower():
                speak("Yes boss")

                command = listen_command(timeout=7, phrase_time=8)

                if command:
                    process_command(command)

        except SystemExit:
            break
        except KeyboardInterrupt:
            break
        except Exception:
            pass


if __name__ == "__main__":
    try:
        start_afnan()
    except KeyboardInterrupt:
        print("\nAfnan AI stopped by user")

"""Core Afnan AI agent — platform-agnostic.

This module contains *only* assistant behaviour: wake word, command
routing, web search, music, screenshots and local-AI fallback.  Every
operating-system specific action is delegated to a
:class:`~afnan_ai.platform.base.PlatformAdapter`, selected at runtime
by :func:`afnan_ai.platform.get_adapter`.  There is intentionally no
OS-specific launching, searching or speech code in this file.
"""

from __future__ import annotations

import os
import urllib.parse
import webbrowser
from datetime import datetime
from pathlib import Path

from afnan_ai import speech as _speech
from afnan_ai.platform import get_adapter
from afnan_ai.platform.base import PlatformAdapter
from afnan_ai.state import AgentState

try:
    import speech_recognition as sr
except Exception:  # pragma: no cover - optional at import time in tests
    sr = None  # type: ignore

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


GIF_PATH = "afnan_animation.gif"


class AfnanAgent:
    """Platform-agnostic voice assistant."""

    def __init__(
        self,
        adapter: PlatformAdapter | None = None,
        state: AgentState | None = None,
        *,
        track_state: bool = True,
    ):
        self.adapter = adapter or get_adapter()
        self.recognizer = sr.Recognizer() if sr is not None else None
        # Centralized, serializable task state.  Components may pass
        # their own AgentState, read ``agent.state``, or ignore it —
        # existing behaviour is unchanged when they do.
        self.track_state = track_state
        self.state: AgentState | None = state

    # -- state helpers ---------------------------------------------------
    def start_task(self, goal: str) -> AgentState:
        """Start tracking a new task and return its AgentState."""
        self.state = AgentState.create(goal)
        self.state.start_task()
        return self.state

    def _state_begin(self, command: str) -> AgentState | None:
        if not self.track_state:
            return None
        if self.state is None or self.state.is_terminal:
            self.state = AgentState.create(command or "voice command")
            self.state.start_task()
        self.state.add_observation(command, source="command")
        self.state.start_step(command or "voice command")
        return self.state

    def _state_succeed(self, result=None) -> None:
        if self.state is not None and self.track_state:
            # complete the running step, but keep the task running so
            # the agent can take the next command in the same session
            if self.state.current_step:
                self.state.complete_step(self.state.current_step, result=result)

    def _state_fail(self, error: str) -> None:
        if self.state is not None and self.track_state:
            if self.state.current_step:
                self.state.fail_step(self.state.current_step, error)
            self.state.add_tool_result("process_command", success=False, error=error)

    # -- speech --------------------------------------------------------
    def speak(self, text: str) -> None:
        _speech.speak(text, self.adapter)

    # -- startup animation ---------------------------------------------
    def show_startup_gif(self, gif_path: str = GIF_PATH) -> None:
        try:
            gif_absolute_path = os.path.abspath(gif_path)
            if not os.path.exists(gif_absolute_path):
                print(f"❌ GIF file not found: {gif_absolute_path}")
                print("💡 Continuing without animation...")
                return

            file_url = Path(gif_absolute_path).as_uri()
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Afnan AI</title>
    <style>
        body {{ margin: 0; padding: 0; background: black; display: flex;
               justify-content: center; align-items: center; height: 100vh;
               overflow: hidden; }}
        .afnan-gif {{ max-width: 90vw; max-height: 90vh; }}
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
            webbrowser.open(Path(os.path.abspath(html_file)).as_uri())
            print("✅ Afnan AI animation opened in browser")
        except Exception as e:
            print(f"❌ GIF Error: {e}")
            print("💡 Continuing without animation...")

    # -- introduction ----------------------------------------------------
    def introduce_yourself(self) -> None:
        self.speak(
            """
Hello! I am Afnan.

Created by Afnan.

I am not just a simple assistant — I am smart, fast, and always ready to help.

I can control your system, search anything, play music and write code,
and assist you like a real AI companion.

What do you want me to do?
"""
        )

    # -- folders ---------------------------------------------------------
    def open_folder_anywhere(self, foldername: str) -> None:
        try:
            foldername = (foldername or "").strip()
            if not foldername:
                self.speak("Please tell me the folder name boss")
                return
            path = self.adapter.find_folder(foldername)
            if path:
                self.speak("Opening folder")
                self.adapter.open_path(path)
            else:
                self.speak("Folder not found boss")
        except Exception as e:
            print("Folder Search Error:", e)
            self.speak("Error while opening folder")

    # -- local AI ----------------------------------------------------------
    def ask_local_ai(self, prompt: str) -> str:
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

    # -- listening -----------------------------------------------------------
    def listen_command(self, timeout: int = 5, phrase_time: int = 6) -> str:
        if sr is None or self.recognizer is None:
            return ""
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("Listening...")
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time
                )
            return self.recognizer.recognize_google(audio, language="en-IN")
        except Exception:
            return ""

    # -- music -----------------------------------------------------------------
    def play_song(self, command: str) -> None:
        try:
            song = command.lower().replace("play", "", 1).strip()
            if not song:
                self.speak("Please tell me the song name.")
                return
            self.speak(f"Playing {song} on YouTube")
            if pywhatkit is not None:
                pywhatkit.playonyt(song)
            else:
                query = urllib.parse.quote(song)
                webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
        except Exception:
            self.speak("Sorry boss")

    # -- screenshot --------------------------------------------------------------
    def take_screenshot(self) -> str | None:
        if pyautogui is None:
            self.speak("Sorry boss, screenshot is not available")
            return None
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = screenshots_dir / f"screenshot_{timestamp}.png"
        screenshot = pyautogui.screenshot()
        screenshot.save(str(file_path))
        self.adapter.open_path(str(file_path))
        return str(file_path)

    # -- command routing (existing behaviour preserved) ---------------------------
    def process_command(self, command: str) -> None:
        raw_command = command or ""
        command = raw_command.lower().strip()
        self._state_begin(command)
        try:
            if "open visual studio code" in command or "open vs code" in command:
                self.speak("Opening Visual Studio Code")
                if not self.adapter.launch_app("vscode"):
                    self.speak("Visual Studio Code not found boss")

            elif "open safari" in command:
                if not self.adapter.supports_app("safari"):
                    self.speak(
                        "Safari is not available on this system, "
                        "opening your default browser"
                    )
                    webbrowser.open("https://www.apple.com/safari/")
                else:
                    self.speak("Opening Safari")
                    self.adapter.launch_app("safari")

            elif "open chrome" in command or "open google chrome" in command:
                self.speak("Opening Chrome")
                if not self.adapter.launch_app("chrome"):
                    webbrowser.open("https://www.google.com")

            elif "open edge" in command or "open microsoft edge" in command:
                self.speak("Opening Microsoft Edge")
                if not self.adapter.launch_app("edge"):
                    self.speak("Microsoft Edge not found boss")

            elif "open youtube" in command:
                self.speak("Opening YouTube")
                webbrowser.open("https://youtube.com")

            elif "open whatsapp" in command:
                self.speak("Opening WhatsApp")
                if not self.adapter.launch_app("whatsapp"):
                    webbrowser.open("https://web.whatsapp.com")

            elif (
                "tell me about yourself" in command
                or "introduce yourself" in command
                or "who are you" in command
            ):
                self.introduce_yourself()

            elif "folder" in command and command.startswith("open"):
                foldername = command.replace("open", "").replace("folder", "").strip()
                self.open_folder_anywhere(foldername)

            elif "search youtube for" in command:
                query = command.replace("search youtube for", "").strip()
                webbrowser.open(
                    "https://www.youtube.com/results?search_query="
                    + urllib.parse.quote(query)
                )

            elif "search google for" in command:
                query = command.replace("search google for", "").strip()
                webbrowser.open(
                    "https://www.google.com/search?q=" + urllib.parse.quote(query)
                )

            elif command.startswith("play "):
                self.play_song(command)

            elif "screenshot" in command:
                self.take_screenshot()
                self.speak("Screenshot taken")

            elif "stop afnan" in command:
                self.speak("Goodbye boss")
                raise SystemExit

            else:
                self.speak("Thinking boss")
                reply = self.ask_local_ai(command)
                self.speak(reply)

            self._state_succeed(result="ok")

        except SystemExit:
            # "stop afnan" is a successful stop, not a failure
            self._state_succeed(result="stopped")
            raise
        except Exception as e:
            print("Command Error:", e)
            self._state_fail(str(e))
            self.speak("Error boss")

    # -- main loop ------------------------------------------------------------------
    def start(self) -> None:
        self.show_startup_gif()
        self.speak("Afnan is activated")
        while True:
            try:
                word = self.listen_command(timeout=5, phrase_time=3)
                if not word:
                    continue
                if "afnan" in word.lower():
                    self.speak("Yes boss")
                    command = self.listen_command(timeout=7, phrase_time=8)
                    if command:
                        self.process_command(command)
            except SystemExit:
                break
            except KeyboardInterrupt:
                break
            except Exception:
                pass


def create_agent(
    adapter: PlatformAdapter | None = None,
    state: AgentState | None = None,
) -> AfnanAgent:
    return AfnanAgent(adapter=adapter, state=state)

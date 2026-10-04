"""Afnan AI — cross-platform voice assistant (entry point).

All platform-specific code lives in :mod:`afnan_ai.platform`
(Windows / macOS / Linux adapters, selected automatically at
runtime).  This file is a thin, backwards-compatible wrapper:
``python main.py`` still works exactly as before, and the old
module-level functions (``speak``, ``process_command``, ...)
are preserved for anyone importing them.

Usage:
    Windows:  python main.py
    macOS/Linux: python3 main.py
"""

from afnan_ai.agent import AfnanAgent, create_agent
from afnan_ai.platform import get_adapter

# Default agent + adapter for this machine (auto-selected at runtime)
adapter = get_adapter()
_agent = AfnanAgent(adapter=adapter)

# Backwards-compatible module state (previous main.py exposed these)
SYSTEM = adapter.name
IS_WINDOWS = adapter.name == "windows"
IS_MACOS = adapter.name == "macos"
IS_LINUX = adapter.name == "linux"
GIF_PATH = "afnan_animation.gif"
recognizer = _agent.recognizer


def speak(text):
    _agent.speak(text)


def show_startup_gif():
    _agent.show_startup_gif()


def introduce_yourself():
    _agent.introduce_yourself()


def open_folder_anywhere(foldername):
    _agent.open_folder_anywhere(foldername)


def open_path(path):
    adapter.open_path(path)


def launch_app(app_key):
    return adapter.launch_app(app_key)


def ask_local_ai(prompt):
    return _agent.ask_local_ai(prompt)


def listen_command(timeout=5, phrase_time=6):
    return _agent.listen_command(timeout=timeout, phrase_time=phrase_time)


def play_song(command):
    _agent.play_song(command)


def take_screenshot():
    return _agent.take_screenshot()


def process_command(command):
    _agent.process_command(command)


def start_afnan():
    _agent.start()


if __name__ == "__main__":
    try:
        start_afnan()
    except KeyboardInterrupt:
        print("\nAfnan AI stopped by user")

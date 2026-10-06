"""Afnan AI — full PC demo (no microphone needed).

Shows the two things you can see and hear on this machine:
1. Voice OUTPUT — Afnan speaks through the speakers
   (Linux needs `sudo apt install -y espeak` first).
2. Browser — the same Chromium the agent drives: it opens visibly,
   navigates to a page, and saves a screenshot.

Usage:
    python demo_pc.py

Note: voice INPUT (microphone / wake word) needs real audio hardware,
so it only works when Afnan runs on Windows/macOS directly, not in WSL.
"""

from __future__ import annotations

import sys


def demo_voice(agent) -> None:
    print("\n--- 1. Voice output (sunno) ---")
    agent.speak("Afnan PC demo shuru ho raha hai. Ab browser khol raha hun.")
    print("done: Afnan spoke")


def demo_browser() -> str:
    print("\n--- 2. Browser (dekho) ---")
    from afnan_ai.browser import BrowserController

    controller = BrowserController(headless=False)
    info = controller.launch()
    print("browser launched:", info.get("browser", "chromium"))
    page = controller.navigate("https://example.com")
    print("navigated:", page.get("url", ""))
    shot = controller.screenshot()
    path = shot.get("path", "demo_screenshot.png")
    print("screenshot saved:", path)
    return path


def main() -> int:
    from afnan_ai.agent import AfnanAgent
    from afnan_ai.platform import get_adapter

    agent = AfnanAgent(adapter=get_adapter())
    demo_voice(agent)
    shot_path = demo_browser()
    agent.speak("Demo mukammal ho gaya.")
    print("\nDemo complete. Screenshot:", shot_path)
    print("Browser window khuli rahegi — band karne ke liye Ctrl+C.")
    try:
        import threading

        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

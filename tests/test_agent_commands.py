import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai.agent import AfnanAgent
from afnan_ai.platform.base import PlatformAdapter


class FakeAdapter(PlatformAdapter):
    """In-memory adapter that records calls instead of touching the OS."""

    name = "linux"

    def __init__(self):
        self.spoken = []
        self.opened = []
        self.launched = []

    def speak_system(self, text):
        self.spoken.append(text)

    def open_path(self, path):
        self.opened.append(path)

    def launch_app(self, app_key):
        self.launched.append(app_key)
        return app_key != "safari"

    def find_folder(self, foldername):
        return "/fake/Downloads" if foldername.lower().startswith("download") else None


def make_agent():
    agent = AfnanAgent(adapter=FakeAdapter())
    agent.speak = lambda text: agent.adapter.spoken.append(text)  # capture, don't TTS
    return agent


class TestExistingFeaturesPreserved(unittest.TestCase):
    def test_open_chrome(self):
        agent = make_agent()
        agent.process_command("open chrome")
        self.assertIn("chrome", agent.adapter.launched)

    def test_open_vs_code(self):
        agent = make_agent()
        agent.process_command("open vs code")
        self.assertIn("vscode", agent.adapter.launched)

    def test_open_youtube_uses_browser(self):
        agent = make_agent()
        with mock.patch("webbrowser.open") as wb:
            agent.process_command("open youtube")
            wb.assert_called_once()
            self.assertIn("youtube", str(wb.call_args).lower())

    def test_search_google_builds_url(self):
        agent = make_agent()
        with mock.patch("webbrowser.open") as wb:
            agent.process_command("search google for python")
            self.assertIn("google.com/search", str(wb.call_args))
            self.assertIn("python", str(wb.call_args))

    def test_search_youtube_builds_url(self):
        agent = make_agent()
        with mock.patch("webbrowser.open") as wb:
            agent.process_command("search youtube for ai")
            self.assertIn("youtube.com/results", str(wb.call_args))

    def test_open_folder_downloads(self):
        agent = make_agent()
        agent.process_command("open folder downloads")
        self.assertIn("/fake/Downloads", agent.adapter.opened)

    def test_folder_not_found_does_not_open(self):
        agent = make_agent()
        agent.process_command("open folder doesnotexist")
        self.assertEqual(agent.adapter.opened, [])

    def test_screenshot_uses_adapter_to_open(self):
        agent = make_agent()
        fake_shot = mock.Mock()
        with mock.patch("afnan_ai.agent.pyautogui") as pg:
            pg.screenshot.return_value = fake_shot
            result = agent.take_screenshot()
            self.assertTrue(result.startswith("screenshots"))
            self.assertTrue(agent.adapter.opened)

    def test_stop_afnan_raises_system_exit(self):
        agent = make_agent()
        with self.assertRaises(SystemExit):
            agent.process_command("stop afnan")

    def test_unknown_command_falls_back_to_ai(self):
        agent = make_agent()
        agent.llm.chat_stream = lambda messages: iter(["ai reply"])
        agent.process_command("what is the weather")
        self.assertIn("ai reply", agent.adapter.spoken)

    def test_safari_on_non_mac_opens_default_browser(self):
        agent = make_agent()  # FakeAdapter is "linux", no Safari
        with mock.patch("webbrowser.open") as wb:
            agent.process_command("open safari")
            wb.assert_called_once()


class TestCoreHasNoOsSpecificCode(unittest.TestCase):
    def test_agent_module_has_no_os_specific_calls(self):
        agent_src = (
            Path(__file__).resolve().parents[1] / "afnan_ai" / "agent.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("os.startfile", '"open", "-a"', "mdfind", "xdg-open", "powershell"):
            self.assertNotIn(
                forbidden, agent_src,
                f"core agent must delegate {forbidden!r} to a platform adapter",
            )


if __name__ == "__main__":
    unittest.main()

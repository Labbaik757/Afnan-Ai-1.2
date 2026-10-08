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


class TestHeardTextNormalization(unittest.TestCase):
    """STT transcripts carry invisible characters (Urdu STT inserts
    zero-width joiners between words). They look identical on the
    console but used to fail every legacy substring match, silently
    sending the command down the slow LLM planning path."""

    def test_normalize_turns_zero_width_separators_into_spaces(self):
        from afnan_ai.agent import _normalize_command_text

        self.assertEqual(
            _normalize_command_text("براؤزر‌اوپن‌کرو"),
            "براؤزر اوپن کرو",
        )
        self.assertEqual(
            _normalize_command_text("براؤزر​اوپن​کرو"),
            "براؤزر اوپن کرو",
        )

    def test_normalize_collapses_whitespace_and_bom(self):
        from afnan_ai.agent import _normalize_command_text

        self.assertEqual(
            _normalize_command_text("﻿  open   browser  "),
            "open browser",
        )
        self.assertEqual(_normalize_command_text(""), "")
        self.assertEqual(_normalize_command_text(None), "")

    def test_urdu_browser_command_with_zwnj_takes_legacy_path(self):
        agent = make_agent()
        # If normalization regresses, the command falls through to
        # the orchestrator instead of the instant legacy handler.
        agent.orchestrator.run = mock.Mock(
            side_effect=AssertionError("must not reach LLM planning")
        )
        from afnan_ai.tools.base import ToolResult
        with mock.patch.object(
            agent, "execute_tool",
            return_value=ToolResult.ok("browser_launch")
        ) as et:
            agent.process_command("براؤزر‌اوپن‌کرو")  # U+200C between words
            et.assert_called()
            calls = [c.args[0] for c in et.call_args_list]
            self.assertIn("browser_launch", calls)
            self.assertIn("browser_navigate", calls)
        self.assertIn("Opening browser", agent.adapter.spoken)

    def test_plain_urdu_browser_command_still_works(self):
        agent = make_agent()
        from afnan_ai.tools.base import ToolResult
        with mock.patch.object(
            agent, "execute_tool",
            return_value=ToolResult.ok("browser_launch")
        ) as et:
            agent.process_command("براؤزر اوپن کرو")
            et.assert_called()
        self.assertIn("Opening browser", agent.adapter.spoken)

    def test_browser_launch_failure_is_spoken(self):
        agent = make_agent()
        from afnan_ai.tools.base import ToolResult, ToolError, ToolErrorCode
        with mock.patch.object(
            agent, "execute_tool",
            return_value=ToolResult.fail(
                "browser_launch",
                ToolError(ToolErrorCode.EXECUTION_FAILED,
                          "launch failed", tool="browser_launch"))
        ):
            agent.process_command("open browser")
        self.assertTrue(
            any("nahin khul saka" in s for s in agent.adapter.spoken),
            f"expected failure message, got: {agent.adapter.spoken}")


if __name__ == "__main__":
    unittest.main()

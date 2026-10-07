import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai import Agent
from afnan_ai.agent import AfnanAgent
from afnan_ai.config import AgentConfig
from afnan_ai.executor import Executor
from afnan_ai.llm.base import LLMConnectionError, LLMProvider
from afnan_ai.orchestrator import OrchestrationStatus
from afnan_ai.platform.base import PlatformAdapter
from afnan_ai.planner import Planner
from afnan_ai.state import TaskStatus
from afnan_ai.tools import ToolRegistry, create_default_registry
from afnan_ai.tools.base import Tool, ToolExecutionError
from afnan_ai.verifier import Verifier


class FakeAdapter(PlatformAdapter):
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
        return True

    def find_folder(self, foldername):
        return (
            "/fake/Downloads"
            if foldername.lower().startswith("download")
            else None
        )


class StubLLM(LLMProvider):
    name = "stub"
    display_name = "Stub"
    model = "stub-1"

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error

    def chat(self, messages):
        if self.error is not None:
            raise self.error
        return self.reply


class FailingTool(Tool):
    name = "always_fails"
    description = "Always fails (test tool)."
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments):
        raise ToolExecutionError("kaboom", tool=self.name)


CHROME_PLAN = json.dumps(
    {
        "goal": "Open Chrome",
        "steps": [
            {
                "step_id": "step_1",
                "description": "Open the Chrome browser",
                "tool_name": "open_application",
                "arguments": {"application": "chrome"},
                "expected_result": "Chrome is launched",
            }
        ],
    }
)

FAIL_PLAN = json.dumps(
    {
        "goal": "Do the thing",
        "steps": [
            {
                "step_id": "step_1",
                "description": "Run the failing tool",
                "tool_name": "always_fails",
                "arguments": {},
                "expected_result": "The tool succeeds",
            }
        ],
    }
)


def make_assistant(llm, **kwargs):
    adapter = FakeAdapter()
    agent = AfnanAgent(adapter=adapter, llm_provider=llm, **kwargs)
    agent.speak = lambda text: adapter.spoken.append(text)
    return agent, adapter


class TestVoiceIndependentAgentInvocation(unittest.TestCase):
    """Agent.run() works with no voice, speech or adapter at all."""

    def test_agent_from_components_runs_without_assistant(self):
        adapter = FakeAdapter()
        registry = create_default_registry(
            adapter, opener=lambda url: True
        )
        agent = Agent.from_components(StubLLM(reply=CHROME_PLAN), registry)
        result = agent.run("Open Chrome")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(adapter.launched, ["chrome"])
        self.assertEqual(result.state.status, TaskStatus.COMPLETED)

    def test_agent_failure_without_assistant(self):
        registry = ToolRegistry([FailingTool()])
        agent = Agent(
            planner=Planner(StubLLM(reply=FAIL_PLAN), registry),
            executor=Executor(registry),
            verifier=Verifier(),
        )
        result = agent.run("Do the thing")
        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        self.assertFalse(result.success)


class TestHandleRequestDelegatesToAgent(unittest.TestCase):
    def test_successful_task_completion(self):
        agent, adapter = make_assistant(
            StubLLM(reply=CHROME_PLAN),
            config=AgentConfig(fast_path=False),
        )
        result = agent.handle_request("Open Chrome please")
        self.assertIsNotNone(result)
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(adapter.launched, ["chrome"])
        self.assertIn("Done boss", adapter.spoken)
        self.assertIs(agent.state, result.state)

    def test_typed_request_uses_same_path(self):
        agent, adapter = make_assistant(
            StubLLM(reply=CHROME_PLAN),
            config=AgentConfig(fast_path=False),
        )
        result = agent.process_command("Open Chrome please")
        self.assertIsNotNone(result)
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(adapter.launched, ["chrome"])

    def test_agent_failure_is_surfaced_and_spoken(self):
        registry = ToolRegistry([FailingTool()])
        adapter = FakeAdapter()
        agent = AfnanAgent(
            adapter=adapter,
            llm_provider=StubLLM(reply=FAIL_PLAN),
            tool_registry=registry,
        )
        agent.speak = lambda text: adapter.spoken.append(text)
        result = agent.handle_request("Open the thing")
        self.assertIsNotNone(result)
        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        self.assertTrue(
            any("could not complete" in s for s in adapter.spoken)
        )
        self.assertEqual(agent.state.status, TaskStatus.FAILED)

    def test_looks_like_task(self):
        self.assertTrue(AfnanAgent._looks_like_task("open chrome please"))
        self.assertTrue(AfnanAgent._looks_like_task("براؤزر کھولو"))
        self.assertTrue(AfnanAgent._looks_like_task("screenshot lo"))
        self.assertFalse(AfnanAgent._looks_like_task("arslan"))
        self.assertFalse(AfnanAgent._looks_like_task("tell me a joke"))
        self.assertFalse(AfnanAgent._looks_like_task("what is the meaning of life"))

    def test_free_conversation_goes_to_llm_not_orchestrator(self):
        agent, adapter = make_assistant(
            StubLLM(reply="Arslan is a name, boss"),
        )
        result = agent.handle_request("arslan")
        # Chat path returns None (no orchestration), speaks the reply.
        self.assertIsNone(result)
        self.assertIn("Arslan is a name, boss", adapter.spoken)


class TestFastPath(unittest.TestCase):
    """Simple commands skip LLM planning when fast_path is on."""

    def test_legacy_command_bypasses_orchestrator(self):
        agent, adapter = make_assistant(StubLLM(reply="unused"))
        agent.orchestrator.run = mock.Mock(
            side_effect=AssertionError("planner must not run")
        )
        result = agent.handle_request("Open Chrome please")
        self.assertIsNone(result)
        self.assertEqual(adapter.launched, ["chrome"])
        self.assertIn("Opening Chrome", adapter.spoken)

    def test_urdu_open_browser(self):
        agent, adapter = make_assistant(StubLLM(reply="unused"))
        agent.orchestrator.run = mock.Mock(
            side_effect=AssertionError("planner must not run")
        )
        with mock.patch("webbrowser.open") as wb:
            result = agent.handle_request("براؤزر اوپن کرو")
        self.assertIsNone(result)
        self.assertIn("Opening browser", adapter.spoken)
        wb.assert_called_once()
        self.assertIn("google.com", str(wb.call_args).lower())

    def test_chitchat_goes_to_chat_fallback(self):
        agent, adapter = make_assistant(StubLLM(reply="chat reply"))
        agent.orchestrator.run = mock.Mock(
            side_effect=AssertionError("planner must not run")
        )
        result = agent.handle_request("kya haal hai")
        self.assertIsNone(result)
        self.assertIn("Thinking boss", adapter.spoken)
        self.assertIn("chat reply", adapter.spoken)

    def test_chitchat_urdu(self):
        self.assertTrue(AfnanAgent._is_chitchat("کیا حال ہے"))
        self.assertTrue(AfnanAgent._is_chitchat("assalamualaikum"))
        self.assertTrue(AfnanAgent._is_chitchat("shukriya boss"))
        self.assertFalse(AfnanAgent._is_chitchat("open chrome"))
        self.assertFalse(AfnanAgent._is_chitchat("research python"))

    def test_fast_path_can_be_disabled(self):
        agent, adapter = make_assistant(
            StubLLM(reply=CHROME_PLAN),
            config=AgentConfig(fast_path=False),
        )
        result = agent.handle_request("open the pod bay doors")
        # Task-like but not a known legacy command: with fast_path
        # off it goes through the orchestrator, not the fast path.
        self.assertIsNotNone(result)


class TestBackwardCompatibility(unittest.TestCase):
    def test_unplannable_request_falls_back_to_legacy_command(self):
        # Model cannot produce a plan -> old direct routing still works
        agent, adapter = make_assistant(StubLLM(reply="not json at all"))
        result = agent.process_command("open chrome")
        self.assertIsNone(result)  # handled by the legacy path
        self.assertEqual(adapter.launched, ["chrome"])

    def test_folder_capability_preserved(self):
        # Opening a folder by name is not a Tool yet; it must keep
        # working through the isolated legacy routing.
        agent, adapter = make_assistant(
            StubLLM(error=LLMConnectionError("model offline"))
        )
        agent.process_command("open folder downloads")
        self.assertEqual(adapter.opened, ["/fake/Downloads"])

    def test_stop_afnan_still_exits(self):
        agent, adapter = make_assistant(StubLLM(reply=CHROME_PLAN))
        with self.assertRaises(SystemExit):
            agent.process_command("stop afnan")
        self.assertIn("Goodbye boss", adapter.spoken)

    def test_empty_request_is_a_no_op(self):
        agent, adapter = make_assistant(StubLLM(reply=CHROME_PLAN))
        self.assertIsNone(agent.handle_request("   "))
        self.assertEqual(adapter.launched, [])


class TestMainHasNoDuplicateLogic(unittest.TestCase):
    def test_main_only_delegates(self):
        main_src = (
            Path(__file__).resolve().parents[1] / "main.py"
        ).read_text(encoding="utf-8")
        # main.py may expose thin delegates, but must not contain
        # planning/execution/verification/recovery logic itself.
        for forbidden in (
            "execute_step(",
            "verify_step(",
            "plan_recovery(",
            "recovery_context",
            "_handle_legacy_command",
        ):
            self.assertNotIn(forbidden, main_src)
        self.assertIn("def handle_request(", main_src)
        self.assertIn("_agent.handle_request", main_src)


if __name__ == "__main__":
    unittest.main()

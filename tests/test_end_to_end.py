"""End-to-end tests for the full Phase-1 Agent pipeline.

These tests drive one user goal through every component in order:

    User Goal -> AgentState -> Planner -> Executor -> Verifier
              -> Recovery/Replanning -> Completion / Failure

plus the same journey through the voice/text entry points and on
all three platform adapters (Windows / macOS / Linux).  Nothing
touches a real OS, browser, microphone or model: the LLM is
scripted and every platform call is recorded by fakes/mocks.
"""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai import Agent, AgentConfig, configure_logging
from afnan_ai.agent import AfnanAgent
from afnan_ai.executor import Executor
from afnan_ai.llm.base import LLMConnectionError, LLMProvider
from afnan_ai.orchestrator import OrchestrationStatus
from afnan_ai.platform import get_adapter
from afnan_ai.platform.base import PlatformAdapter
from afnan_ai.platform.linux import LinuxAdapter
from afnan_ai.platform.macos import MacOSAdapter
from afnan_ai.platform.windows import WindowsAdapter
from afnan_ai.planner import Planner
from afnan_ai.state import AgentState, TaskStatus
from afnan_ai.tools import create_default_registry
from afnan_ai.tools.base import Tool, ToolExecutionError
from afnan_ai.verifier import Verifier


# ----------------------------------------------------------------------
# Scripted LLM + fake platform
# ----------------------------------------------------------------------

class SequencedLLM(LLMProvider):
    name = "seq"
    display_name = "Seq"
    model = "seq-1"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def chat(self, messages):
        self.prompts.append(messages)
        if self.replies:
            return self.replies.pop(0)
        raise LLMConnectionError("no scripted reply left")


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
        return None


class FailingValueTool(Tool):
    name = "attempt_action"
    description = "Attempt an action that always fails (test tool)."
    input_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def __init__(self, sink):
        self.sink = sink

    def run(self, arguments):
        self.sink.append(arguments["value"])
        raise ToolExecutionError(
            f"action {arguments['value']} failed", tool=self.name
        )


def plan_json(steps, goal="Research Python"):
    return json.dumps({"goal": goal, "steps": steps})


CHROME_STEP = {
    "step_id": "step_1",
    "description": "Open the Chrome browser",
    "tool_name": "open_application",
    "arguments": {"application": "chrome"},
    "expected_result": "Chrome is launched",
}
FAIL_STEP = {
    "step_id": "step_2",
    "description": "Attempt the flaky action",
    "tool_name": "attempt_action",
    "arguments": {"value": "flaky"},
    "expected_result": "The action succeeds",
}
SEARCH_STEP = {
    "step_id": "step_r1",
    "description": "Search Google for Python instead",
    "tool_name": "search_google",
    "arguments": {"query": "Python"},
    "expected_result": "Google results for Python are open",
}


def make_pipeline(llm, sink, adapter=None, **agent_kwargs):
    adapter = adapter or FakeAdapter()
    registry = create_default_registry(adapter, opener=lambda url: True)
    registry.register(FailingValueTool(sink))
    agent = Agent(
        planner=Planner(llm, registry),
        executor=Executor(registry),
        verifier=Verifier(),
        **agent_kwargs,
    )
    return agent, adapter


# ----------------------------------------------------------------------
# The full journey: goal -> ... -> recovery -> completion
# ----------------------------------------------------------------------

class TestFullPipelineWithRecovery(unittest.TestCase):
    def test_goal_to_completion_through_every_component(self):
        sink = []
        llm = SequencedLLM(
            [plan_json([CHROME_STEP, FAIL_STEP]), plan_json([SEARCH_STEP])]
        )
        agent, adapter = make_pipeline(llm, sink)

        result = agent.run("Research Python")

        # Completion end-state
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertTrue(result.success)
        state = result.state
        self.assertEqual(state.status, TaskStatus.COMPLETED)

        # Planner produced the original plan; Recovery produced a
        # second, different plan after the failure
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("Recovery context", llm.prompts[1][-1]["content"]
                      if isinstance(llm.prompts[1], list) else str(llm.prompts[1]))

        # Executor ran chrome, then the flaky action (failed once,
        # never blindly retried), then the recovered search
        self.assertEqual(adapter.launched, ["chrome"])
        self.assertEqual(sink, ["flaky"])
        self.assertEqual(
            [r.success for r in state.tool_results], [True, False, True]
        )

        # Verifier judged all three executed steps
        verdicts = {
            v["step_id"]: v["status"]
            for v in state.metadata["verifications"]
        }
        self.assertEqual(verdicts["step_1"], "verified")
        self.assertEqual(verdicts["step_2"], "failed")
        self.assertEqual(verdicts["step_r1"], "verified")

        # Recovery was attempted exactly once, with the failure as
        # its reason, and recorded in AgentState
        attempts = state.metadata["recovery_attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["trigger"], "execution_failed")
        self.assertEqual(attempts[0]["outcome"], "replanned")
        self.assertIn("flaky", attempts[0]["reason"])

        # The plan and the journey are all in the state, and the
        # final state survives a save/load round trip unchanged
        self.assertIn("plan", state.metadata)
        with tempfile.TemporaryDirectory() as tmp:
            loaded = AgentState.load(state.save(Path(tmp) / "state.json"))
        self.assertEqual(loaded.status, TaskStatus.COMPLETED)
        self.assertEqual(loaded.goal, state.goal)
        self.assertEqual(loaded.metadata["recovery_attempts"], attempts)

        # The aggregated execution report matches the completed
        # task; the verification report honestly keeps the failed
        # step in its history (recovery fixed the task, it does
        # not rewrite what happened)
        self.assertEqual(result.execution.status, "completed")
        self.assertEqual(len(result.verification.results), 3)
        self.assertEqual(
            result.verification.status.value, "failed"  # step_2 did fail
        )
        json.loads(result.to_json())

    def test_goal_to_failure_end_state(self):
        sink = []
        # Every plan (original + both recovery plans) fails
        llm = SequencedLLM(
            [plan_json([FAIL_STEP])] * 3
        )
        agent, _ = make_pipeline(llm, sink)
        result = agent.run("Research Python")

        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        self.assertFalse(result.success)
        self.assertEqual(result.state.status, TaskStatus.FAILED)
        self.assertTrue(result.state.error)
        self.assertEqual([s.name for s in result.state.failed_steps],
                         ["step_2"])
        # plan 1 fails -> recovery rejects the identical replan
        # twice more would be repeats; the LLM only had 3 replies
        self.assertEqual(sink, ["flaky"])  # never blindly retried

    def test_planning_failure_end_state(self):
        sink = []
        llm = SequencedLLM(["no json here"])
        agent, adapter = make_pipeline(llm, sink)
        result = agent.run("Research Python")
        self.assertEqual(
            result.status, OrchestrationStatus.PLANNING_FAILED
        )
        self.assertEqual(result.state.status, TaskStatus.FAILED)
        self.assertEqual(adapter.launched, [])
        self.assertEqual(result.state.tool_results, [])


# ----------------------------------------------------------------------
# Same journey through the voice/text entry points
# ----------------------------------------------------------------------

class TestEntryPointEndToEnd(unittest.TestCase):
    def make_assistant(self, llm):
        adapter = FakeAdapter()
        registry = create_default_registry(
            adapter, opener=lambda url: True
        )
        assistant = AfnanAgent(
            adapter=adapter, llm_provider=llm, tool_registry=registry
        )
        assistant.speak = lambda text: adapter.spoken.append(text)
        return assistant, adapter

    def test_voice_request_recovers_and_completes(self):
        sink = []
        llm = SequencedLLM(
            [plan_json([FAIL_STEP]), plan_json([SEARCH_STEP])]
        )
        assistant, adapter = self.make_assistant(llm)
        # register the flaky tool into the assistant's registry too
        assistant.tools.register(FailingValueTool(sink))

        result = assistant.handle_request("Research Python for me")

        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertIn("Done boss", adapter.spoken)
        self.assertEqual(assistant.state.status, TaskStatus.COMPLETED)
        self.assertEqual(
            len(assistant.state.metadata["recovery_attempts"]), 1
        )

    def test_typed_request_failure_is_reported(self):
        sink = []
        llm = SequencedLLM([plan_json([FAIL_STEP])] * 3)
        assistant, adapter = self.make_assistant(llm)
        assistant.tools.register(FailingValueTool(sink))

        result = assistant.process_command("Research Python for me")

        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        self.assertTrue(
            any("could not complete" in s for s in adapter.spoken)
        )
        self.assertEqual(sink, ["flaky"])


# ----------------------------------------------------------------------
# Cross-platform: the same pipeline on all three adapters
# ----------------------------------------------------------------------

class TestCrossPlatformPipeline(unittest.TestCase):
    PLAN = plan_json([CHROME_STEP, SEARCH_STEP], goal="Open Chrome and search")

    def _run_with_adapter(self, adapter, which_map):
        registry = create_default_registry(
            adapter, opener=lambda url: True
        )
        llm = SequencedLLM([self.PLAN])
        agent = Agent(
            planner=Planner(llm, registry),
            executor=Executor(registry),
            verifier=Verifier(),
        )
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return mock.Mock(returncode=0, stdout="")

        with mock.patch("subprocess.run", side_effect=fake_run), \
            mock.patch(
                "shutil.which",
                side_effect=lambda name: which_map.get(name),
            ):
            result = agent.run("Open Chrome and search")
        return result, calls

    def test_pipeline_on_windows_adapter(self):
        result, calls = self._run_with_adapter(WindowsAdapter(), {})
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        flat = [str(part) for cmd in calls for part in cmd]
        self.assertIn("chrome", flat)  # cmd /c start chrome

    def test_pipeline_on_macos_adapter(self):
        result, calls = self._run_with_adapter(MacOSAdapter(), {})
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertIn(["open", "-a", "Google Chrome"], calls)

    def test_pipeline_on_linux_adapter(self):
        result, calls = self._run_with_adapter(
            LinuxAdapter(), {"google-chrome": "/usr/bin/google-chrome"}
        )
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertIn(["google-chrome"], calls)

    def test_adapter_selection_matrix(self):
        self.assertIsInstance(get_adapter("Windows"), WindowsAdapter)
        self.assertIsInstance(get_adapter("Darwin"), MacOSAdapter)
        self.assertIsInstance(get_adapter("Linux"), LinuxAdapter)


# ----------------------------------------------------------------------
# Phase-1 infrastructure: config + structured logging
# ----------------------------------------------------------------------

class TestPhase1Infrastructure(unittest.TestCase):
    def test_config_defaults(self):
        cfg = AgentConfig()
        self.assertEqual(cfg.wake_word, "afnan")
        self.assertEqual(cfg.max_iterations, 10)
        self.assertEqual(cfg.max_recovery_attempts, 2)

    def test_config_from_env(self):
        env = {
            "AFNAN_MAX_ITERATIONS": "4",
            "AFNAN_WAKE_WORD": "afnan",
            "AFNAN_MAX_RECOVERY_ATTEMPTS": "bogus",
        }
        with mock.patch.dict("os.environ", env, clear=False):
            cfg = AgentConfig.from_env()
        self.assertEqual(cfg.max_iterations, 4)
        self.assertEqual(cfg.max_recovery_attempts, 2)  # bogus ignored

    def test_agent_uses_config_limits(self):
        cfg = AgentConfig(max_iterations=3, max_recovery_attempts=1)
        assistant = AfnanAgent(adapter=FakeAdapter(), config=cfg)
        self.assertEqual(assistant.max_iterations, 3)
        self.assertEqual(assistant.orchestrator.max_recovery_attempts, 1)

    def test_structured_logging_emits_task_lines(self):
        stream = io.StringIO()
        configure_logging(stream=stream)
        sink = []
        llm = SequencedLLM([plan_json([CHROME_STEP])])
        agent, _ = make_pipeline(llm, sink)
        agent.run("Research Python")
        out = stream.getvalue()
        self.assertIn("afnan_ai.orchestrator", out)
        self.assertIn("completed", out)


if __name__ == "__main__":
    unittest.main()

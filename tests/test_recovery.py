import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai import Agent
from afnan_ai.executor import Executor
from afnan_ai.llm.base import LLMConnectionError, LLMProvider
from afnan_ai.orchestrator import OrchestrationStatus
from afnan_ai.platform.base import PlatformAdapter
from afnan_ai.planner import Planner
from afnan_ai.recovery import RecoveryManager
from afnan_ai.state import TaskStatus
from afnan_ai.tools import create_default_registry
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
        return None


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


class SequencedLLM(LLMProvider):
    """Returns replies in order; raises once exhausted."""

    name = "seq"
    display_name = "Seq"
    model = "seq-1"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if self.replies:
            return self.replies.pop(0)
        raise LLMConnectionError("no scripted reply left")

    def last_prompt(self, messages=None):
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


class RecordingTool(Tool):
    name = "record_action"
    description = "Record a value (test tool)."
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
        return {"recorded": arguments["value"], "opened": True}


def fail_plan(value):
    return json.dumps(
        {
            "goal": "Do the thing",
            "steps": [
                {
                    "step_id": "step_1",
                    "description": f"Attempt action {value}",
                    "tool_name": "attempt_action",
                    "arguments": {"value": value},
                    "expected_result": "The action succeeds",
                }
            ],
        }
    )


def search_plan():
    return json.dumps(
        {
            "goal": "Do the thing",
            "steps": [
                {
                    "step_id": "step_r1",
                    "description": "Search Google for Python instead",
                    "tool_name": "search_google",
                    "arguments": {"query": "Python"},
                    "expected_result": "Google results for Python are open",
                }
            ],
        }
    )


def make_agent(llm, sink, **kwargs):
    adapter = FakeAdapter()
    registry = create_default_registry(adapter, opener=lambda url: True)
    registry.register(FailingValueTool(sink))
    registry.register(RecordingTool(sink))
    agent = Agent(
        planner=Planner(llm, registry),
        executor=Executor(registry),
        verifier=Verifier(),
        **kwargs,
    )
    return agent, adapter


class TestSuccessfulRecovery(unittest.TestCase):
    def test_failed_step_recovers_with_different_plan(self):
        sink = []
        llm = SequencedLLM([fail_plan("v1"), search_plan()])
        agent, _ = make_agent(llm, sink)
        result = agent.run("Do the thing")

        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertTrue(result.success)
        self.assertEqual(sink, ["v1"])  # failed once, never retried
        self.assertEqual(len(result.recovery_attempts), 1)
        attempt = result.recovery_attempts[0]
        self.assertEqual(attempt["trigger"], "execution_failed")
        self.assertEqual(attempt["outcome"], "replanned")
        self.assertEqual(attempt["tool_name"], "attempt_action")
        self.assertIn("v1", attempt["reason"] + str(attempt["arguments"]))
        # recorded in AgentState too
        state_attempts = result.state.metadata["recovery_attempts"]
        self.assertEqual(len(state_attempts), 1)
        self.assertEqual(result.state.status, TaskStatus.COMPLETED)
        # the whole result stays serializable
        json.loads(result.to_json())

    def test_uncertain_step_triggers_recovery(self):
        sink = []
        vague_plan = json.dumps(
            {
                "goal": "Do the thing",
                "steps": [
                    {
                        "step_id": "step_1",
                        "description": "Record something",
                        "tool_name": "record_action",
                        "arguments": {"value": "x"},
                        "expected_result": "Done",
                    }
                ],
            }
        )
        llm = SequencedLLM([vague_plan, search_plan()])
        agent, _ = make_agent(llm, sink)
        result = agent.run("Do the thing")

        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(len(result.recovery_attempts), 1)
        self.assertEqual(
            result.recovery_attempts[0]["trigger"], "verification_uncertain"
        )
        self.assertEqual(
            result.recovery_attempts[0]["outcome"], "replanned"
        )


class TestRepeatedFailureNotRetried(unittest.TestCase):
    def test_same_plan_again_is_rejected_as_repeated_action(self):
        sink = []
        llm = StubLLM(reply=fail_plan("v1"))  # always the same plan
        agent, _ = make_agent(llm, sink)
        result = agent.run("Do the thing")

        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        self.assertEqual(sink, ["v1"])  # executed exactly once
        self.assertEqual(len(result.recovery_attempts), 1)
        attempt = result.recovery_attempts[0]
        self.assertEqual(attempt["outcome"], "repeated_action")
        self.assertEqual(
            result.error["details"]["recovery"]["code"], "repeated_action"
        )
        self.assertEqual(result.state.status, TaskStatus.FAILED)


class TestPlannerReplanFailure(unittest.TestCase):
    def test_invalid_replan_output_fails_task(self):
        sink = []
        llm = SequencedLLM([fail_plan("v1"), "sorry, no plan"])
        agent, _ = make_agent(llm, sink)
        result = agent.run("Do the thing")

        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        self.assertEqual(sink, ["v1"])
        self.assertEqual(len(result.recovery_attempts), 1)
        self.assertEqual(
            result.recovery_attempts[0]["outcome"], "planner_failed"
        )
        self.assertEqual(
            result.error["details"]["recovery"]["code"], "planner_failed"
        )

    def test_replan_connection_failure_fails_task(self):
        sink = []
        llm = SequencedLLM([fail_plan("v1")])  # then LLM is gone
        agent, _ = make_agent(llm, sink)
        result = agent.run("Do the thing")
        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        self.assertEqual(
            result.recovery_attempts[0]["outcome"], "planner_failed"
        )


class TestRecoveryLimit(unittest.TestCase):
    def test_recovery_attempts_are_capped(self):
        sink = []
        llm = SequencedLLM(
            [fail_plan("v1"), fail_plan("v2"), fail_plan("v3")]
        )
        agent, _ = make_agent(llm, sink)  # default max_recovery_attempts=2
        result = agent.run("Do the thing")

        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        # each distinct action tried exactly once, then recovery stopped
        self.assertEqual(sink, ["v1", "v2", "v3"])
        self.assertEqual(len(result.recovery_attempts), 2)
        self.assertTrue(
            all(
                a["outcome"] == "replanned"
                for a in result.recovery_attempts
            )
        )
        self.assertTrue(result.error["details"]["recovery_exhausted"])
        self.assertEqual(
            len(result.state.metadata["recovery_attempts"]), 2
        )
        self.assertEqual(result.state.status, TaskStatus.FAILED)

    def test_recovery_respects_iteration_limit(self):
        sink = []
        llm = SequencedLLM(
            [fail_plan("v1"), fail_plan("v2"), fail_plan("v3")]
        )
        agent, _ = make_agent(llm, sink, max_iterations=2)
        result = agent.run("Do the thing")
        # v1 + v2 executed; the iteration cap stops the task before v3
        self.assertEqual(
            result.status, OrchestrationStatus.MAX_ITERATIONS_EXCEEDED
        )
        self.assertEqual(sink, ["v1", "v2"])
        self.assertEqual(result.iterations, 2)

    def test_recovery_can_be_disabled(self):
        sink = []
        llm = SequencedLLM([fail_plan("v1"), search_plan()])
        agent, _ = make_agent(llm, sink, max_recovery_attempts=0)
        result = agent.run("Do the thing")
        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        self.assertEqual(result.recovery_attempts, [])
        self.assertEqual(sink, ["v1"])
        self.assertEqual(llm.calls, 1)  # planner never asked again

    def test_manager_rejects_negative_attempts(self):
        with self.assertRaises(ValueError):
            RecoveryManager(Planner(StubLLM(reply="{}"), []), max_attempts=-1)


if __name__ == "__main__":
    unittest.main()

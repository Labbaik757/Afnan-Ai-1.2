import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai import Agent
from afnan_ai.agent import AfnanAgent
from afnan_ai.executor import Executor
from afnan_ai.llm.base import LLMConnectionError, LLMProvider
from afnan_ai.orchestrator import OrchestrationStatus
from afnan_ai.platform.base import PlatformAdapter
from afnan_ai.planner import Planner
from afnan_ai.state import AgentState, TaskStatus
from afnan_ai.tools import ToolRegistry, create_default_registry
from afnan_ai.tools.base import Tool, ToolExecutionError
from afnan_ai.verifier import VerificationStatus, Verifier


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
        raise ToolExecutionError("kaboom: this tool failed", tool=self.name)


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


def plan_json(steps, goal="Open Chrome and search for Python"):
    return json.dumps({"goal": goal, "steps": steps})


def chrome_step(step_id="step_1"):
    return {
        "step_id": step_id,
        "description": "Open the Chrome browser",
        "tool_name": "open_application",
        "arguments": {"application": "chrome"},
        "expected_result": "Chrome is launched",
    }


def search_step(step_id="step_2"):
    return {
        "step_id": step_id,
        "description": "Search Google for Python",
        "tool_name": "search_google",
        "arguments": {"query": "Python"},
        "expected_result": "Google results for Python are open",
    }


def record_step(step_id, value):
    return {
        "step_id": step_id,
        "description": f"Record {value}",
        "tool_name": "record_action",
        "arguments": {"value": value},
        "expected_result": f"{value} value is recorded and opened",
    }


def make_orchestrator(llm, registry=None, adapter=None, **kwargs):
    adapter = adapter or FakeAdapter()
    registry = registry or create_default_registry(
        adapter, opener=lambda url: True
    )
    agent = Agent(
        planner=Planner(llm, registry),
        executor=Executor(registry),
        verifier=Verifier(),
        **kwargs,
    )
    return agent, registry, adapter


class TestSuccessfulCompletion(unittest.TestCase):
    def test_full_lifecycle_completes(self):
        llm = StubLLM(reply=plan_json([chrome_step(), search_step()]))
        agent, _, adapter = make_orchestrator(llm)
        result = agent.run("Open Chrome and search for Python")

        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertTrue(result.success)
        self.assertEqual(result.iterations, 2)
        self.assertEqual(adapter.launched, ["chrome"])

        state = result.state
        self.assertEqual(state.status, TaskStatus.COMPLETED)
        self.assertEqual(
            [s.name for s in state.completed_steps], ["step_1", "step_2"]
        )
        self.assertEqual(state.failed_steps, [])
        self.assertEqual(len(state.metadata["verifications"]), 2)
        self.assertTrue(
            all(
                v["status"] == "verified"
                for v in state.metadata["verifications"]
            )
        )
        self.assertIsNotNone(result.plan)
        self.assertTrue(result.execution.success)
        self.assertEqual(
            result.verification.status, VerificationStatus.VERIFIED
        )
        # whole result stays serializable
        json.loads(result.to_json())

    def test_run_reuses_provided_state(self):
        llm = StubLLM(reply=plan_json([chrome_step()]))
        agent, _, _ = make_orchestrator(llm)
        state = AgentState.create("Open Chrome and search for Python")
        result = agent.run(
            "Open Chrome and search for Python", state=state
        )
        self.assertTrue(result.success)
        self.assertIs(result.state, state)
        self.assertEqual(state.status, TaskStatus.COMPLETED)


class TestPlanningFailure(unittest.TestCase):
    def test_invalid_llm_output_is_planning_failed(self):
        llm = StubLLM(reply="Here is your plan, boss: first open Chrome...")
        agent, _, adapter = make_orchestrator(llm)
        result = agent.run("Open Chrome")

        self.assertEqual(
            result.status, OrchestrationStatus.PLANNING_FAILED
        )
        self.assertFalse(result.success)
        self.assertIsNone(result.plan)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.error["code"], "invalid_llm_output")
        self.assertEqual(adapter.launched, [])  # nothing executed
        self.assertEqual(result.state.status, TaskStatus.FAILED)
        self.assertIn("not valid JSON", result.state.error)

    def test_llm_connection_failure_is_planning_failed(self):
        llm = StubLLM(error=LLMConnectionError("no server"))
        agent, _, adapter = make_orchestrator(llm)
        result = agent.run("Open Chrome")
        self.assertEqual(
            result.status, OrchestrationStatus.PLANNING_FAILED
        )
        self.assertEqual(result.error["code"], "llm_connection_failed")
        self.assertEqual(adapter.launched, [])

    def test_empty_goal_rejected(self):
        agent, _, _ = make_orchestrator(StubLLM(reply="{}"))
        with self.assertRaises(ValueError):
            agent.run("   ")


class TestExecutionFailure(unittest.TestCase):
    def test_failing_tool_stops_task_as_failed(self):
        recorded = []
        registry = ToolRegistry([FailingTool(), RecordingTool(recorded)])
        llm = StubLLM(
            reply=plan_json(
                [
                    {
                        "step_id": "step_1",
                        "description": "Run the failing tool",
                        "tool_name": "always_fails",
                        "arguments": {},
                        "expected_result": "The tool succeeds",
                    },
                    record_step("step_2", "must-not-run"),
                ],
                goal="Do the thing",
            )
        )
        agent, _, _ = make_orchestrator(llm, registry=registry)
        result = agent.run("Do the thing")

        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        self.assertFalse(result.success)
        self.assertEqual(recorded, [])  # later step never ran
        self.assertEqual(result.iterations, 1)
        state = result.state
        self.assertEqual(state.status, TaskStatus.FAILED)
        self.assertEqual([s.name for s in state.failed_steps], ["step_1"])
        self.assertFalse(state.tool_results[0].success)
        # the step that never ran is reported skipped, not done
        skipped = result.execution.step_results[1]
        self.assertTrue(skipped.skipped)
        self.assertFalse(skipped.success)

    def test_unknown_tool_in_plan_fails_task(self):
        # A planner that hands back a step naming a tool the
        # registry does not have (e.g. a stale plan): the Executor
        # must fail the step and the Agent must fail the task.
        from afnan_ai.planner import PlanStep, TaskPlan

        class FixedPlanner:
            def plan(self, goal, state=None):
                return TaskPlan(
                    goal=goal,
                    steps=[
                        PlanStep(
                            step_id="step_1",
                            description="Fly to the moon",
                            tool_name="fly_to_moon",
                            arguments={},
                            expected_result="On the moon",
                        )
                    ],
                )

        recorded = []
        registry = ToolRegistry([RecordingTool(recorded)])
        agent = Agent(
            planner=FixedPlanner(),
            executor=Executor(registry),
            verifier=Verifier(),
        )
        result = agent.run("Fly to the moon")
        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        self.assertEqual(
            result.execution.step_results[0].error["code"], "tool_not_found"
        )
        self.assertEqual(recorded, [])
        self.assertEqual(result.state.status, TaskStatus.FAILED)


class TestMaxIterations(unittest.TestCase):
    def make_long_plan_agent(self, recorded, max_iterations=None):
        registry = ToolRegistry([RecordingTool(recorded)])
        steps = [record_step(f"step_{i}", f"v{i}") for i in range(1, 6)]
        llm = StubLLM(reply=plan_json(steps, goal="Record five values"))
        kwargs = {}
        if max_iterations is not None:
            kwargs["max_iterations"] = max_iterations
        agent, _, _ = make_orchestrator(llm, registry=registry, **kwargs)
        return agent

    def test_max_iterations_stops_long_plan(self):
        recorded = []
        agent = self.make_long_plan_agent(recorded, max_iterations=2)
        result = agent.run("Record five values")

        self.assertEqual(
            result.status, OrchestrationStatus.MAX_ITERATIONS_EXCEEDED
        )
        self.assertFalse(result.success)
        self.assertEqual(result.iterations, 2)
        self.assertEqual(recorded, ["v1", "v2"])  # steps 3-5 never ran
        self.assertEqual(result.error["code"], "max_iterations_exceeded")
        self.assertEqual(result.state.status, TaskStatus.FAILED)
        self.assertIn("Maximum iteration limit", result.state.error)
        # unexecuted steps are reported as skipped, never successful
        self.assertEqual(len(result.execution.skipped_steps), 3)

    def test_per_run_limit_override(self):
        recorded = []
        agent = self.make_long_plan_agent(recorded, max_iterations=10)
        result = agent.run("Record five values", max_iterations=1)
        self.assertEqual(
            result.status, OrchestrationStatus.MAX_ITERATIONS_EXCEEDED
        )
        self.assertEqual(recorded, ["v1"])

    def test_limit_must_be_positive(self):
        llm = StubLLM(reply="{}")
        registry = ToolRegistry()
        for bad in (0, -1, 1.5, "3", True):
            with self.assertRaises(ValueError, msg=str(bad)):
                Agent(
                    planner=Planner(llm, registry),
                    executor=Executor(registry),
                    verifier=Verifier(),
                    max_iterations=bad,
                )

    def test_plan_within_limit_completes(self):
        recorded = []
        agent = self.make_long_plan_agent(recorded, max_iterations=5)
        result = agent.run("Record five values")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(recorded, ["v1", "v2", "v3", "v4", "v5"])


class TestAgentOrchestrationIntegration(unittest.TestCase):
    def make_agent(self, reply):
        adapter = FakeAdapter()
        agent = AfnanAgent(
            adapter=adapter, llm_provider=StubLLM(reply=reply)
        )
        agent.speak = lambda text: adapter.spoken.append(text)
        return agent, adapter

    def test_afnan_agent_run_task_end_to_end(self):
        agent, adapter = self.make_agent(
            plan_json([chrome_step(), search_step()])
        )
        result = agent.run_task("Open Chrome and search for Python")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(adapter.launched, ["chrome"])
        self.assertIs(agent.state, result.state)
        self.assertEqual(agent.state.status, TaskStatus.COMPLETED)
        self.assertIs(agent.orchestrator.planner, agent.planner)
        self.assertIs(agent.orchestrator.executor, agent.executor)
        self.assertIs(agent.orchestrator.verifier, agent.verifier)

    def test_afnan_agent_run_task_planning_failure(self):
        agent, adapter = self.make_agent("definitely not json")
        result = agent.run_task("Open Chrome")
        self.assertEqual(
            result.status, OrchestrationStatus.PLANNING_FAILED
        )
        self.assertEqual(adapter.launched, [])

    def test_afnan_agent_run_task_max_iterations(self):
        steps = [chrome_step("step_1"), search_step("step_2"),
                 chrome_step("step_3")]
        agent, adapter = self.make_agent(plan_json(steps))
        result = agent.run_task(
            "Open Chrome and search for Python", max_iterations=1
        )
        self.assertEqual(
            result.status, OrchestrationStatus.MAX_ITERATIONS_EXCEEDED
        )
        self.assertEqual(adapter.launched, ["chrome"])

    def test_afnan_agent_default_max_iterations(self):
        agent, _ = self.make_agent(plan_json([chrome_step()]))
        self.assertEqual(agent.max_iterations, 10)
        self.assertEqual(agent.orchestrator.max_iterations, 10)

    def test_existing_voice_command_still_works(self):
        agent, adapter = self.make_agent(plan_json([chrome_step()]))
        agent.process_command("open chrome")
        self.assertEqual(adapter.launched, ["chrome"])
        agent.process_command("open youtube")
        self.assertTrue(adapter.launched)


if __name__ == "__main__":
    unittest.main()

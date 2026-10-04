import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai.agent import AfnanAgent
from afnan_ai.llm.base import (
    LLMConnectionError,
    LLMInvalidResponseError,
    LLMProvider,
    LLMUnavailableError,
)
from afnan_ai.planner import (
    PlanStep,
    Planner,
    PlannerErrorCode,
    PlanningError,
    TaskPlan,
)
from afnan_ai.platform.base import PlatformAdapter
from afnan_ai.state import AgentState
from afnan_ai.tools import ToolRegistry, create_default_registry


class StubLLM(LLMProvider):
    """LLM stub returning a canned reply (or raising a canned error)."""

    name = "stub"
    display_name = "Stub"
    model = "stub-1"

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.prompts = []

    def chat(self, messages):
        self.prompts.append(messages[-1]["content"])
        if self.error is not None:
            raise self.error
        return self.reply


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


def valid_output(steps=None):
    return json.dumps(
        {
            "goal": "Open Chrome and search for Python",
            "steps": steps
            or [
                {
                    "step_id": "step_1",
                    "description": "Open the Chrome browser",
                    "tool_name": "open_application",
                    "arguments": {"application": "chrome"},
                    "expected_result": "Chrome is launched",
                },
                {
                    "step_id": "step_2",
                    "description": "Search Google for Python",
                    "tool_name": "search_google",
                    "arguments": {"query": "Python"},
                    "expected_result": "Google results for Python are open",
                },
            ],
        }
    )


def make_planner(reply=None, error=None, adapter=None):
    llm = StubLLM(reply=reply, error=error)
    registry = create_default_registry(adapter or FakeAdapter())
    return Planner(llm, registry), llm, registry


class TestSuccessfulPlanning(unittest.TestCase):
    def test_plan_from_valid_llm_output(self):
        planner, _, _ = make_planner(reply=valid_output())
        plan = planner.plan("Open Chrome and search for Python")
        self.assertIsInstance(plan, TaskPlan)
        self.assertEqual(plan.goal, "Open Chrome and search for Python")
        self.assertTrue(plan.plan_id)
        self.assertEqual(len(plan.steps), 2)

        step = plan.steps[0]
        self.assertIsInstance(step, PlanStep)
        self.assertEqual(step.step_id, "step_1")
        self.assertEqual(step.tool_name, "open_application")
        self.assertEqual(step.arguments, {"application": "chrome"})
        self.assertEqual(step.expected_result, "Chrome is launched")
        self.assertEqual(plan.steps[1].tool_name, "search_google")

    def test_planner_does_not_execute_tools(self):
        adapter = FakeAdapter()
        opened = []
        registry = create_default_registry(adapter, opener=opened.append)
        planner = Planner(StubLLM(reply=valid_output()), registry)
        planner.plan("Open Chrome and search for Python")
        # planning must not launch apps or open any URL
        self.assertEqual(adapter.launched, [])
        self.assertEqual(adapter.opened, [])
        self.assertEqual(opened, [])

    def test_prompt_includes_goal_state_and_tools(self):
        planner, llm, _ = make_planner(reply=valid_output())
        state = AgentState.create("previous task")
        state.start_step("open chrome")
        state.complete_step("open chrome")
        state.add_observation("user likes python", source="test")
        planner.plan("Open Chrome and search for Python", state=state)

        prompt = llm.prompts[0]
        self.assertIn("Open Chrome and search for Python", prompt)
        self.assertIn("open chrome", prompt)  # completed step from state
        self.assertIn("open_application", prompt)
        self.assertIn("search_google", prompt)
        self.assertIn("take_screenshot", prompt)

    def test_plan_carries_state_task_id(self):
        planner, _, _ = make_planner(reply=valid_output())
        state = AgentState.create("something", task_id="task-42")
        plan = planner.plan("Open Chrome and search for Python", state=state)
        self.assertEqual(plan.task_id, "task-42")

    def test_fenced_json_output_is_accepted(self):
        planner, _, _ = make_planner(reply="```json\n" + valid_output() + "\n```")
        plan = planner.plan("Open Chrome and search for Python")
        self.assertEqual(len(plan.steps), 2)

    def test_plan_serialization_round_trip(self):
        planner, _, _ = make_planner(reply=valid_output())
        plan = planner.plan("Open Chrome and search for Python")
        restored = TaskPlan.from_json(plan.to_json())
        self.assertEqual(restored.goal, plan.goal)
        self.assertEqual(restored.plan_id, plan.plan_id)
        self.assertEqual(restored.steps[0].tool_name, "open_application")
        self.assertEqual(restored.steps[1].arguments, {"query": "Python"})
        json.dumps(plan.to_dict())

    def test_parse_plan_without_calling_llm(self):
        planner, llm, _ = make_planner(reply="should not be used")
        plan = planner.parse_plan(valid_output(), goal="goal x")
        self.assertEqual(plan.goal, "goal x")
        self.assertEqual(llm.prompts, [])


class TestInvalidPlanning(unittest.TestCase):
    def assert_planning_error(self, reply, *codes):
        planner, _, _ = make_planner(reply=reply)
        with self.assertRaises(PlanningError) as ctx:
            planner.plan("some goal")
        self.assertIn(ctx.exception.code, codes)
        return ctx.exception

    def test_non_json_output(self):
        err = self.assert_planning_error(
            "Sure! First open Chrome, then search...",
            PlannerErrorCode.INVALID_LLM_OUTPUT,
        )
        self.assertIn("not valid JSON", err.error.message)

    def test_json_array_instead_of_object(self):
        self.assert_planning_error(
            '[{"step_id": "step_1"}]', PlannerErrorCode.INVALID_LLM_OUTPUT
        )

    def test_missing_steps(self):
        self.assert_planning_error(
            '{"goal": "x"}', PlannerErrorCode.INVALID_LLM_OUTPUT
        )

    def test_empty_steps(self):
        self.assert_planning_error(
            '{"goal": "x", "steps": []}', PlannerErrorCode.INVALID_LLM_OUTPUT
        )

    def test_unknown_top_level_field(self):
        data = json.loads(valid_output())
        data["reasoning"] = "because"
        self.assert_planning_error(
            json.dumps(data), PlannerErrorCode.INVALID_LLM_OUTPUT
        )

    def test_step_missing_a_field(self):
        data = json.loads(valid_output())
        del data["steps"][0]["expected_result"]
        self.assert_planning_error(
            json.dumps(data), PlannerErrorCode.INVALID_LLM_OUTPUT
        )

    def test_step_with_extra_field(self):
        data = json.loads(valid_output())
        data["steps"][0]["confidence"] = 0.9
        self.assert_planning_error(
            json.dumps(data), PlannerErrorCode.INVALID_LLM_OUTPUT
        )

    def test_unknown_tool_in_step(self):
        data = json.loads(valid_output())
        data["steps"][0]["tool_name"] = "fly_to_moon"
        err = self.assert_planning_error(
            json.dumps(data), PlannerErrorCode.INVALID_PLAN
        )
        self.assertIn("fly_to_moon", err.error.message)

    def test_missing_tool_argument(self):
        data = json.loads(valid_output())
        data["steps"][1]["arguments"] = {}  # search_google needs query
        self.assert_planning_error(json.dumps(data), PlannerErrorCode.INVALID_PLAN)

    def test_invalid_tool_argument_type(self):
        data = json.loads(valid_output())
        data["steps"][1]["arguments"] = {"query": 123}
        self.assert_planning_error(json.dumps(data), PlannerErrorCode.INVALID_PLAN)

    def test_duplicate_step_id(self):
        data = json.loads(valid_output())
        data["steps"][1]["step_id"] = "step_1"
        self.assert_planning_error(json.dumps(data), PlannerErrorCode.INVALID_PLAN)

    def test_blank_description(self):
        data = json.loads(valid_output())
        data["steps"][0]["description"] = "  "
        self.assert_planning_error(json.dumps(data), PlannerErrorCode.INVALID_PLAN)

    def test_arguments_must_be_object(self):
        data = json.loads(valid_output())
        data["steps"][0]["arguments"] = "chrome"
        self.assert_planning_error(json.dumps(data), PlannerErrorCode.INVALID_PLAN)

    def test_invalid_output_does_not_execute_tools(self):
        adapter = FakeAdapter()
        planner, _, _ = make_planner(reply="not json at all", adapter=adapter)
        with self.assertRaises(PlanningError):
            planner.plan("open chrome")
        self.assertEqual(adapter.launched, [])


class TestFailedPlanning(unittest.TestCase):
    def test_connection_failure(self):
        planner, _, _ = make_planner(error=LLMConnectionError("no server"))
        with self.assertRaises(PlanningError) as ctx:
            planner.plan("some goal")
        self.assertEqual(
            ctx.exception.code, PlannerErrorCode.LLM_CONNECTION_FAILED
        )

    def test_unavailable_llm(self):
        planner, _, _ = make_planner(error=LLMUnavailableError("not installed"))
        with self.assertRaises(PlanningError) as ctx:
            planner.plan("some goal")
        self.assertEqual(ctx.exception.code, PlannerErrorCode.LLM_UNAVAILABLE)

    def test_invalid_llm_response_error(self):
        planner, _, _ = make_planner(error=LLMInvalidResponseError("bad shape"))
        with self.assertRaises(PlanningError) as ctx:
            planner.plan("some goal")
        self.assertEqual(
            ctx.exception.code, PlannerErrorCode.LLM_INVALID_RESPONSE
        )

    def test_empty_llm_response(self):
        planner, _, _ = make_planner(reply="   ")
        with self.assertRaises(PlanningError) as ctx:
            planner.plan("some goal")
        self.assertEqual(
            ctx.exception.code, PlannerErrorCode.LLM_INVALID_RESPONSE
        )

    def test_unexpected_llm_exception_is_wrapped(self):
        planner, _, _ = make_planner(error=RuntimeError("weird crash"))
        with self.assertRaises(PlanningError) as ctx:
            planner.plan("some goal")
        self.assertEqual(ctx.exception.code, PlannerErrorCode.PLANNING_FAILED)

    def test_empty_goal_rejected_without_calling_llm(self):
        planner, llm, _ = make_planner(reply=valid_output())
        with self.assertRaises(PlanningError) as ctx:
            planner.plan("   ")
        self.assertEqual(ctx.exception.code, PlannerErrorCode.EMPTY_GOAL)
        self.assertEqual(llm.prompts, [])

    def test_no_tools_available(self):
        planner = Planner(StubLLM(reply=valid_output()), ToolRegistry())
        with self.assertRaises(PlanningError) as ctx:
            planner.plan("some goal")
        self.assertEqual(ctx.exception.code, PlannerErrorCode.NO_TOOLS_AVAILABLE)

    def test_planning_error_is_serializable(self):
        planner, _, _ = make_planner(error=LLMConnectionError("down"))
        try:
            planner.plan("goal")
        except PlanningError as e:
            self.assertEqual(e.to_dict()["code"], "llm_connection_failed")
            json.dumps(e.to_dict())
        else:  # pragma: no cover
            self.fail("PlanningError not raised")


class TestAgentPlannerIntegration(unittest.TestCase):
    def make_agent(self, reply=None, error=None):
        adapter = FakeAdapter()
        llm = StubLLM(reply=reply if reply is not None else valid_output(), error=error)
        agent = AfnanAgent(adapter=adapter, llm_provider=llm)
        agent.speak = lambda text: adapter.spoken.append(text)
        return agent, adapter

    def test_agent_create_plan_does_not_execute(self):
        agent, adapter = self.make_agent()
        plan = agent.create_plan("Open Chrome and search for Python")
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(adapter.launched, [])
        self.assertEqual(adapter.opened, [])

    def test_agent_planner_uses_agent_tools_and_llm(self):
        agent, _ = self.make_agent()
        self.assertIs(agent.planner.llm, agent.llm)
        self.assertIs(agent.planner.tools, agent.tools)

    def test_agent_create_plan_records_in_state(self):
        agent, _ = self.make_agent()
        state = agent.start_task("Open Chrome and search for Python")
        plan = agent.create_plan("Open Chrome and search for Python", state=state)
        self.assertEqual(plan.task_id, state.task_id)
        self.assertEqual(state.metadata["last_plan"]["goal"], plan.goal)
        self.assertTrue(
            any(o.source == "planner" for o in state.observations)
        )

    def test_agent_invalid_planning_raises_structured(self):
        agent, adapter = self.make_agent(reply="I cannot plan that")
        with self.assertRaises(PlanningError) as ctx:
            agent.create_plan("do something")
        self.assertEqual(
            ctx.exception.code, PlannerErrorCode.INVALID_LLM_OUTPUT
        )
        self.assertEqual(adapter.launched, [])

    def test_agent_failed_planning_raises_structured(self):
        agent, _ = self.make_agent(error=LLMConnectionError("down"))
        with self.assertRaises(PlanningError) as ctx:
            agent.create_plan("do something")
        self.assertEqual(
            ctx.exception.code, PlannerErrorCode.LLM_CONNECTION_FAILED
        )


if __name__ == "__main__":
    unittest.main()

import ast
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai.agent import AfnanAgent
from afnan_ai.executor import (
    ExecutionError,
    Executor,
    ExecutorErrorCode,
)
from afnan_ai.llm.base import LLMProvider
from afnan_ai.platform.base import PlatformAdapter
from afnan_ai.planner import PlanStep, TaskPlan
from afnan_ai.state import AgentState, TaskStatus
from afnan_ai.tools import ToolRegistry, create_default_registry
from afnan_ai.tools.base import Tool, ToolExecutionError


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
        return {"recorded": arguments["value"]}


class FailingTool(Tool):
    name = "always_fails"
    description = "Always fails cleanly (test tool)."
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments):
        raise ToolExecutionError("kaboom: this tool failed", tool=self.name)


class CrashingTool(Tool):
    name = "always_crashes"
    description = "Raises an unexpected exception (test tool)."
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments):
        raise RuntimeError("unexpected explosion")


class StubLLM(LLMProvider):
    name = "stub"
    display_name = "Stub"
    model = "stub-1"

    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages):
        return self.reply


def step(step_id, tool_name, arguments=None, description=None, expected="done"):
    return PlanStep(
        step_id=step_id,
        description=description or f"Do {step_id}",
        tool_name=tool_name,
        arguments=arguments or {},
        expected_result=expected,
    )


def make_plan(steps, goal="Test goal", **kwargs):
    return TaskPlan(goal=goal, steps=steps, **kwargs)


def default_executor(adapter=None, opener=None, extra_tools=(), **kwargs):
    adapter = adapter or FakeAdapter()
    registry = create_default_registry(adapter, opener=opener or (lambda url: True))
    for tool in extra_tools:
        registry.register(tool)
    return Executor(registry, **kwargs), registry, adapter


class TestSuccessfulExecution(unittest.TestCase):
    def test_steps_execute_in_order_through_registry(self):
        adapter = FakeAdapter()
        opened = []
        executor, _, _ = default_executor(adapter=adapter, opener=opened.append)
        plan = make_plan(
            [
                step("step_1", "open_application", {"application": "chrome"}),
                step("step_2", "search_google", {"query": "Python"}),
            ],
            goal="Open Chrome and search for Python",
        )
        report = executor.execute_plan(plan)

        self.assertTrue(report.success)
        self.assertEqual(report.status, "completed")
        self.assertEqual(adapter.launched, ["chrome"])
        self.assertEqual(len(opened), 1)
        self.assertIn("google.com/search?q=Python", opened[0])
        self.assertEqual(
            [r.step_id for r in report.step_results], ["step_1", "step_2"]
        )
        self.assertTrue(all(r.success for r in report.step_results))
        self.assertEqual(report.step_results[0].output["application"], "chrome")

    def test_custom_tool_receives_validated_arguments(self):
        recorded = []
        registry = ToolRegistry([RecordingTool(recorded)])
        executor = Executor(registry)
        report = executor.execute_plan(
            make_plan([step("s1", "record_action", {"value": "hello"})])
        )
        self.assertTrue(report.success)
        self.assertEqual(recorded, ["hello"])
        self.assertEqual(report.step_results[0].output, {"recorded": "hello"})

    def test_report_is_serializable(self):
        executor, _, _ = default_executor()
        report = executor.execute_plan(
            make_plan([step("s1", "open_application", {"application": "chrome"})])
        )
        data = json.loads(report.to_json())
        self.assertEqual(data["status"], "completed")
        self.assertTrue(data["success"])
        self.assertEqual(data["step_results"][0]["tool_name"], "open_application")


class TestFailedExecution(unittest.TestCase):
    def test_failing_tool_is_not_reported_successful(self):
        registry = ToolRegistry([FailingTool()])
        state = AgentState.create("Test goal")
        report = Executor(registry).execute_plan(
            make_plan([step("s1", "always_fails")]), state=state
        )
        self.assertFalse(report.success)
        self.assertEqual(report.status, "failed")
        result = report.step_results[0]
        self.assertFalse(result.success)
        self.assertEqual(result.status, "failed")
        self.assertIn("kaboom", result.error["message"])
        # and AgentState agrees — nothing faked as completed
        self.assertEqual(state.status, TaskStatus.FAILED)
        self.assertFalse(state.is_successful)
        self.assertEqual([s.name for s in state.failed_steps], ["s1"])
        self.assertEqual(state.completed_steps, [])
        self.assertFalse(state.tool_results[0].success)

    def test_crashing_tool_is_wrapped_as_execution_failure(self):
        registry = ToolRegistry([CrashingTool()])
        report = Executor(registry).execute_plan(
            make_plan([step("s1", "always_crashes")])
        )
        self.assertFalse(report.success)
        self.assertEqual(
            report.step_results[0].error["code"], "execution_failed"
        )

    def test_unavailable_capability_fails_honestly(self):
        # screenshot with capture unavailable must fail, not "succeed"
        adapter = FakeAdapter()
        registry = create_default_registry(
            adapter,
            opener=lambda url: True,
            screenshot_capture=lambda: None,
        )
        report = Executor(registry).execute_plan(
            make_plan([step("s1", "take_screenshot")])
        )
        self.assertFalse(report.success)
        self.assertEqual(report.status, "failed")

    def test_failure_stops_plan_and_skips_rest_by_default(self):
        recorded = []
        registry = ToolRegistry([FailingTool(), RecordingTool(recorded)])
        report = Executor(registry).execute_plan(
            make_plan(
                [
                    step("s1", "always_fails"),
                    step("s2", "record_action", {"value": "must not run"}),
                ]
            )
        )
        self.assertFalse(report.success)
        self.assertEqual(recorded, [])  # second step never ran
        skipped = report.step_results[1]
        self.assertTrue(skipped.skipped)
        self.assertFalse(skipped.success)
        self.assertEqual(skipped.status, "skipped")

    def test_continue_on_failure_attempts_all_but_still_fails(self):
        recorded = []
        registry = ToolRegistry([FailingTool(), RecordingTool(recorded)])
        executor = Executor(registry, stop_on_failure=False)
        state = AgentState.create("Test goal")
        report = executor.execute_plan(
            make_plan(
                [
                    step("s1", "always_fails"),
                    step("s2", "record_action", {"value": "ran anyway"}),
                ]
            ),
            state=state,
        )
        self.assertEqual(recorded, ["ran anyway"])
        self.assertFalse(report.success)  # overall still failed
        self.assertEqual(state.status, TaskStatus.FAILED)
        self.assertEqual([s.name for s in state.completed_steps], ["s2"])
        self.assertEqual([s.name for s in state.failed_steps], ["s1"])


class TestInvalidToolAndArguments(unittest.TestCase):
    def test_unknown_tool_fails_without_executing_anything(self):
        recorded = []
        registry = ToolRegistry([RecordingTool(recorded)])
        state = AgentState.create("Test goal")
        report = Executor(registry).execute_plan(
            make_plan([step("s1", "fly_to_moon", {"value": "x"})]), state=state
        )
        self.assertFalse(report.success)
        self.assertEqual(
            report.step_results[0].error["code"], "tool_not_found"
        )
        self.assertEqual(recorded, [])
        self.assertEqual([s.name for s in state.failed_steps], ["s1"])

    def test_missing_argument_fails_before_execution(self):
        opened = []
        executor, _, _ = default_executor(opener=opened.append)
        report = executor.execute_plan(
            make_plan([step("s1", "search_google", {})])  # query required
        )
        self.assertFalse(report.success)
        self.assertEqual(
            report.step_results[0].error["code"], "missing_arguments"
        )
        self.assertEqual(opened, [])  # tool body never ran

    def test_wrong_argument_type_fails(self):
        executor, _, _ = default_executor()
        report = executor.execute_plan(
            make_plan([step("s1", "search_google", {"query": 123})])
        )
        self.assertFalse(report.success)
        self.assertEqual(
            report.step_results[0].error["code"], "invalid_arguments"
        )

    def test_unknown_argument_fails(self):
        executor, _, _ = default_executor()
        report = executor.execute_plan(
            make_plan(
                [step("s1", "search_google", {"query": "x", "evil": "y"})]
            )
        )
        self.assertFalse(report.success)
        self.assertEqual(
            report.step_results[0].error["code"], "invalid_arguments"
        )

    def test_non_object_arguments_fail(self):
        bad_step = PlanStep(
            step_id="s1",
            description="bad args",
            tool_name="search_google",
            arguments="query=python",  # type: ignore[arg-type]
            expected_result="done",
        )
        executor, _, _ = default_executor()
        report = executor.execute_plan(make_plan([bad_step]))
        self.assertFalse(report.success)
        self.assertEqual(
            report.step_results[0].error["code"], "invalid_arguments"
        )

    def test_empty_plan_raises_structured_error(self):
        executor, _, _ = default_executor()
        with self.assertRaises(ExecutionError) as ctx:
            executor.execute_plan(make_plan([]))
        self.assertEqual(ctx.exception.code, ExecutorErrorCode.EMPTY_PLAN)

    def test_non_plan_raises_structured_error(self):
        executor, _, _ = default_executor()
        with self.assertRaises(ExecutionError) as ctx:
            executor.execute_plan({"goal": "x", "steps": []})
        self.assertEqual(ctx.exception.code, ExecutorErrorCode.INVALID_PLAN)


class TestNoArbitraryCodeExecution(unittest.TestCase):
    def test_code_like_tool_name_is_just_an_unknown_tool(self):
        executor, _, _ = default_executor()
        for evil_name in ("eval", "exec", "__import__", "os.system",
                          "open_url; import os"):
            report = executor.execute_plan(
                make_plan([step("s1", evil_name, {})])
            )
            self.assertFalse(report.success, evil_name)
            self.assertEqual(
                report.step_results[0].error["code"], "tool_not_found"
            )

    def test_code_like_argument_is_treated_as_plain_data(self):
        recorded = []
        registry = ToolRegistry([RecordingTool(recorded)])
        payload = "__import__('os').system('echo hacked')"
        report = Executor(registry).execute_plan(
            make_plan([step("s1", "record_action", {"value": payload})])
        )
        self.assertTrue(report.success)
        self.assertEqual(recorded, [payload])  # stored literally, not run

    def test_executor_source_has_no_code_evaluation_calls(self):
        source = (
            Path(__file__).resolve().parents[1] / "afnan_ai" / "executor.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        } | {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for forbidden in ("eval", "exec", "compile", "__import__",
                          "system", "popen", "run", "Popen"):
            self.assertNotIn(forbidden, called)


class TestStateUpdates(unittest.TestCase):
    def test_successful_execution_updates_state(self):
        executor, _, _ = default_executor()
        state = AgentState.create("Open Chrome and search for Python")
        report = executor.execute_plan(
            make_plan(
                [
                    step("step_1", "open_application", {"application": "chrome"}),
                    step("step_2", "search_google", {"query": "Python"}),
                ],
                goal="Open Chrome and search for Python",
            ),
            state=state,
        )
        self.assertTrue(report.success)
        self.assertEqual(state.status, TaskStatus.COMPLETED)
        self.assertEqual(
            [s.name for s in state.completed_steps], ["step_1", "step_2"]
        )
        self.assertEqual(state.failed_steps, [])
        self.assertEqual(len(state.tool_results), 2)
        self.assertTrue(all(t.success for t in state.tool_results))
        self.assertEqual(
            [t.tool for t in state.tool_results],
            ["open_application", "search_google"],
        )
        self.assertTrue(any(o.source == "executor" for o in state.observations))
        self.assertEqual(report.task_id, state.task_id)
        # state remains fully serializable after execution
        restored = AgentState.from_json(state.to_json())
        self.assertEqual(restored.status, TaskStatus.COMPLETED)

    def test_state_created_when_not_provided(self):
        executor, _, _ = default_executor()
        plan = make_plan(
            [step("s1", "open_application", {"application": "chrome"})],
            task_id="task-99",
        )
        executor.execute_plan(plan)
        self.assertIsNotNone(executor.state)
        self.assertEqual(executor.state.task_id, "task-99")
        self.assertEqual(executor.state.goal, "Test goal")
        self.assertEqual(executor.state.status, TaskStatus.COMPLETED)

    def test_failed_execution_marks_task_failed_with_error(self):
        registry = ToolRegistry([FailingTool()])
        state = AgentState.create("Test goal")
        Executor(registry).execute_plan(
            make_plan([step("s1", "always_fails")]), state=state
        )
        self.assertEqual(state.status, TaskStatus.FAILED)
        self.assertIn("kaboom", state.error)
        failed_result = state.tool_results[0]
        self.assertFalse(failed_result.success)
        self.assertIn("kaboom", failed_result.error)


class TestAgentExecutorIntegration(unittest.TestCase):
    def make_agent(self, llm=None):
        adapter = FakeAdapter()
        agent = AfnanAgent(adapter=adapter, llm_provider=llm)
        agent.speak = lambda text: adapter.spoken.append(text)
        return agent, adapter

    def test_agent_execute_plan(self):
        agent, adapter = self.make_agent()
        plan = make_plan(
            [step("s1", "open_application", {"application": "chrome"})],
            goal="Open Chrome",
        )
        report = agent.execute_plan(plan)
        self.assertTrue(report.success)
        self.assertEqual(adapter.launched, ["chrome"])
        self.assertEqual(agent.state.status, TaskStatus.COMPLETED)
        self.assertIs(agent.executor.registry, agent.tools)

    def test_agent_plan_and_execute_end_to_end(self):
        llm = StubLLM(
            json.dumps(
                {
                    "goal": "Open Chrome",
                    "steps": [
                        {
                            "step_id": "step_1",
                            "description": "Open Chrome",
                            "tool_name": "open_application",
                            "arguments": {"application": "chrome"},
                            "expected_result": "Chrome is launched",
                        }
                    ],
                }
            )
        )
        agent, adapter = self.make_agent(llm=llm)
        report = agent.plan_and_execute("Open Chrome")
        self.assertTrue(report.success)
        self.assertEqual(adapter.launched, ["chrome"])
        self.assertEqual(agent.state.status, TaskStatus.COMPLETED)

    def test_agent_failed_execution_is_visible(self):
        agent, _ = self.make_agent()
        plan = make_plan([step("s1", "no_such_tool", {})], goal="Impossible")
        report = agent.execute_plan(plan)
        self.assertFalse(report.success)
        self.assertEqual(agent.state.status, TaskStatus.FAILED)

    def test_existing_voice_command_still_works(self):
        agent, adapter = self.make_agent(llm=StubLLM("ok"))
        agent.process_command("open chrome")
        self.assertEqual(adapter.launched, ["chrome"])


if __name__ == "__main__":
    unittest.main()

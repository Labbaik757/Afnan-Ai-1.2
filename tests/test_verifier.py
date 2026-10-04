import ast
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai.agent import AfnanAgent
from afnan_ai.executor import Executor, StepExecutionResult
from afnan_ai.llm.base import LLMProvider
from afnan_ai.platform.base import PlatformAdapter
from afnan_ai.planner import PlanStep, TaskPlan
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

    def __init__(self, reply="ok"):
        self.reply = reply

    def chat(self, messages):
        return self.reply


def step(step_id, tool_name, arguments=None, expected="", description=None):
    return PlanStep(
        step_id=step_id,
        description=description or f"Do {step_id}",
        tool_name=tool_name,
        arguments=arguments or {},
        expected_result=expected,
    )


def make_plan(steps, goal="Test goal", **kwargs):
    return TaskPlan(goal=goal, steps=steps, **kwargs)


def executed_result(step_id, tool_name, success=True, output=None, error=None, skipped=False):
    return StepExecutionResult(
        step_id=step_id,
        description=f"Do {step_id}",
        tool_name=tool_name,
        success=success,
        output=output,
        error=error,
        skipped=skipped,
    )


class TestSuccessfulVerification(unittest.TestCase):
    def test_verified_when_output_confirms_expected(self):
        s = step("step_1", "open_application", {"application": "chrome"},
                 expected="Chrome is launched")
        result = executed_result(
            "step_1", "open_application",
            output={"application": "chrome", "launched": True},
        )
        verification = Verifier().verify_step(s, result)
        self.assertEqual(verification.status, VerificationStatus.VERIFIED)
        self.assertTrue(verification.success)
        self.assertGreaterEqual(verification.confidence, 0.7)

    def test_verified_search_result(self):
        s = step("step_1", "search_google", {"query": "Python"},
                 expected="Google results for Python are open")
        result = executed_result(
            "step_1", "search_google",
            output={
                "query": "Python",
                "url": "https://www.google.com/search?q=Python",
                "opened": True,
            },
        )
        verification = Verifier().verify_step(s, result)
        self.assertEqual(verification.status, VerificationStatus.VERIFIED)

    def test_verify_plan_after_real_execution(self):
        adapter = FakeAdapter()
        registry = create_default_registry(adapter, opener=lambda url: True)
        plan = make_plan(
            [
                step("step_1", "open_application", {"application": "chrome"},
                     expected="Chrome is launched"),
                step("step_2", "search_google", {"query": "Python"},
                     expected="Google results for Python are open"),
            ],
            goal="Open Chrome and search for Python",
        )
        state = AgentState.create(plan.goal)
        report = Executor(registry).execute_plan(plan, state=state)
        self.assertTrue(report.success)

        verification = Verifier().verify_plan(plan, report, state=state)
        self.assertEqual(verification.status, VerificationStatus.VERIFIED)
        self.assertTrue(verification.success)
        self.assertEqual(
            verification.count(VerificationStatus.VERIFIED), 2
        )
        self.assertEqual(adapter.launched, ["chrome"])  # ran exactly once

    def test_verification_is_serializable(self):
        s = step("step_1", "open_application", {"application": "chrome"},
                 expected="Chrome is launched")
        v = Verifier().verify_step(
            s,
            executed_result("step_1", "open_application",
                            output={"application": "chrome", "launched": True}),
        )
        data = json.loads(v.to_json())
        self.assertEqual(data["status"], "verified")
        self.assertEqual(data["step_id"], "step_1")


class TestFailedVerification(unittest.TestCase):
    def test_failed_execution_is_failed_verification(self):
        registry = ToolRegistry([FailingTool()])
        plan = make_plan(
            [step("s1", "always_fails", expected="The tool succeeds")]
        )
        state = AgentState.create(plan.goal)
        report = Executor(registry).execute_plan(plan, state=state)
        self.assertFalse(report.success)

        v = Verifier().verify_step(plan.steps[0], report.step_results[0],
                                   state=state)
        self.assertEqual(v.status, VerificationStatus.FAILED)
        self.assertFalse(v.success)
        self.assertIn("kaboom", v.reason)
        # verifier must not flip the failed task into a success
        self.assertEqual(state.status, TaskStatus.FAILED)

    def test_skipped_step_is_failed_verification(self):
        s = step("s2", "record_action", {"value": "x"},
                 expected="The value is recorded")
        result = executed_result("s2", "record_action", success=False,
                                 skipped=True)
        v = Verifier().verify_step(s, result)
        self.assertEqual(v.status, VerificationStatus.FAILED)
        self.assertIn("skipped", v.reason)

    def test_wrong_application_is_a_contradiction(self):
        s = step("step_1", "open_application", {"application": "chrome"},
                 expected="Chrome is launched")
        result = executed_result(
            "step_1", "open_application",
            output={"application": "edge", "launched": True},
        )
        v = Verifier().verify_step(s, result)
        self.assertEqual(v.status, VerificationStatus.FAILED)

    def test_unrelated_output_does_not_confirm_expected(self):
        s = step("step_1", "open_application", {"application": "chrome"},
                 expected="Screenshot image file is saved")
        result = executed_result(
            "step_1", "open_application",
            output={"application": "chrome", "launched": True},
        )
        v = Verifier().verify_step(s, result)
        self.assertEqual(v.status, VerificationStatus.FAILED)

    def test_output_flag_false_is_failed(self):
        s = step("step_1", "open_url", {"url": "https://example.com"},
                 expected="Example page is opened")
        result = executed_result(
            "step_1", "open_url", output={"url": "https://example.com",
                                          "opened": False},
        )
        v = Verifier().verify_step(s, result)
        self.assertEqual(v.status, VerificationStatus.FAILED)


class TestUncertainVerification(unittest.TestCase):
    def test_no_execution_result_is_uncertain(self):
        s = step("s1", "open_application", {"application": "chrome"},
                 expected="Chrome is launched")
        v = Verifier().verify_step(s, None)
        self.assertEqual(v.status, VerificationStatus.UNCERTAIN)
        self.assertFalse(v.success)

    def test_missing_expected_result_is_uncertain(self):
        s = step("s1", "open_application", {"application": "chrome"},
                 expected="")
        result = executed_result(
            "s1", "open_application",
            output={"application": "chrome", "launched": True},
        )
        v = Verifier().verify_step(s, result)
        self.assertEqual(v.status, VerificationStatus.UNCERTAIN)

    def test_vague_expected_result_is_uncertain(self):
        s = step("s1", "open_application", {"application": "chrome"},
                 expected="done")
        result = executed_result(
            "s1", "open_application",
            output={"application": "chrome", "launched": True},
        )
        v = Verifier().verify_step(s, result)
        self.assertEqual(v.status, VerificationStatus.UNCERTAIN)

    def test_success_without_output_is_uncertain(self):
        s = step("s1", "open_application", {"application": "chrome"},
                 expected="Chrome is launched")
        result = executed_result("s1", "open_application", output=None)
        v = Verifier().verify_step(s, result)
        self.assertEqual(v.status, VerificationStatus.UNCERTAIN)

    def test_partial_match_is_uncertain(self):
        s = step("s1", "open_application", {"application": "chrome"},
                 expected="Chrome browser window is launched maximized "
                          "on the main monitor")
        result = executed_result(
            "s1", "open_application",
            output={"application": "chrome", "launched": True},
        )
        v = Verifier().verify_step(s, result)
        self.assertEqual(v.status, VerificationStatus.UNCERTAIN)
        self.assertIn("partly", v.reason)

    def test_conflicting_state_evidence_is_uncertain(self):
        s = step("s1", "open_application", {"application": "chrome"},
                 expected="Chrome is launched")
        state = AgentState.create("Test goal")
        state.start_step("s1")
        state.fail_step("s1", "earlier attempt failed")
        result = executed_result(
            "s1", "open_application",
            output={"application": "chrome", "launched": True},
        )
        v = Verifier().verify_step(s, result, state=state, record=False)
        self.assertEqual(v.status, VerificationStatus.UNCERTAIN)

    def test_mixed_plan_is_uncertain_overall(self):
        plan = make_plan(
            [
                step("s1", "open_application", {"application": "chrome"},
                     expected="Chrome is launched"),
                step("s2", "open_application", {"application": "chrome"},
                     expected="done"),
            ]
        )
        results = [
            executed_result("s1", "open_application",
                            output={"application": "chrome", "launched": True}),
            executed_result("s2", "open_application",
                            output={"application": "chrome", "launched": True}),
        ]
        from afnan_ai.executor import ExecutionReport
        report = ExecutionReport(plan_id=plan.plan_id, goal=plan.goal,
                                   status="completed", step_results=results)
        v = Verifier().verify_plan(plan, report)
        self.assertEqual(v.status, VerificationStatus.UNCERTAIN)
        self.assertFalse(v.success)


class TestNoReExecutionAndStateRecording(unittest.TestCase):
    def test_verifier_never_re_executes_tools(self):
        recorded = []
        registry = ToolRegistry([RecordingTool(recorded)])
        plan = make_plan(
            [step("s1", "record_action", {"value": "hello"},
                  expected="hello value is recorded and opened")]
        )
        state = AgentState.create(plan.goal)
        report = Executor(registry).execute_plan(plan, state=state)
        self.assertEqual(recorded, ["hello"])

        verifier = Verifier()
        verifier.verify_plan(plan, report, state=state)
        verifier.verify_step(plan.steps[0], report.step_results[0], state=state)
        verifier.verify_plan(plan, report, state=state)
        # still exactly one execution — verification only analyzed
        self.assertEqual(recorded, ["hello"])

    def test_verification_recorded_in_agent_state(self):
        plan = make_plan(
            [step("step_1", "open_application", {"application": "chrome"},
                  expected="Chrome is launched")]
        )
        state = AgentState.create(plan.goal)
        tools_before = len(state.tool_results)
        v = Verifier().verify_step(
            plan.steps[0],
            executed_result("step_1", "open_application",
                            output={"application": "chrome", "launched": True}),
            state=state,
        )
        self.assertEqual(v.status, VerificationStatus.VERIFIED)
        entries = state.metadata["verifications"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["status"], "verified")
        self.assertEqual(entries[0]["step_id"], "step_1")
        self.assertTrue(
            any(o.source == "verifier" for o in state.observations)
        )
        # recording a judgement is not a tool execution
        self.assertEqual(len(state.tool_results), tools_before)
        # state still serializes with the verification inside
        restored = AgentState.from_json(state.to_json())
        self.assertEqual(
            restored.metadata["verifications"][0]["status"], "verified"
        )

    def test_verify_from_state_evidence_alone(self):
        # no execution result object — AgentState itself is the evidence
        adapter = FakeAdapter()
        registry = create_default_registry(adapter, opener=lambda url: True)
        plan = make_plan(
            [step("step_1", "open_application", {"application": "chrome"},
                  expected="Chrome is launched")]
        )
        state = AgentState.create(plan.goal)
        Executor(registry).execute_plan(plan, state=state)
        v = Verifier().verify_step(plan.steps[0], None, state=state)
        self.assertEqual(v.status, VerificationStatus.VERIFIED)

    def test_verifier_source_has_no_execution_calls(self):
        source = (
            Path(__file__).resolve().parents[1] / "afnan_ai" / "verifier.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        called_attrs = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for forbidden in ("execute", "execute_or_raise", "run",
                          "eval", "exec", "compile", "__import__"):
            self.assertNotIn(forbidden, called_attrs)
            self.assertNotIn(forbidden, called_names)


class TestAgentVerifierIntegration(unittest.TestCase):
    def make_agent(self):
        adapter = FakeAdapter()
        agent = AfnanAgent(adapter=adapter, llm_provider=StubLLM())
        agent.speak = lambda text: adapter.spoken.append(text)
        return agent, adapter

    def test_agent_execute_and_verify(self):
        agent, adapter = self.make_agent()
        plan = make_plan(
            [step("step_1", "open_application", {"application": "chrome"},
                  expected="Chrome is launched")],
            goal="Open Chrome",
        )
        report, verification = agent.execute_and_verify(plan)
        self.assertTrue(report.success)
        self.assertEqual(verification.status, VerificationStatus.VERIFIED)
        self.assertEqual(adapter.launched, ["chrome"])  # executed once only
        self.assertEqual(
            agent.state.metadata["verifications"][0]["status"], "verified"
        )
        self.assertIs(agent.verifier.__class__, Verifier)

    def test_agent_verify_failed_execution(self):
        agent, _ = self.make_agent()
        plan = make_plan(
            [step("s1", "no_such_tool", expected="Something happens")],
            goal="Impossible",
        )
        report, verification = agent.execute_and_verify(plan)
        self.assertFalse(report.success)
        self.assertEqual(verification.status, VerificationStatus.FAILED)
        self.assertEqual(agent.state.status, TaskStatus.FAILED)

    def test_existing_voice_command_still_works(self):
        agent, adapter = self.make_agent()
        agent.process_command("open chrome")
        self.assertEqual(adapter.launched, ["chrome"])


if __name__ == "__main__":
    unittest.main()

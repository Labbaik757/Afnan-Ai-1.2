"""Tests for afnan_ai.verify — the verifier-repair loop."""

import unittest

from afnan_ai.tools.base import Tool
from afnan_ai.tools.registry import ToolRegistry
from afnan_ai.verify import (
    MAX_PLAN_STEPS,
    AnswerVerifier,
    PlanVerifier,
    RepairOutcome,
    VerificationResult,
    verify_then_repair,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class SearchTool(Tool):
    name = "search_web"
    description = "search the web"
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def run(self, arguments):
        return {"ok": True}


class OpenTool(Tool):
    name = "open_page"
    description = "open a page"
    input_schema = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }

    def run(self, arguments):
        return {"ok": True}


class FillTool(Tool):
    name = "fill_field"
    description = "fill a form field"
    input_schema = {
        "type": "object",
        "properties": {
            "field": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["field", "value"],
    }

    def run(self, arguments):
        return {"ok": True}


class SubmitTool(Tool):
    name = "submit_form"
    description = "submit the form"
    input_schema = {"type": "object", "properties": {},
                    "required": []}

    def run(self, arguments):
        return {"ok": True}


def make_registry():
    return ToolRegistry([SearchTool(), OpenTool(), FillTool(), SubmitTool()])


class StubLLM:
    """Returns canned plan JSON on successive generate() calls."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.calls = 0

    def generate(self, prompt):
        self.prompts.append(prompt)
        self.calls += 1
        if self.calls <= len(self.replies):
            return self.replies[self.calls - 1]
        return self.replies[-1]


GOOD_PLAN = [
    {"step_id": "1", "description": "search for cats",
     "tool_name": "search_web", "arguments": {"query": "cats"}},
    {"step_id": "2", "description": "open the first result",
     "tool_name": "open_page", "arguments": {"url": "https://x.test"}},
]


# ---------------------------------------------------------------------------
# PlanVerifier
# ---------------------------------------------------------------------------


class TestPlanVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = PlanVerifier(make_registry())

    def test_valid_plan_passes(self):
        result = self.verifier.verify_plan(GOOD_PLAN, "find cats")
        self.assertTrue(result.passed)
        self.assertEqual(result.issues, [])
        self.assertEqual(result.checked_steps, 2)

    def test_hallucinated_tool_detected(self):
        plan = [{"step_id": "1", "description": "do magic",
                 "tool_name": "teleport_somewhere",
                 "arguments": {}}]
        result = self.verifier.verify_plan(plan, "teleport")
        self.assertFalse(result.passed)
        self.assertTrue(any("teleport_somewhere" in i for i in result.issues))
        self.assertTrue(any("hallucinated" in i for i in result.issues))
        self.assertTrue(result.retryable)

    def test_hallucinated_tool_suggests_close_match(self):
        plan = [{"step_id": "1", "description": "search",
                 "tool_name": "search_webs",  # typo of search_web
                 "arguments": {"query": "x"}}]
        result = self.verifier.verify_plan(plan, "search")
        self.assertFalse(result.passed)
        self.assertTrue(any("search_web" in i for i in result.issues))

    def test_missing_tool_name(self):
        plan = [{"step_id": "1", "description": "no tool here",
                 "arguments": {}}]
        result = self.verifier.verify_plan(plan, "goal")
        self.assertFalse(result.passed)
        self.assertTrue(any("missing tool name" in i for i in result.issues))

    def test_missing_required_params(self):
        plan = [{"step_id": "1", "description": "search with no query",
                 "tool_name": "search_web", "arguments": {}}]
        result = self.verifier.verify_plan(plan, "search")
        self.assertFalse(result.passed)
        self.assertTrue(any("query" in i for i in result.issues))

    def test_wrong_argument_type_flagged(self):
        plan = [{"step_id": "1", "description": "search",
                 "tool_name": "search_web", "arguments": "not-a-dict"}]
        result = self.verifier.verify_plan(plan, "search")
        self.assertFalse(result.passed)
        self.assertTrue(any("must be an object" in i for i in result.issues))

    def test_submit_before_fill_flagged(self):
        plan = [
            {"step_id": "1", "description": "submit the login form",
             "tool_name": "submit_form", "arguments": {}},
            {"step_id": "2", "description": "fill username",
             "tool_name": "fill_field",
             "arguments": {"field": "user", "value": "boss"}},
        ]
        result = self.verifier.verify_plan(plan, "log in")
        self.assertFalse(result.passed)
        self.assertTrue(any("before any" in i and "fill" in i
                            for i in result.issues))

    def test_fill_then_submit_passes_ordering(self):
        plan = [
            {"step_id": "1", "description": "fill username",
             "tool_name": "fill_field",
             "arguments": {"field": "user", "value": "boss"}},
            {"step_id": "2", "description": "submit the form",
             "tool_name": "submit_form", "arguments": {}},
        ]
        result = self.verifier.verify_plan(plan, "log in")
        self.assertTrue(result.passed)

    def test_too_many_steps_rejected(self):
        plan = [{"step_id": str(i), "description": f"step {i}",
                 "tool_name": "search_web", "arguments": {"query": "x"}}
                for i in range(MAX_PLAN_STEPS + 5)]
        result = self.verifier.verify_plan(plan, "huge")
        self.assertFalse(result.passed)
        self.assertTrue(any("exceeding the sanity bound" in i
                            for i in result.issues))
        self.assertTrue(result.retryable)

    def test_empty_plan_rejected_but_retryable(self):
        result = self.verifier.verify_plan([], "do something")
        self.assertFalse(result.passed)
        self.assertTrue(result.retryable)

    def test_non_list_plan_rejected(self):
        result = self.verifier.verify_plan("just do it", "goal")
        self.assertFalse(result.passed)
        self.assertTrue(any("not a list" in i for i in result.issues))

    def test_duplicate_consecutive_steps_flagged(self):
        plan = [
            {"step_id": "1", "description": "search",
             "tool_name": "search_web", "arguments": {"query": "x"}},
            {"step_id": "2", "description": "search again",
             "tool_name": "search_web", "arguments": {"query": "x"}},
        ]
        result = self.verifier.verify_plan(plan, "search")
        self.assertFalse(result.passed)
        self.assertTrue(any("probable loop" in i for i in result.issues))

    def test_empty_registry_not_retryable(self):
        verifier = PlanVerifier(ToolRegistry())
        result = verifier.verify_plan(GOOD_PLAN, "find cats")
        self.assertFalse(result.passed)
        self.assertFalse(result.retryable)

    def test_accepts_alternate_step_keys(self):
        # "tool" / "params" aliases for "tool_name" / "arguments"
        plan = [{"id": "1", "desc": "search",
                 "tool": "search_web", "params": {"query": "cats"}}]
        result = self.verifier.verify_plan(plan, "find cats")
        self.assertTrue(result.passed)

    def test_accepts_taskplan_object(self):
        from afnan_ai.planner import TaskPlan, PlanStep
        plan = TaskPlan(
            goal="find cats",
            steps=[PlanStep(step_id="1", description="search",
                            tool_name="search_web",
                            arguments={"query": "cats"})],
        )
        result = self.verifier.verify_plan(plan, "find cats")
        self.assertTrue(result.passed)

    def test_feedback_text_lists_issues(self):
        bad = [{"step_id": "1", "description": "x",
                "tool_name": "nope_tool", "arguments": {}}]
        result = self.verifier.verify_plan(bad, "goal")
        feedback = self.verifier.feedback_text(result)
        self.assertIn("nope_tool", feedback)
        self.assertIn("avoid", feedback.lower())


# ---------------------------------------------------------------------------
# verify_then_repair
# ---------------------------------------------------------------------------

import json  # noqa: E402


class TestVerifyThenRepair(unittest.TestCase):
    def setUp(self):
        self.verifier = PlanVerifier(make_registry())

    def test_first_attempt_passes_no_repair(self):
        llm = StubLLM([json.dumps(GOOD_PLAN)])
        outcome = verify_then_repair(llm, self.verifier, "find cats")
        self.assertTrue(outcome.result.passed)
        self.assertEqual(outcome.attempts, 1)
        self.assertFalse(outcome.repaired)
        self.assertEqual(outcome.issues, [])

    def test_repair_loop_fixes_hallucinated_tool(self):
        bad = json.dumps([
            {"step_id": "1", "description": "search",
             "tool_name": "search_webs", "arguments": {"query": "cats"}},
        ])
        llm = StubLLM([bad, json.dumps(GOOD_PLAN)])
        outcome = verify_then_repair(llm, self.verifier, "find cats")
        self.assertTrue(outcome.result.passed)
        self.assertTrue(outcome.repaired)
        self.assertEqual(outcome.attempts, 2)
        # The repair prompt must carry the verifier's feedback.
        self.assertTrue(any("search_webs" in p for p in llm.prompts[1:]))

    def test_max_attempts_respected(self):
        bad = json.dumps([
            {"step_id": "1", "description": "bad",
             "tool_name": "nope_tool", "arguments": {}},
        ])
        llm = StubLLM([bad])  # always returns the bad plan
        outcome = verify_then_repair(llm, self.verifier, "goal",
                                     max_attempts=3)
        self.assertFalse(outcome.result.passed)
        self.assertEqual(outcome.attempts, 3)
        self.assertFalse(outcome.repaired)
        self.assertTrue(outcome.issues)  # best attempt's issues returned
        self.assertEqual(llm.calls, 3)

    def test_unparseable_llm_output_retries_then_gives_up(self):
        llm = StubLLM(["sorry, I cannot plan that."])
        outcome = verify_then_repair(llm, self.verifier, "goal",
                                     max_attempts=2)
        self.assertFalse(outcome.result.passed)
        self.assertEqual(outcome.attempts, 2)

    def test_non_retryable_stops_early(self):
        verifier = PlanVerifier(ToolRegistry())  # empty registry
        llm = StubLLM([json.dumps(GOOD_PLAN)])
        outcome = verify_then_repair(llm, verifier, "goal", max_attempts=3)
        self.assertFalse(outcome.result.passed)
        self.assertEqual(outcome.attempts, 1)  # stopped, not retried
        self.assertEqual(llm.calls, 1)

    def test_custom_plan_fn_used(self):
        def my_plan_fn(llm, goal, tool_names, feedback):
            return GOOD_PLAN

        outcome = verify_then_repair(None, self.verifier, "find cats",
                                     plan_fn=my_plan_fn)
        self.assertTrue(outcome.result.passed)

    def test_best_attempt_returned_when_all_fail(self):
        # First attempt: hallucinated tool (invalid tool, 1 issue).
        # Second: valid tool but missing params (1 issue) -> tie broken
        # by valid-tool count, so the better plan is kept.
        worse = json.dumps([
            {"step_id": "1", "description": "x",
             "tool_name": "nope_tool", "arguments": {}},
        ])
        better = json.dumps([
            {"step_id": "1", "description": "x",
             "tool_name": "search_web", "arguments": {}},
        ])
        llm = StubLLM([worse, better])
        outcome = verify_then_repair(llm, self.verifier, "goal",
                                     max_attempts=2)
        self.assertFalse(outcome.result.passed)
        self.assertEqual(len(outcome.issues), 1)
        self.assertEqual(outcome.plan[0]["tool_name"], "search_web")


# ---------------------------------------------------------------------------
# AnswerVerifier
# ---------------------------------------------------------------------------


class TestAnswerVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = AnswerVerifier()

    def test_good_answer_passes(self):
        self.assertTrue(self.verifier.verify_answer(
            "What is the capital of France?",
            "The capital of France is Paris."))

    def test_empty_answer_fails(self):
        self.assertFalse(self.verifier.verify_answer("hello?", ""))
        self.assertFalse(self.verifier.verify_answer("hello?", "   "))

    def test_dodge_answer_fails(self):
        self.assertFalse(self.verifier.verify_answer(
            "What is 2+2?", "I don't know the answer to that."))
        self.assertFalse(self.verifier.verify_answer(
            "hello", "As an AI, I cannot help with that."))

    def test_unrelated_answer_fails(self):
        self.assertFalse(self.verifier.verify_answer(
            "What is the capital of France?",
            "Bananas are a great source of potassium."))

    def test_short_greeting_passes(self):
        # "say hi" has no significant keywords; keyword check is skipped
        self.assertTrue(self.verifier.verify_answer("say hi", "Hi there!"))

    def test_reasons_explain_failure(self):
        ok, reasons = self.verifier.verify_answer_with_reasons("hi?", "")
        self.assertFalse(ok)
        self.assertTrue(reasons)

    def test_llm_judge_can_reject(self):
        judge = StubLLM(["NO"])
        verifier = AnswerVerifier(llm=judge)
        self.assertFalse(verifier.verify_answer(
            "What is the capital of France?",
            "The capital of France is Paris."))

    def test_llm_judge_can_confirm(self):
        judge = StubLLM(["YES"])
        verifier = AnswerVerifier(llm=judge)
        self.assertTrue(verifier.verify_answer(
            "What is the capital of France?",
            "The capital of France is Paris."))

    def test_repair_answer_without_llm(self):
        verifier = AnswerVerifier(llm=None)
        answer, ok = verifier.repair_answer("hi?", "")
        self.assertEqual(answer, "")
        self.assertFalse(ok)

    def test_repair_answer_regenerates(self):
        # Initial answer fails heuristics (empty); the LLM regenerates it,
        # then the judge confirms the new one.
        class FixingLLM:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt):
                self.calls += 1
                if prompt.startswith("Does the following"):
                    return "YES"  # judge prompt
                return "The capital of France is Paris."  # repair prompt

        llm = FixingLLM()
        verifier = AnswerVerifier(llm=llm)
        answer, ok = verifier.repair_answer(
            "What is the capital of France?", "")
        self.assertTrue(ok)
        self.assertIn("Paris", answer)
        self.assertGreaterEqual(llm.calls, 2)  # regen + judge


if __name__ == "__main__":
    unittest.main()

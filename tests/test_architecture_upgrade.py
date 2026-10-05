"""Phase-3 architecture upgrade tests.

Covers the observation-driven agent loop (re-decided bounded
batches, completion signal, step/LLM/replan/repeat/time limits,
in-loop recovery), the modern Playwright ARIA accessibility API,
planner parse-repair, semantic low-confidence approval gating,
upload completion verification and dialog observation.  All
against fake backends and queued LLMs — no real browser, network
or model.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from afnan_ai.browser.accessibility import (
    normalize_snapshot,
    parse_aria_snapshot,
)
from afnan_ai.browser.backend import PlaywrightBackend
from afnan_ai.browser.base import BrowserErrorCode, BrowserException
from afnan_ai.browser.security import ApprovalGate
from afnan_ai.executor import Executor
from afnan_ai.llm.base import LLMProvider
from afnan_ai.orchestrator import Agent, OrchestrationStatus
from afnan_ai.planner import Planner, PlanningError
from afnan_ai.recovery import RecoveryManager
from afnan_ai.tools.base import Tool, ToolExecutionError
from afnan_ai.tools.registry import ToolRegistry
from afnan_ai.verifier import Verifier

from test_browser_advanced import (
    DynamicBackend,
    QueueLLM,
    el,
    make_controller,
    plan_json,
    step,
)
from test_browser_operations import OpsBackend
from test_browser_workflow import _serp_page, make_agent

COMPLETE = '{"goal": "g", "complete": true, "steps": []}'


class EchoTool(Tool):
    name = "echo"
    description = "Echo a value."
    input_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def __init__(self):
        self.calls: list[str] = []

    def run(self, arguments):
        self.calls.append(arguments["value"])
        return f"echoed {arguments['value']}"


class BoomTool(Tool):
    name = "boom"
    description = "Always fails."
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def run(self, arguments):
        raise ToolExecutionError("kaboom", tool=self.name)


def make_orchestrator(replies, tools):
    llm = QueueLLM(replies)
    registry = ToolRegistry(tools)
    planner = Planner(llm, registry)
    return Agent(
        planner=planner,
        executor=Executor(registry),
        verifier=Verifier(),
        recovery=RecoveryManager(planner),
    ), llm


def echo_step(step_id, value):
    return step(step_id, "echo", {"value": value}, f"echoed {value}")


# ----------------------------------------------------------------------
# Modern ARIA accessibility API
# ----------------------------------------------------------------------

ARIA_YAML = (
    '- heading "Welcome" [level=1]\n'
    '- button "Sign in"\n'
    '- textbox "Email"\n'
    "- list:\n"
    '  - listitem "First item"\n'
    '  - listitem "Second item"\n'
)


class TestAriaSnapshot(unittest.TestCase):
    def test_parse_and_normalize(self):
        tree = parse_aria_snapshot(ARIA_YAML)
        nodes = normalize_snapshot(tree)
        by_role = {(n["role"], n["name"]) for n in nodes}
        self.assertIn(("heading", "Welcome"), by_role)
        self.assertIn(("button", "Sign in"), by_role)
        self.assertIn(("textbox", "Email"), by_role)
        self.assertIn(("listitem", "Second item"), by_role)
        heading = next(n for n in nodes if n["role"] == "heading")
        self.assertEqual(heading["level"], 1)

    def test_backend_prefers_aria_snapshot(self):
        class AriaPage:
            def aria_snapshot(self):
                return ARIA_YAML

        tree = PlaywrightBackend().accessibility_snapshot(AriaPage())
        roles = [c["role"] for c in tree["children"]]
        self.assertIn("button", roles)
        self.assertIn("heading", [c["role"] for c in tree["children"]])

    def test_backend_legacy_fallback_still_works(self):
        class LegacyPage:
            class accessibility:
                @staticmethod
                def snapshot():
                    return {
                        "role": "button", "name": "Old",
                        "children": [],
                    }

        tree = PlaywrightBackend().accessibility_snapshot(LegacyPage())
        self.assertEqual(tree["name"], "Old")

    def test_controller_tree_from_aria(self):
        class AriaBackend(DynamicBackend):
            def accessibility_snapshot(self, handle):
                return parse_aria_snapshot(ARIA_YAML)

        backend = AriaBackend()
        backend.add_page(
            "https://app.example/", title="App", text="Welcome",
            elements=[],
        )
        controller, _ = make_controller(backend)
        tree = controller.accessibility_tree()
        self.assertEqual(tree["source"], "accessibility")
        names = {n["name"] for n in tree["nodes"]}
        self.assertIn("Sign in", names)
        self.assertIn("Welcome", names)


# ----------------------------------------------------------------------
# Planner completion signal + bounded parse repair
# ----------------------------------------------------------------------

class TestPlannerUpgrade(unittest.TestCase):
    def make_planner(self, replies, **kwargs):
        echo = EchoTool()
        planner = Planner(
            QueueLLM(replies), ToolRegistry([echo]), **kwargs
        )
        return planner

    def test_invalid_output_repaired_once(self):
        planner = self.make_planner([
            "this is not json",
            plan_json("g", [echo_step("s1", "a")]),
        ])
        plan = planner.plan("g")
        self.assertEqual(len(plan.steps), 1)

    def test_repair_is_bounded(self):
        planner = self.make_planner(["junk", "more junk"])
        with self.assertRaises(PlanningError):
            planner.plan("g")

    def test_repair_can_be_disabled(self):
        planner = self.make_planner(["junk"], max_parse_retries=0)
        with self.assertRaises(PlanningError):
            planner.plan("g")

    def test_completion_signal_parses(self):
        planner = self.make_planner([COMPLETE])
        plan = planner.plan("g")
        self.assertEqual(plan.steps, [])
        self.assertTrue(plan.metadata.get("task_complete"))


# ----------------------------------------------------------------------
# Observation-driven agent loop
# ----------------------------------------------------------------------

class TestAgentLoop(unittest.TestCase):
    def test_multi_cycle_completion(self):
        echo = EchoTool()
        agent, llm = make_orchestrator([
            plan_json("g", [echo_step("s1", "a")]),
            plan_json("g", [echo_step("s2", "b")]),
            COMPLETE,
        ], [echo])
        result = agent.run_loop("g")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(echo.calls, ["a", "b"])
        self.assertEqual(result.iterations, 2)

    def test_batch_remainder_is_not_followed_blindly(self):
        echo = EchoTool()
        agent, llm = make_orchestrator([
            plan_json("g", [
                echo_step("s1", "v1"), echo_step("s2", "v2"),
                echo_step("s3", "v3"), echo_step("s4", "v4"),
                echo_step("s5", "v5"),
            ]),
            COMPLETE,
        ], [echo])
        result = agent.run_loop("g", batch_limit=2)
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        # Only the first bounded batch ran; the Planner then
        # re-decided from fresh state and reported completion.
        self.assertEqual(echo.calls, ["v1", "v2"])

    def test_step_limit_stops_safely(self):
        echo = EchoTool()
        replies = [
            plan_json("g", [echo_step(f"s{i}", f"v{i}")])
            for i in range(6)
        ]
        agent, _ = make_orchestrator(replies, [echo])
        result = agent.run_loop("g", max_steps=3)
        self.assertEqual(
            result.status, OrchestrationStatus.MAX_ITERATIONS_EXCEEDED
        )
        self.assertEqual(
            result.error["code"], "step_limit_exceeded"
        )
        self.assertEqual(len(echo.calls), 3)

    def test_repeated_identical_action_stops(self):
        echo = EchoTool()
        replies = [
            plan_json("g", [echo_step("s1", "same")])
            for _ in range(5)
        ]
        agent, _ = make_orchestrator(replies, [echo])
        result = agent.run_loop("g", max_repeated_actions=2)
        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        self.assertEqual(result.error["code"], "repeated_action")
        self.assertEqual(echo.calls, ["same", "same"])

    def test_llm_call_limit(self):
        echo = EchoTool()
        replies = [
            plan_json("g", [echo_step(f"s{i}", f"v{i}")])
            for i in range(6)
        ]
        agent, _ = make_orchestrator(replies, [echo])
        result = agent.run_loop("g", max_llm_calls=2)
        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        self.assertEqual(result.error["code"], "llm_call_limit")
        self.assertEqual(len(echo.calls), 2)

    def test_replan_limit(self):
        echo = EchoTool()
        replies = [
            plan_json("g", [echo_step(f"s{i}", f"v{i}")])
            for i in range(6)
        ]
        agent, _ = make_orchestrator(replies, [echo])
        result = agent.run_loop("g", max_replans=1)
        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        self.assertEqual(result.error["code"], "replan_limit")

    def test_recovery_inside_loop(self):
        echo = EchoTool()
        boom = BoomTool()
        agent, _ = make_orchestrator([
            plan_json("g", [
                step("s1", "boom", {}, "boom worked")
            ]),
            plan_json("g", [echo_step("r1", "recovered")]),
            COMPLETE,
        ], [echo, boom])
        result = agent.run_loop("g")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(echo.calls, ["recovered"])
        self.assertEqual(len(result.recovery_attempts), 1)


# ----------------------------------------------------------------------
# Semantic confidence gating
# ----------------------------------------------------------------------

class TestSemanticGating(unittest.TestCase):
    def make_semantic_controller(self, approver=None):
        backend = DynamicBackend()
        backend.add_page(
            "https://app.example/home", title="Home",
            text="Welcome home",
            elements=[
                el("a", "Login",
                   {"href": "https://app.example/account"}),
            ],
        )
        backend.add_page(
            "https://app.example/account", title="Account",
            text="Your account", elements=[],
        )
        controller, _ = make_controller(backend)
        if approver is not None:
            controller.approval_gate = ApprovalGate(approver=approver)
        controller.navigate("https://app.example/home")
        return controller

    def test_uncertain_match_requires_approval(self):
        controller = self.make_semantic_controller()
        found = controller.find_semantic("Login button")
        best = found["matches"][0]
        self.assertEqual(best["tier"], "uncertain")
        self.assertIsNotNone(best["element_ref"])
        with self.assertRaises(BrowserException) as ctx:
            controller.click({"ref": best["element_ref"]})
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.APPROVAL_REQUIRED
        )
        # The click never happened: still on the home page.
        self.assertTrue(
            controller.current_page()["url"].endswith("/home")
        )

    def test_uncertain_match_with_approver_proceeds(self):
        controller = self.make_semantic_controller(
            approver=lambda request: True
        )
        found = controller.find_semantic("Login button")
        controller.click({"ref": found["matches"][0]["element_ref"]})
        self.assertTrue(
            controller.current_page()["url"].endswith("/account")
        )

    def test_confident_match_needs_no_approval(self):
        backend = DynamicBackend()
        backend.add_page(
            "https://app.example/home", title="Home", text="Welcome",
            elements=[
                el("button", "Login",
                   {"href": "https://app.example/account"}),
            ],
        )
        backend.add_page(
            "https://app.example/account", title="Account",
            text="Your account", elements=[],
        )
        controller, _ = make_controller(backend)
        controller.navigate("https://app.example/home")
        found = controller.find_semantic("Login button")
        best = found["matches"][0]
        self.assertEqual(best["tier"], "actionable")
        controller.click({"ref": best["element_ref"]})

    def test_low_confidence_hides_ref(self):
        controller = self.make_semantic_controller()
        found = controller.find_semantic("Zebra crossing settings")
        self.assertTrue(found["matches"])
        self.assertIsNone(found["matches"][0]["element_ref"])

    def test_find_semantic_tool_delegates(self):
        agent, backend = make_agent(
            [], backend=OpsBackend()
        )
        backend.add_page(
            "https://app.example/home", title="Home", text="Welcome",
            elements=[el("button", "Login")],
        )
        agent.browser.navigate("https://app.example/home")
        result = agent.tools.execute(
            "browser_find_semantic", {"description": "Login button"}
        )
        self.assertTrue(result.success)
        self.assertEqual(result.output["best_confidence"], 0.99)


# ----------------------------------------------------------------------
# Upload verification + dialog observation
# ----------------------------------------------------------------------

class UploadBackend(DynamicBackend):
    shown_name = None

    def set_input_files(self, handle, el_handle, path, timeout_ms):
        page = self._page(handle)
        name = self.shown_name or Path(path).name
        page["elements"][el_handle[1]]["value"] = (
            f"C:\\fakepath\\{name}"
        )


class TestUploadVerification(unittest.TestCase):
    def make_upload(self, shown_name=None):
        backend = UploadBackend()
        backend.shown_name = shown_name
        backend.add_page(
            "https://app.example/upload", title="Upload",
            text="Upload a file",
            elements=[
                el("input", "", {"type": "file", "id": "f"}),
            ],
        )
        controller, _ = make_controller(
            backend, approval_gate=ApprovalGate(
                approver=lambda request: True
            ),
        )
        controller.navigate("https://app.example/upload")
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False
        )
        tmp.write("hello")
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return controller, tmp.name

    def test_verified_upload(self):
        controller, path = self.make_upload()
        result = controller.upload_file(
            {"selector": "input"}, path
        )
        self.assertTrue(result["upload_verified"])

    def test_mismatched_upload_fails(self):
        controller, path = self.make_upload(shown_name="other.txt")
        with self.assertRaises(BrowserException) as ctx:
            controller.upload_file({"selector": "input"}, path)
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.OPERATION_FAILED
        )


class TestDialogObservation(unittest.TestCase):
    def test_dialogs_surface_in_observation(self):
        class DialogBackend(DynamicBackend):
            def dialogs(self, handle):
                return [{
                    "type": "alert", "message": "Saved",
                    "action": "dismissed",
                }]

        backend = DialogBackend()
        backend.add_page(
            "https://app.example/", title="App", text="Hi",
            elements=[],
        )
        controller, _ = make_controller(backend)
        observed = controller.observe()
        self.assertEqual(observed["dialogs"][0]["type"], "alert")
        self.assertEqual(observed["dialogs"][0]["message"], "Saved")


# ----------------------------------------------------------------------
# Loop mode through the unified browser workflow
# ----------------------------------------------------------------------

class TestLoopBrowserWorkflow(unittest.TestCase):
    def test_loop_search_open_extract(self):
        backend = OpsBackend()
        _serp_page(backend)
        agent, _ = make_agent([
            plan_json("Research the Python guide", [
                step("s1", "browser_search",
                     {"query": "python guide"},
                     "Search results Python Guide"),
            ]),
            plan_json("Research the Python guide", [
                step("s2", "browser_open_result", {"index": 0},
                     "Python programming language readable syntax"),
            ]),
            COMPLETE,
        ], backend=backend)
        outcome = agent.run_browser_goal(
            "Research the Python guide", loop=True
        )
        self.assertEqual(outcome.status, "completed")
        self.assertIn("readable syntax", outcome.answer)

    def test_loop_recovers_from_wrong_route(self):
        backend = OpsBackend()
        backend.add_page(
            "https://shop.example/right", title="Right Page",
            text="The correct destination content",
            elements=[],
        )
        agent, _ = make_agent([
            plan_json("Open the right page", [
                step("s1", "browser_click",
                     {"selector": ".missing"},
                     "Correct destination opened"),
            ]),
            plan_json("Open the right page", [
                step("r1", "browser_navigate",
                     {"url": "https://shop.example/right"},
                     "Right Page correct destination content"),
            ]),
            COMPLETE,
        ], backend=backend)
        outcome = agent.run_browser_goal(
            "Open the right page", loop=True
        )
        self.assertEqual(outcome.status, "completed")
        self.assertGreaterEqual(outcome.recovery_attempts, 1)


if __name__ == "__main__":
    unittest.main()

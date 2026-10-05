"""Browser reliability layer tests.

Covers: verification grounded in actual page state (not just
tool success), structured recovery strategies for browser
failures, popup/new-tab handling, and end-to-end browser tasks
through Agent orchestration: success, failed action →
recovery/replanning (never blindly repeating the failed action),
dynamic content, and final verification.
"""
import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai.agent import AfnanAgent
from afnan_ai.browser import BrowserController
from afnan_ai.browser.backend import BrowserBackend
from afnan_ai.browser.reliability import BrowserReliability
from afnan_ai.llm.base import LLMProvider
from afnan_ai.orchestrator import OrchestrationStatus
from afnan_ai.planner import PlanStep
from afnan_ai.state import AgentState
from afnan_ai.verifier import VerificationStatus, Verifier


# ----------------------------------------------------------------------
# Fake site: login requires a filled username; help opens a popup;
# a Continue button appears late (dynamic content)
# ----------------------------------------------------------------------

class FakeElement:
    def __init__(self, tag, attrs=None, text="", *, visible=True,
                 enabled=True, value=""):
        self.tag = tag
        self.attrs = dict(attrs or {})
        self.text = text
        self.visible = visible
        self.enabled = enabled
        self.value = value
        self.attached = True
        self.clicked = 0

    @property
    def editable(self):
        return self.tag in ("input", "textarea", "select")


class FakeRelPage:
    def __init__(self):
        self.history = ["about:blank"]
        self.index = 0
        self.elements = []
        self.text = ""
        self.page_title = ""
        self.pending = []

    @property
    def url(self):
        return self.history[self.index]

    def load(self, url):
        for el in self.elements:
            el.attached = False
        self.pending = []
        if "broken" in url:
            raise RuntimeError("net::ERR_NAME_NOT_RESOLVED")
        if "login" in url:
            self.page_title = "Login"
            self.text = "Welcome, please sign in"
            self.elements = [
                FakeElement("input", {"id": "username", "type": "text",
                                      "placeholder": "Username"}),
                FakeElement("input", {"id": "password", "type": "password"}),
                FakeElement("button", {"id": "signin"}, text="Sign in"),
                FakeElement("a", {"id": "help",
                                  "data-popup-url": "https://app.example/help"},
                            text="Help"),
            ]
            self.pending.append(
                (FakeElement("button", {"id": "continue"}, text="Continue"), 2)
            )
        elif "home" in url:
            self.page_title = "Dashboard"
            self.text = "Your dashboard"
            self.elements = [
                FakeElement("button", {"id": "logout"}, text="Logout")
            ]
        elif "help" in url:
            self.page_title = "Help"
            self.text = "How can we help?"
            self.elements = [
                FakeElement("button", {"id": "close"}, text="Close")
            ]
        else:
            self.page_title = ""
            self.text = ""
            self.elements = []

    def goto(self, url):
        self.history = self.history[: self.index + 1] + [url]
        self.index += 1
        self.load(url)

    def pump(self):
        still = []
        for el, left in self.pending:
            left -= 1
            if left <= 0:
                self.elements.append(el)
            else:
                still.append((el, left))
        self.pending = still


class FakeRelBackend(BrowserBackend):
    name = "fake-rel"

    def __init__(self):
        self.pages = []
        self.started = False
        self.clicked_ids = []

    def start(self, browser, headless):
        self.started = True

    def connect(self, endpoint):
        self.started = True

    def stop(self):
        self.started = False
        self.pages.clear()

    def new_page(self):
        page = FakeRelPage()
        self.pages.append(page)
        return page

    def close_page(self, handle):
        self.pages.remove(handle)

    def goto(self, handle, url):
        handle.goto(url)

    def go_back(self, handle):
        if handle.index > 0:
            handle.index -= 1
            handle.load(handle.url)

    def go_forward(self, handle):
        if handle.index < len(handle.history) - 1:
            handle.index += 1
            handle.load(handle.url)

    def reload(self, handle):
        handle.load(handle.url)

    def page_url(self, handle):
        return handle.url

    def page_title(self, handle):
        return handle.page_title

    def list_pages(self):
        return list(self.pages)

    # -- elements ---------------------------------------------------
    def query_elements(self, handle, locator, limit):
        return [el for el in handle.elements
                if el.attached and self._matches(el, locator)][:limit]

    @staticmethod
    def _matches(el, locator):
        if locator.get("selector"):
            sel = locator["selector"]
            return el.attrs.get("id") == sel[1:] if sel.startswith("#") else el.tag == sel
        if locator.get("text"):
            return locator["text"] in el.text
        if locator.get("role"):
            role = {"button": "button", "a": "link"}.get(el.tag, "")
            if role != locator["role"]:
                return False
            name = locator.get("name")
            return not name or name.lower() in el.text.lower()
        if locator.get("placeholder"):
            return el.attrs.get("placeholder") == locator["placeholder"]
        return False

    def element_info(self, handle, element):
        if not element.attached:
            from afnan_ai.browser import BrowserErrorCode, BrowserException
            raise BrowserException(
                "element is detached", code=BrowserErrorCode.STALE_ELEMENT
            )
        return {
            "tag": element.tag,
            "text": element.text,
            "attributes": dict(element.attrs),
            "visible": element.visible,
            "enabled": element.enabled,
            "editable": element.editable,
            "value": element.value,
        }

    def click_element(self, handle, element, timeout_ms):
        element.clicked += 1
        self.clicked_ids.append(element.attrs.get("id"))
        popup = element.attrs.get("data-popup-url")
        if popup:
            self.new_page().goto(popup)
            return
        if element.attrs.get("id") == "signin":
            username = next(
                (el for el in handle.elements
                 if el.attrs.get("id") == "username"),
                None,
            )
            if username is not None and username.value:
                handle.goto("https://app.example/home")

    def fill_element(self, handle, element, text, timeout_ms):
        element.value = text

    def clear_element(self, handle, element, timeout_ms):
        element.value = ""

    def select_option(self, handle, element, value, timeout_ms):
        element.value = value

    def press_key(self, handle, element, key, timeout_ms):
        pass

    def scroll_page(self, handle, dx, dy):
        pass

    def scroll_to_element(self, handle, element):
        pass

    # -- observation --------------------------------------------------
    def page_text(self, handle):
        return handle.text

    def interactive_elements(self, handle, limit):
        tags = ("a", "button", "input", "select", "textarea")
        return [el for el in handle.elements if el.tag in tags][:limit]

    def wait_for(self, handle, spec, timeout_ms):
        from afnan_ai.browser import BrowserErrorCode, BrowserException
        for _ in range(50):
            if self._met(handle, spec):
                return
            if not handle.pending:
                break
            handle.pump()
        if self._met(handle, spec):
            return
        raise BrowserException(
            f"condition {spec['kind']} not met within {timeout_ms}ms",
            code=BrowserErrorCode.TIMEOUT,
        )

    def _met(self, handle, spec):
        kind = spec["kind"]
        if kind == "element_present":
            return bool(self.query_elements(handle, spec["locator"], 1))
        if kind == "text_present":
            return spec["text"] in handle.text
        if kind == "url_contains":
            return spec["value"] in handle.url
        if kind == "title_contains":
            return spec["value"] in handle.page_title
        return False

    def screenshot(self, handle):
        return b"\x89PNG\r\n\x1a\nfake"


def make_controller():
    backend = FakeRelBackend()
    controller = BrowserController(backend=backend)
    return controller, backend


def signin_clicks(backend):
    return backend.clicked_ids.count("signin")


class QueueLLM(LLMProvider):
    name = "queue"
    display_name = "Queue"
    model = "queue-1"

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages):
        if not self.replies:
            raise AssertionError("QueueLLM ran out of replies")
        return self.replies.pop(0)


def plan_json(goal, steps):
    return json.dumps({"goal": goal, "steps": steps})


def step(step_id, tool, arguments, expected):
    return {
        "step_id": step_id,
        "description": f"{tool} step",
        "tool_name": tool,
        "arguments": arguments,
        "expected_result": expected,
    }


LOGIN = "https://app.example/login"


def make_agent(replies):
    controller, backend = make_controller()
    agent = AfnanAgent(
        llm_provider=QueueLLM(replies),
        browser_controller=controller,
    )
    return agent, backend


# ----------------------------------------------------------------------
# Verifier judged on actual page state
# ----------------------------------------------------------------------

class TestStateBasedVerification(unittest.TestCase):
    OBS_LOGIN = {
        "url": LOGIN,
        "title": "Login",
        "text": "Welcome, please sign in",
        "page_changed": False,
        "elements": [
            {"tag": "input", "text": "", "attributes": {"id": "username"}},
            {"tag": "button", "text": "Sign in", "attributes": {"id": "signin"}},
        ],
    }
    OBS_HOME = {
        "url": "https://app.example/home",
        "title": "Dashboard",
        "text": "Your dashboard",
        "page_changed": True,
        "elements": [
            {"tag": "button", "text": "Logout", "attributes": {"id": "logout"}}
        ],
    }

    def _step(self):
        return PlanStep(
            step_id="s1",
            description="Click sign in",
            tool_name="browser_click",
            arguments={"selector": "#signin"},
            expected_result="Dashboard home page is open",
        )

    def test_verified_from_observed_state_not_tool_report(self):
        verifier = Verifier(
            observation_provider=lambda step: self.OBS_HOME
        )
        result = verifier.verify_step(
            self._step(),
            {"success": True, "output": {"action": "click"}},
            record=False,
        )
        self.assertEqual(result.status, VerificationStatus.VERIFIED)
        self.assertIn("observation", result.evidence)

    def test_failed_when_page_state_contradicts_success_report(self):
        verifier = Verifier(
            observation_provider=lambda step: self.OBS_LOGIN
        )
        # decisive: none of the expected keywords appear on the
        # observed login page, despite the tool claiming the home
        # page — the observed state wins over the report
        step = PlanStep(
            step_id="s1",
            description="Click sign in",
            tool_name="browser_click",
            arguments={"selector": "#signin"},
            expected_result="Dashboard home is open",
        )
        result = verifier.verify_step(
            step,
            {"success": True,
             "output": {"action": "click",
                        "page": {"url": "https://app.example/home"}}},
            record=False,
        )
        self.assertEqual(result.status, VerificationStatus.FAILED)
        self.assertIn("Observed state", result.reason)

    def test_without_provider_behaviour_unchanged(self):
        verifier = Verifier()
        result = verifier.verify_step(
            self._step(),
            {"success": True, "output": {"action": "click"}},
            record=False,
        )
        # unchanged deterministic behaviour: nothing in the thin
        # output matches the expectation -> failed, as before
        self.assertEqual(result.status, VerificationStatus.FAILED)


# ----------------------------------------------------------------------
# Recovery strategies (advisor)
# ----------------------------------------------------------------------

class TestRecoveryAdvice(unittest.TestCase):
    def setUp(self):
        self.controller, self.backend = make_controller()
        self.controller.launch()
        self.controller.new_tab(LOGIN)
        self.reliability = BrowserReliability(self.controller)

    def _result(self, reason):
        from afnan_ai.verifier import VerificationResult
        return VerificationResult(
            step_id="s1",
            tool_name="browser_click",
            status=VerificationStatus.FAILED,
            reason=reason,
        )

    def test_element_not_found_with_candidate_locators(self):
        step = self._step_for("browser_click")
        advice = self.reliability.advise(
            step,
            self._result("No element matches locator {'selector': '#x'}"),
            self.reliability.observe_state(step),
        )
        self.assertEqual(advice["kind"], "element_not_found")
        self.assertEqual(
            advice["strategy"], "relocate_from_observation"
        )
        selectors = {
            c.get("selector") for c in advice["candidate_locators"]
        }
        self.assertIn("#username", selectors)
        self.assertIn("#signin", selectors)

    def test_timeout_and_stale_strategies(self):
        step = self._step_for("browser_click")
        advice = self.reliability.advise(
            step, self._result("Timed out waiting for element"), None
        )
        self.assertEqual(advice["strategy"], "wait_for_condition_first")
        advice = self.reliability.advise(
            step,
            self._result("Element ref belongs to an older version of the page"),
            None,
        )
        self.assertEqual(advice["kind"], "stale_element")

    def test_navigation_failure_strategy(self):
        step = self._step_for("browser_navigate")
        advice = self.reliability.advise(
            step,
            self._result("Execution failed: Navigation to 'x' failed"),
            None,
        )
        self.assertEqual(advice["kind"], "navigation_failed")

    def test_popup_noted_in_advice(self):
        self.controller.click({"selector": "#help"})
        step = self._step_for("browser_click")
        observation = self.reliability.observe_state(step)
        advice = self.reliability.advise(
            step, self._result("Outcome unclear"), observation
        )
        self.assertIn("tabs are open", advice["advice"])
        self.assertEqual(len(advice["open_tabs"]), 2)

    def test_non_browser_step_gets_no_advice(self):
        step = self._step_for("open_url")
        self.assertIsNone(
            self.reliability.advise(step, self._result("failed"), None)
        )
        self.assertIsNone(self.reliability.observe_state(step))

    def _step_for(self, tool):
        return PlanStep(
            step_id="s1", description="x", tool_name=tool,
            arguments={}, expected_result="something",
        )


class TestAdvisorRecordedInState(unittest.TestCase):
    def test_failed_verification_carries_advice_into_state(self):
        controller, backend = make_controller()
        controller.launch()
        controller.new_tab(LOGIN)
        reliability = BrowserReliability(controller)
        state = AgentState.create("Log in")
        state.start_task()
        verifier = Verifier(
            observation_provider=reliability.observe_state,
            failure_advisor=reliability.advise,
        )
        step = PlanStep(
            step_id="s1", description="Click", tool_name="browser_click",
            arguments={"selector": "#missing"},
            expected_result="Dashboard home page is open",
        )
        result = verifier.verify_step(
            step,
            {"success": False,
             "error": "No element matches locator {'selector': '#missing'}"},
            state=state,
        )
        self.assertEqual(result.status, VerificationStatus.FAILED)
        self.assertEqual(
            result.evidence["recovery_advice"]["kind"],
            "element_not_found",
        )
        self.assertIn("Recovery guidance", result.reason)
        verifications = state.metadata["verifications"]
        self.assertEqual(
            verifications[-1]["evidence"]["recovery_advice"]["strategy"],
            "relocate_from_observation",
        )


# ----------------------------------------------------------------------
# End-to-end browser tasks through Agent orchestration
# ----------------------------------------------------------------------

LAUNCH = step("s1", "browser_launch", {}, "Browser is running")
NAVIGATE = step("s2", "browser_navigate", {"url": LOGIN},
                "Login page with username field")


class TestEndToEndBrowserTasks(unittest.TestCase):
    def test_successful_task_verified_against_page_state(self):
        plan = plan_json("Sign in", [
            LAUNCH, NAVIGATE,
            step("s3", "browser_type",
                 {"selector": "#username", "text": "afnan"},
                 "Username input value afnan"),
            step("s4", "browser_click", {"selector": "#signin"},
                 "Dashboard home page is open"),
        ])
        agent, backend = make_agent([plan])
        result = agent.orchestrator.run("Sign in as afnan")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(result.recovery_attempts, [])
        self.assertEqual(
            agent.browser.current_page()["url"],
            "https://app.example/home",
        )

    def test_failed_action_recovers_without_blind_repeat(self):
        # v1 clicks Sign in without typing the username: the click
        # "succeeds" but the page does not change, so verification
        # (against the observed page) does not pass and recovery
        # replans — with a different action, not the same click.
        v1 = plan_json("Sign in", [
            LAUNCH, NAVIGATE,
            step("s3", "browser_click", {"selector": "#signin"},
                 "Dashboard home page is open"),
        ])
        fixed = plan_json("Sign in", [
            step("t1", "browser_type",
                 {"selector": "#username", "text": "afnan"},
                 "Username input value afnan"),
            # an alternative action (different target), not the
            # same click that already failed to produce the change
            step("t2", "browser_click", {"text": "Sign in"},
                 "Dashboard home page is open"),
        ])
        agent, backend = make_agent([v1, fixed])
        result = agent.orchestrator.run("Sign in as afnan")

        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(len(result.recovery_attempts), 1)
        self.assertEqual(
            result.recovery_attempts[0]["outcome"], "replanned"
        )
        # the original click ran once, the recovery plan's click
        # ran once — nothing was repeated blindly
        self.assertEqual(signin_clicks(backend), 2)
        self.assertEqual(
            agent.browser.current_page()["url"],
            "https://app.example/home",
        )
        # the failed/uncertain verification carried recovery advice
        verifications = result.state.metadata["verifications"]
        advised = [
            v for v in verifications
            if "recovery_advice" in (v.get("evidence") or {})
        ]
        self.assertTrue(advised)

    def test_blind_repeat_rejected_and_never_executed(self):
        v1 = plan_json("Sign in", [
            LAUNCH, NAVIGATE,
            step("s3", "browser_click", {"selector": "#does-not-exist"},
                 "Dashboard home page is open"),
        ])
        blind_repeat = plan_json("Sign in", [
            step("r1", "browser_click",
                 {"selector": "#does-not-exist"},
                 "Dashboard home page is open"),
        ])
        agent, backend = make_agent([v1, blind_repeat])
        result = agent.orchestrator.run("Sign in as afnan")

        # the recovery plan repeated the exact failed action, so
        # it was rejected before execution and the task failed
        # honestly instead of running the same action again
        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        self.assertEqual(len(result.recovery_attempts), 1)
        self.assertEqual(
            result.recovery_attempts[0]["outcome"], "repeated_action"
        )
        click_results = [
            r for r in result.state.tool_results
            if r.tool == "browser_click"
        ]
        self.assertEqual(len(click_results), 1)  # only the original

    def test_navigation_failure_recovers_to_working_route(self):
        v1 = plan_json("Open the app", [
            LAUNCH,
            step("s2", "browser_navigate",
                 {"url": "https://app.example/broken"},
                 "App page is open"),
        ])
        fixed = plan_json("Open the app", [
            step("r1", "browser_navigate", {"url": LOGIN},
                 "Login page with username field"),
        ])
        agent, backend = make_agent([v1, fixed])
        result = agent.orchestrator.run("Open the app")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(
            agent.browser.current_page()["url"], LOGIN
        )
        verifications = result.state.metadata["verifications"]
        nav_advice = [
            (v.get("evidence") or {}).get("recovery_advice", {})
            for v in verifications
        ]
        self.assertTrue(
            any(a.get("kind") == "navigation_failed" for a in nav_advice)
        )

    def test_unexpected_popup_recovers_by_selecting_tab(self):
        v1 = plan_json("Open help", [
            LAUNCH, NAVIGATE,
            step("s3", "browser_click", {"selector": "#help"},
                 "Help page is open"),
        ])
        fixed = plan_json("Open help", [
            step("r1", "browser_select_tab", {"tab_id": "tab_2"},
                 "Help tab with Close button"),
        ])
        agent, backend = make_agent([v1, fixed])
        result = agent.orchestrator.run("Open help")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(
            agent.browser.current_page()["url"],
            "https://app.example/help",
        )
        verifications = result.state.metadata["verifications"]
        popup_advice = [
            (v.get("evidence") or {}).get("recovery_advice", {})
            for v in verifications
        ]
        self.assertTrue(
            any("open_tabs" in a for a in popup_advice if a)
        )

    def test_dynamic_content_wait_then_act(self):
        plan = plan_json("Continue when ready", [
            LAUNCH, NAVIGATE,
            step("s3", "browser_wait_for",
                 {"condition": "element_present", "selector": "#continue"},
                 "Continue button visible on page"),
            step("s4", "browser_click", {"selector": "#continue"},
                 "Continue button on page"),
        ])
        agent, backend = make_agent([plan])
        result = agent.orchestrator.run("Continue when ready")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        page = backend.pages[0]
        continued = next(
            el for el in page.elements if el.attrs.get("id") == "continue"
        )
        self.assertEqual(continued.clicked, 1)


class TestAgentWiring(unittest.TestCase):
    def test_reliability_wired_by_default(self):
        controller, backend = make_controller()
        agent = AfnanAgent(
            llm_provider=QueueLLM(["{}"]), browser_controller=controller
        )
        self.assertIsNotNone(agent.browser_reliability)
        self.assertIsNotNone(agent.verifier.observation_provider)
        self.assertIsNotNone(agent.verifier.failure_advisor)

    def test_reliability_absent_when_browser_tools_disabled(self):
        agent = AfnanAgent(
            llm_provider=QueueLLM(["{}"]), enable_browser_tools=False
        )
        self.assertIsNone(agent.browser_reliability)
        self.assertIsNone(agent.verifier.observation_provider)


if __name__ == "__main__":
    unittest.main()

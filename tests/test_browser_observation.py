"""Browser observation system tests: structured page state,
condition-based waits (dynamic content), popup adoption,
screenshots recorded as AgentState observations, and failure
cases that must never look successful.

All against an in-memory fake backend — no real browser.
"""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai import Agent
from afnan_ai.browser import (
    BrowserController,
    BrowserErrorCode,
    BrowserException,
    register_browser_tools,
)
from afnan_ai.browser.backend import BrowserBackend
from afnan_ai.executor import Executor
from afnan_ai.llm.base import LLMProvider
from afnan_ai.orchestrator import OrchestrationStatus
from afnan_ai.planner import Planner
from afnan_ai.state import AgentState
from afnan_ai.tools import ToolRegistry
from afnan_ai.verifier import Verifier


# ----------------------------------------------------------------------
# Fake page/backend with dynamic content, popups and screenshots
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


PAGES = {
    "https://app.example/login": (
        "Login",
        "Welcome, please sign in",
        lambda: [
            FakeElement("input", {"id": "username", "type": "text",
                                  "placeholder": "Username"}),
            FakeElement("button", {"id": "signin"}, text="Sign in"),
            FakeElement("a", {"id": "help",
                              "data-popup-url": "https://app.example/help"},
                        text="Help"),
            FakeElement("div", {"id": "banner"}, text="Welcome"),
        ],
    ),
    "https://app.example/home": (
        "Home",
        "Your dashboard",
        lambda: [FakeElement("button", {"id": "logout"}, text="Logout")],
    ),
    "https://app.example/help": (
        "Help",
        "How can we help?",
        lambda: [FakeElement("button", {"id": "close"}, text="Close")],
    ),
}


class FakeObsPage:
    def __init__(self):
        self.history = ["about:blank"]
        self.index = 0
        self.elements = []
        self.text = ""
        self.page_title = ""
        self.pending = []  # (element, pumps_left)

    @property
    def url(self):
        return self.history[self.index]

    def load(self, url):
        for el in self.elements:
            el.attached = False
        title, text, factory = PAGES.get(
            url, ("", "", lambda: [])
        )
        self.page_title = title
        self.text = text
        self.elements = factory()

    def goto(self, url):
        self.history = self.history[: self.index + 1] + [url]
        self.index += 1
        self.load(url)

    def pump(self):
        """Simulate time passing: pending elements materialize."""
        still = []
        for el, left in self.pending:
            left -= 1
            if left <= 0:
                self.elements.append(el)
            else:
                still.append((el, left))
        self.pending = still


class FakeObsBackend(BrowserBackend):
    name = "fake-obs"

    def __init__(self):
        self.pages = []
        self.started = False
        self.empty_screenshot = False

    def start(self, browser, headless):
        self.started = True

    def connect(self, endpoint):
        self.started = True

    def stop(self):
        self.started = False
        self.pages.clear()

    def new_page(self):
        page = FakeObsPage()
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
            if sel.startswith("#"):
                return el.attrs.get("id") == sel[1:]
            return el.tag == sel
        if locator.get("text"):
            return locator["text"] in el.text
        if locator.get("role"):
            role = {"button": "button", "a": "link"}.get(el.tag, "")
            if role != locator["role"]:
                return False
            name = locator.get("name")
            return not name or name.lower() in el.text.lower()
        return False

    def element_info(self, handle, element):
        if not element.attached:
            raise BrowserException(
                "element is detached",
                code=BrowserErrorCode.STALE_ELEMENT,
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
        popup = element.attrs.get("data-popup-url")
        if popup:
            page = self.new_page()
            page.goto(popup)

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
        for _ in range(50):
            if self._condition_met(handle, spec):
                return
            if not handle.pending:
                break
            handle.pump()
        if self._condition_met(handle, spec):
            return
        raise BrowserException(
            f"condition {spec['kind']} not met within {timeout_ms}ms",
            code=BrowserErrorCode.TIMEOUT,
        )

    def _condition_met(self, handle, spec):
        kind = spec["kind"]
        if kind == "element_present":
            return bool(self.query_elements(handle, spec["locator"], 1))
        if kind == "element_hidden":
            found = self.query_elements(handle, spec["locator"], 5)
            return not found or all(not el.visible for el in found)
        if kind == "text_present":
            if spec["text"] in handle.text:
                return True
            return any(spec["text"] in el.text for el in handle.elements)
        if kind == "url_contains":
            return spec["value"] in handle.url
        if kind == "title_contains":
            return spec["value"] in handle.page_title
        return False

    def screenshot(self, handle):
        if self.empty_screenshot:
            return b""
        return b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes"


def make_controller(url="https://app.example/login"):
    backend = FakeObsBackend()
    controller = BrowserController(backend=backend)
    controller.launch()
    controller.new_tab(url)
    return controller, backend


# ----------------------------------------------------------------------
# Page state observation
# ----------------------------------------------------------------------

class TestObservePage(unittest.TestCase):
    def setUp(self):
        self.controller, self.backend = make_controller()

    def test_observe_returns_structured_state(self):
        obs = self.controller.observe()
        self.assertEqual(obs["url"], "https://app.example/login")
        self.assertEqual(obs["title"], "Login")
        self.assertIn("please sign in", obs["text"])
        self.assertEqual(obs["kind"], "observation")
        tags = {el["tag"] for el in obs["elements"]}
        self.assertIn("input", tags)
        self.assertIn("button", tags)
        # non-interactive div is not listed
        ids = {el["attributes"].get("id") for el in obs["elements"]}
        self.assertNotIn("banner", ids)

    def test_observed_element_ref_is_usable(self):
        obs = self.controller.observe()
        username = next(
            el for el in obs["elements"]
            if el["attributes"].get("id") == "username"
        )
        self.controller.type_text({"ref": username["ref"]}, "afnan")
        backend_el = next(
            el for el in self.backend.pages[0].elements
            if el.attrs.get("id") == "username"
        )
        self.assertEqual(backend_el.value, "afnan")

    def test_page_change_detection(self):
        first = self.controller.observe()
        self.assertTrue(first["page_changed"])  # first sight
        second = self.controller.observe()
        self.assertFalse(second["page_changed"])  # nothing changed
        self.controller.navigate("https://app.example/home")
        third = self.controller.observe()
        self.assertTrue(third["page_changed"])
        self.assertEqual(third["url"], "https://app.example/home")

    def test_observe_before_launch_fails_structured(self):
        controller = BrowserController(backend=FakeObsBackend())
        with self.assertRaises(BrowserException) as ctx:
            controller.observe()
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.BROWSER_NOT_STARTED
        )


# ----------------------------------------------------------------------
# Waits on dynamic content
# ----------------------------------------------------------------------

class TestWaitFor(unittest.TestCase):
    def setUp(self):
        self.controller, self.backend = make_controller()
        self.page = self.backend.pages[0]

    def test_wait_for_delayed_element_then_click(self):
        self.page.pending.append(
            (FakeElement("button", {"id": "delayed"}, text="Continue"), 2)
        )
        # not there yet
        with self.assertRaises(BrowserException) as ctx:
            self.controller.find_elements({"selector": "#delayed"})
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.ELEMENT_NOT_FOUND
        )
        result = self.controller.wait_for(
            "element_present", locator={"selector": "#delayed"}
        )
        self.assertTrue(result["satisfied"])
        self.controller.click({"selector": "#delayed"})

    def test_wait_for_text_and_url(self):
        result = self.controller.wait_for(
            "text_present", text="please sign in"
        )
        self.assertTrue(result["satisfied"])
        self.controller.navigate("https://app.example/home")
        result = self.controller.wait_for(
            "url_contains", value="home"
        )
        self.assertEqual(result["page"]["url"], "https://app.example/home")

    def test_wait_timeout_is_structured_not_success(self):
        with self.assertRaises(BrowserException) as ctx:
            self.controller.wait_for(
                "element_present",
                locator={"selector": "#never"},
                timeout_ms=100,
            )
        self.assertEqual(ctx.exception.code, BrowserErrorCode.TIMEOUT)

    def test_wait_validation_errors(self):
        with self.assertRaises(BrowserException) as ctx:
            self.controller.wait_for("element_present")
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.INVALID_LOCATOR
        )
        with self.assertRaises(BrowserException) as ctx:
            self.controller.wait_for("nonsense_condition")
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.INVALID_LOCATOR
        )


# ----------------------------------------------------------------------
# Popups / new tabs
# ----------------------------------------------------------------------

class TestPopupAdoption(unittest.TestCase):
    def test_popup_becomes_a_selectable_tab(self):
        controller, backend = make_controller()
        controller.click({"selector": "#help"})
        tabs = controller.list_tabs()
        self.assertEqual(len(tabs), 2)
        urls = {t["url"] for t in tabs}
        self.assertIn("https://app.example/help", urls)
        popup = next(
            t for t in tabs if t["url"].endswith("/help")
        )
        controller.select_tab(popup["tab_id"])
        self.assertEqual(
            controller.current_page()["url"], "https://app.example/help"
        )


# ----------------------------------------------------------------------
# Screenshots + AgentState observation recording
# ----------------------------------------------------------------------

class TestScreenshot(unittest.TestCase):
    def test_screenshot_saved_with_metadata(self):
        controller, backend = make_controller()
        with tempfile.TemporaryDirectory() as tmp:
            shot = controller.screenshot(output_dir=tmp)
            path = Path(shot["path"])
            self.assertTrue(path.exists())
            self.assertGreater(shot["size_bytes"], 0)
            self.assertEqual(shot["url"], "https://app.example/login")
            self.assertEqual(
                shot["observation"]["type"], "screenshot"
            )

    def test_empty_screenshot_is_failure_not_success(self):
        controller, backend = make_controller()
        backend.empty_screenshot = True
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(BrowserException) as ctx:
                controller.screenshot(output_dir=tmp)
            self.assertEqual(
                ctx.exception.code, BrowserErrorCode.OPERATION_FAILED
            )

    def test_screenshot_recorded_in_agent_state_via_executor(self):
        backend = FakeObsBackend()
        controller = BrowserController(backend=backend)
        registry = ToolRegistry()
        register_browser_tools(registry, controller)
        with tempfile.TemporaryDirectory() as tmp:
            plan_json = json.dumps({
                "goal": "Open login, wait, screenshot",
                "steps": [
                    {"step_id": "step_1", "description": "Launch",
                     "tool_name": "browser_launch", "arguments": {},
                     "expected_result": "Browser is running"},
                    {"step_id": "step_2", "description": "Open login",
                     "tool_name": "browser_navigate",
                     "arguments": {"url": "https://app.example/login"},
                     "expected_result": "Login page is open"},
                    {"step_id": "step_3", "description": "Wait for field",
                     "tool_name": "browser_wait_for",
                     "arguments": {"condition": "element_present",
                                   "selector": "#username"},
                     "expected_result": "Username field is present"},
                    {"step_id": "step_4", "description": "Screenshot",
                     "tool_name": "browser_screenshot",
                     "arguments": {"output_dir": tmp},
                     "expected_result": "Screenshot file is saved"},
                ],
            })

            class StubLLM(LLMProvider):
                name = "stub"
                display_name = "Stub"
                model = "stub-1"

                def chat(self, messages):
                    return plan_json

            from afnan_ai.planner import Planner
            agent = Agent(
                planner=Planner(StubLLM(), registry),
                executor=Executor(registry),
                verifier=Verifier(),
                recover_on_uncertain=False,
            )
            result = agent.run("Open login, wait, screenshot")
            self.assertEqual(
                result.status, OrchestrationStatus.COMPLETED
            )
            state = result.state
            # tool result recorded
            shot_results = [
                r for r in state.tool_results
                if r.tool == "browser_screenshot" and r.success
            ]
            self.assertEqual(len(shot_results), 1)
            # structured observation recorded too
            shot_obs = [
                o for o in state.observations
                if o.metadata.get("observation", {}).get("type")
                == "screenshot"
            ]
            self.assertEqual(len(shot_obs), 1)
            self.assertTrue(
                Path(shot_obs[0].metadata["observation"]["path"]).exists()
            )
            # state stays serializable with the observation inside
            restored = AgentState.from_json(state.to_json())
            self.assertEqual(
                len(restored.observations), len(state.observations)
            )


if __name__ == "__main__":
    unittest.main()

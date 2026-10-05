import json
import re
import sys
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
from afnan_ai.tools import ToolRegistry
from afnan_ai.verifier import Verifier


# ----------------------------------------------------------------------
# In-memory page with a small DOM (a login form)
# ----------------------------------------------------------------------

class FakeElement:
    def __init__(self, tag, attrs=None, text="", *, visible=True,
                 enabled=True, value="", options=None, hangs=False):
        self.tag = tag
        self.attrs = dict(attrs or {})
        self.text = text
        self.visible = visible
        self.enabled = enabled
        self.value = value
        self.options = list(options or [])
        self.hangs = hangs
        self.attached = True
        self.clicked = 0
        self.keys = []
        self.scrolled_to = False

    @property
    def editable(self):
        return self.tag in ("input", "textarea", "select")


def login_dom():
    return [
        FakeElement("input", {"id": "username", "type": "text",
                              "placeholder": "Username", "label": "Username"}),
        FakeElement("input", {"id": "password", "type": "password",
                              "placeholder": "Password"}),
        FakeElement("button", {"id": "signin"}, text="Sign in"),
        FakeElement("button", {"id": "slow"}, text="Slow", hangs=True),
        FakeElement("button", {"id": "ghost"}, text="Ghost", visible=False),
        FakeElement("select", {"id": "country"}, value="pk",
                    options=["pk", "in", "us"]),
        FakeElement("a", {"id": "forgot", "href": "/reset"},
                    text="Forgot password?"),
        FakeElement("div", {"id": "welcome"}, text="Welcome back"),
    ]


def home_dom():
    return [
        FakeElement("h1", {"id": "dash"}, text="Dashboard"),
        FakeElement("button", {"id": "logout"}, text="Logout"),
    ]


class FakeInteractivePage:
    def __init__(self):
        self.history = ["about:blank"]
        self.index = 0
        self.elements = []
        self.keys = []
        self.scroll = (0, 0)

    @property
    def url(self):
        return self.history[self.index]

    def load(self, url):
        for el in self.elements:
            el.attached = False
        if "login" in url:
            self.elements = login_dom()
        elif "home" in url:
            self.elements = home_dom()
        else:
            self.elements = []

    def goto(self, url):
        self.history = self.history[: self.index + 1] + [url]
        self.index += 1
        self.load(url)


class FakeInteractiveBackend(BrowserBackend):
    name = "fake-interactive"

    def __init__(self):
        self.pages = []
        self.started = False

    def start(self, browser, headless):
        self.started = True

    def connect(self, endpoint):
        self.started = True

    def stop(self):
        self.started = False
        self.pages.clear()

    def new_page(self):
        page = FakeInteractivePage()
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
        return "Login" if "login" in handle.url else handle.url

    # -- elements -----------------------------------------------------
    def query_elements(self, handle, locator, limit):
        return [el for el in handle.elements
                if el.attached and self._matches(el, locator)][:limit]

    @staticmethod
    def _matches(el, locator):
        if locator.get("selector"):
            sel = locator["selector"]
            if sel.startswith("#"):
                return el.attrs.get("id") == sel[1:]
            m = re.fullmatch(r"([a-z]+)#([\w-]+)", sel)
            if m:
                return el.tag == m.group(1) and el.attrs.get("id") == m.group(2)
            m = re.fullmatch(r"\[([\w-]+)='?([\w-]+)'?\]", sel)
            if m:
                return el.attrs.get(m.group(1)) == m.group(2)
            return el.tag == sel
        if locator.get("test_id"):
            return el.attrs.get("data-testid") == locator["test_id"]
        if locator.get("label"):
            return locator["label"] in (
                el.attrs.get("label", "") + el.attrs.get("aria-label", "")
            )
        if locator.get("placeholder"):
            return el.attrs.get("placeholder") == locator["placeholder"]
        if locator.get("role"):
            role = el.attrs.get("role") or {
                "button": "button", "a": "link",
                "select": "combobox", "input": "textbox",
            }.get(el.tag, "")
            if role != locator["role"]:
                return False
            name = locator.get("name")
            if name:
                accessible = el.attrs.get("aria-label") or el.text
                return name.lower() in accessible.lower()
            return True
        if locator.get("text"):
            return locator["text"] in el.text
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

    def _usable(self, element):
        if not element.attached:
            raise BrowserException(
                "element is detached",
                code=BrowserErrorCode.STALE_ELEMENT,
            )
        if element.hangs:
            raise BrowserException(
                "element never became actionable",
                code=BrowserErrorCode.TIMEOUT,
            )

    def click_element(self, handle, element, timeout_ms):
        self._usable(element)
        element.clicked += 1

    def fill_element(self, handle, element, text, timeout_ms):
        self._usable(element)
        element.value = text

    def clear_element(self, handle, element, timeout_ms):
        self._usable(element)
        element.value = ""

    def select_option(self, handle, element, value, timeout_ms):
        self._usable(element)
        if value not in element.options:
            raise BrowserException(
                f"option {value!r} not in select",
                code=BrowserErrorCode.OPERATION_FAILED,
            )
        element.value = value

    def press_key(self, handle, element, key, timeout_ms):
        if element is not None:
            self._usable(element)
            element.keys.append(key)
        else:
            handle.keys.append(key)

    def scroll_page(self, handle, dx, dy):
        handle.scroll = (dx, dy)

    def scroll_to_element(self, handle, element):
        self._usable(element)
        element.scrolled_to = True


def make_controller():
    backend = FakeInteractiveBackend()
    controller = BrowserController(backend=backend)
    controller.launch()
    controller.new_tab("https://app.example/login")
    return controller, backend


def current_page_elements(backend):
    return backend.pages[0].elements


def find_el(backend, element_id):
    for el in current_page_elements(backend):
        if el.attrs.get("id") == element_id:
            return el
    raise AssertionError(f"no element {element_id}")


# ----------------------------------------------------------------------
# Controller-level interaction tests
# ----------------------------------------------------------------------

class TestFindAndInspect(unittest.TestCase):
    def setUp(self):
        self.controller, self.backend = make_controller()

    def test_find_by_selector_returns_identity_and_properties(self):
        found = self.controller.find_elements({"selector": "#username"})
        self.assertEqual(len(found), 1)
        el = found[0]
        self.assertEqual(el["tag"], "input")
        self.assertTrue(el["ref"].startswith("el_"))
        self.assertEqual(el["attributes"]["placeholder"], "Username")
        self.assertTrue(el["editable"])

    def test_find_by_accessibility_role_and_name(self):
        found = self.controller.find_elements(
            {"role": "button", "name": "Sign in"}
        )
        self.assertEqual(found[0]["attributes"]["id"], "signin")

    def test_find_by_text_and_label_fallbacks(self):
        by_text = self.controller.find_elements({"text": "Forgot"})
        self.assertEqual(by_text[0]["tag"], "a")
        by_label = self.controller.find_elements({"label": "Username"})
        self.assertEqual(by_label[0]["attributes"]["id"], "username")

    def test_find_not_found(self):
        with self.assertRaises(BrowserException) as ctx:
            self.controller.find_elements({"selector": "#missing"})
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.ELEMENT_NOT_FOUND
        )

    def test_find_without_locator(self):
        with self.assertRaises(BrowserException) as ctx:
            self.controller.find_elements({})
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.INVALID_LOCATOR
        )

    def test_inspect_by_ref_and_unknown_ref(self):
        ref = self.controller.find_elements({"selector": "#country"})[0]["ref"]
        info = self.controller.inspect_element({"ref": ref})
        self.assertEqual(info["tag"], "select")
        with self.assertRaises(BrowserException) as ctx:
            self.controller.inspect_element({"ref": "el_999"})
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.INVALID_ELEMENT
        )


class TestInteractions(unittest.TestCase):
    def setUp(self):
        self.controller, self.backend = make_controller()

    def test_click_by_ref_and_by_locator(self):
        ref = self.controller.find_elements({"selector": "#signin"})[0]["ref"]
        result = self.controller.click({"ref": ref})
        self.assertEqual(result["action"], "click")
        self.assertEqual(find_el(self.backend, "signin").clicked, 1)
        self.controller.click({"locator": {"text": "Sign in"}})
        self.assertEqual(find_el(self.backend, "signin").clicked, 2)

    def test_click_hidden_element_is_rejected_before_acting(self):
        ref = self.controller.find_elements({"selector": "#ghost"})[0]["ref"]
        with self.assertRaises(BrowserException) as ctx:
            self.controller.click({"ref": ref})
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.INVALID_ELEMENT
        )
        self.assertEqual(find_el(self.backend, "ghost").clicked, 0)

    def test_type_clear_and_value(self):
        self.controller.type_text({"selector": "#username"}, "afnan")
        self.assertEqual(find_el(self.backend, "username").value, "afnan")
        self.controller.type_text(
            {"selector": "#username"}, "boss", clear_first=True
        )
        self.assertEqual(find_el(self.backend, "username").value, "boss")
        self.controller.clear_field({"selector": "#username"})
        self.assertEqual(find_el(self.backend, "username").value, "")

    def test_type_into_non_editable_is_rejected(self):
        with self.assertRaises(BrowserException) as ctx:
            self.controller.type_text({"selector": "#welcome"}, "hi")
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.INVALID_ELEMENT
        )

    def test_select_option(self):
        result = self.controller.select_option(
            {"selector": "#country"}, "us"
        )
        self.assertEqual(result["value"], "us")
        self.assertEqual(find_el(self.backend, "country").value, "us")
        with self.assertRaises(BrowserException) as ctx:
            self.controller.select_option({"selector": "#username"}, "us")
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.INVALID_ELEMENT
        )
        with self.assertRaises(BrowserException) as ctx:
            self.controller.select_option({"selector": "#country"}, "fr")
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.OPERATION_FAILED
        )

    def test_press_key_on_element_and_page(self):
        ref = self.controller.find_elements({"selector": "#username"})[0]["ref"]
        self.controller.press_key("Enter", {"ref": ref})
        self.assertEqual(find_el(self.backend, "username").keys, ["Enter"])
        self.controller.press_key("Escape")
        self.assertEqual(self.backend.pages[0].keys, ["Escape"])

    def test_scroll_page_and_to_element(self):
        result = self.controller.scroll_page(dy=800)
        self.assertEqual(result["action"], "scroll")
        self.assertEqual(self.backend.pages[0].scroll, (0, 800))
        self.controller.scroll_page(target={"selector": "#country"})
        self.assertTrue(find_el(self.backend, "country").scrolled_to)


class TestStaleTimeoutAndFailures(unittest.TestCase):
    def setUp(self):
        self.controller, self.backend = make_controller()

    def test_ref_goes_stale_after_navigation(self):
        ref = self.controller.find_elements({"selector": "#username"})[0]["ref"]
        self.controller.navigate("https://app.example/home")
        with self.assertRaises(BrowserException) as ctx:
            self.controller.click({"ref": ref})
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.STALE_ELEMENT
        )

    def test_detached_element_is_stale(self):
        ref = self.controller.find_elements({"selector": "#username"})[0]["ref"]
        find_el(self.backend, "username").attached = False
        with self.assertRaises(BrowserException) as ctx:
            self.controller.inspect_element({"ref": ref})
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.STALE_ELEMENT
        )

    def test_timeout_is_structured(self):
        with self.assertRaises(BrowserException) as ctx:
            self.controller.click({"selector": "#slow"})
        self.assertEqual(ctx.exception.code, BrowserErrorCode.TIMEOUT)


# ----------------------------------------------------------------------
# Tool-level tests through the registry + the Agent pipeline
# ----------------------------------------------------------------------

class TestInteractionTools(unittest.TestCase):
    def setUp(self):
        self.backend = FakeInteractiveBackend()
        self.controller = BrowserController(backend=self.backend)
        self.registry = ToolRegistry()
        register_browser_tools(self.registry, self.controller)
        self.registry.execute("browser_launch", {})
        self.registry.execute(
            "browser_navigate", {"url": "https://app.example/login"}
        )

    def test_type_via_tool(self):
        result = self.registry.execute(
            "browser_type", {"selector": "#username", "text": "afnan"}
        )
        self.assertTrue(result.success)
        self.assertEqual(find_el(self.backend, "username").value, "afnan")
        self.assertIn("element", result.output)

    def test_not_found_via_tool(self):
        result = self.registry.execute(
            "browser_click", {"selector": "#nope"}
        )
        self.assertFalse(result.success)
        self.assertEqual(
            result.error.details["browser_error"]["code"],
            "element_not_found",
        )

    def test_stale_via_tool(self):
        found = self.registry.execute(
            "browser_find_elements", {"selector": "#username"}
        )
        ref = found.output["elements"][0]["ref"]
        self.registry.execute(
            "browser_navigate", {"url": "https://app.example/home"}
        )
        result = self.registry.execute("browser_click", {"ref": ref})
        self.assertFalse(result.success)
        self.assertEqual(
            result.error.details["browser_error"]["code"], "stale_element"
        )

    def test_missing_target_and_missing_text(self):
        result = self.registry.execute("browser_click", {})
        self.assertFalse(result.success)
        self.assertEqual(
            result.error.details["browser_error"]["code"],
            "invalid_locator",
        )
        result = self.registry.execute(
            "browser_type", {"selector": "#username"}
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error.code.value, "missing_arguments")


class StubLLM(LLMProvider):
    name = "stub"
    display_name = "Stub"
    model = "stub-1"

    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages):
        return self.reply


class TestAgentPlansBrowserInteractions(unittest.TestCase):
    def test_login_flow_through_planner_executor(self):
        backend = FakeInteractiveBackend()
        controller = BrowserController(backend=backend)
        registry = ToolRegistry()
        register_browser_tools(registry, controller)
        plan = json.dumps(
            {
                "goal": "Log in as afnan",
                "steps": [
                    {
                        "step_id": "step_1",
                        "description": "Launch the browser",
                        "tool_name": "browser_launch",
                        "arguments": {},
                        "expected_result": "The browser is running",
                    },
                    {
                        "step_id": "step_2",
                        "description": "Open the login page",
                        "tool_name": "browser_navigate",
                        "arguments": {"url": "https://app.example/login"},
                        "expected_result": "Login page is open",
                    },
                    {
                        "step_id": "step_3",
                        "description": "Type the username",
                        "tool_name": "browser_type",
                        "arguments": {
                            "selector": "#username", "text": "afnan"
                        },
                        "expected_result": "Username field has value afnan",
                    },
                    {
                        "step_id": "step_4",
                        "description": "Click Sign in",
                        "tool_name": "browser_click",
                        "arguments": {"role": "button", "name": "Sign in"},
                        "expected_result": "Sign in button is clicked",
                    },
                ],
            }
        )
        agent = Agent(
            planner=Planner(StubLLM(plan), registry),
            executor=Executor(registry),
            verifier=Verifier(),
            recover_on_uncertain=False,
        )
        result = agent.run("Log in as afnan")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)

        # the username was typed on the login DOM (navigate home in
        # step 2 loaded it; typing happened after)
        self.assertEqual(
            controller.current_url(), "https://app.example/login"
        )
        username = [
            el for el in backend.pages[0].elements
            if el.attrs.get("id") == "username"
        ][0]
        self.assertEqual(username.value, "afnan")


if __name__ == "__main__":
    unittest.main()

"""Unified browser perception + computer-use tests.

The perception layer (accessibility → DOM → visual fallback,
semantic location, validated computer actions with post-action
observation) is tested against DynamicBackend fakes — no real
browser.  Workflows covered: search-and-submit, login
navigation, stale-target recovery, multi-tab isolation,
visual fallback, low-confidence gating and AgentState
recording.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from afnan_ai.browser.base import BrowserErrorCode, BrowserException
from afnan_ai.browser.controller import BrowserController
from afnan_ai.browser.perception import BrowserPerception
from afnan_ai.browser.runtime import AfnanBrowserRuntime
from afnan_ai.browser.security import ApprovalGate
from afnan_ai.browser.tools import create_browser_tools
from afnan_ai.screen import ScreenObservation, VisualElement, VisualRegion
from afnan_ai.tools import ToolRegistry

from test_browser_advanced import DynamicBackend, el, make_controller
from test_browser_workflow import make_agent


class PerceptionBackend(DynamicBackend):
    """DynamicBackend + AX snapshots, boxes, mouse, checked."""

    def __init__(self):
        super().__init__()
        self.mouse_events: list = []
        self.focused: list = []
        self.checked_calls: list = []

    def add_page(self, url, *, ax=None, **kwargs):
        page = super().add_page(url, **kwargs)
        page["ax"] = ax
        return page

    def accessibility_snapshot(self, handle):
        page = self._page(handle)
        if page.get("ax") is None:
            raise BrowserException(
                "no accessibility tree",
                code=BrowserErrorCode.OPERATION_FAILED,
            )
        return page["ax"]

    def element_box(self, handle, element):
        page = self._page(handle)
        index = element[1]
        return {
            "x": float(index * 100), "y": 10.0,
            "width": 80.0, "height": 20.0,
        }

    def mouse_click(self, handle, x, y, click_count=1):
        self.mouse_events.append(("click", x, y, click_count))

    def mouse_move(self, handle, x, y):
        self.mouse_events.append(("move", x, y))

    def mouse_drag(self, handle, x1, y1, x2, y2):
        self.mouse_events.append(("drag", x1, y1, x2, y2))

    def focus_element(self, handle, element, timeout_ms):
        self.focused.append(element)

    def press_key(self, handle, element, key, timeout_ms):
        self.mouse_events.append(("key", key))

    def set_checked(self, handle, element, checked, timeout_ms):
        page = self._page(handle)
        page["elements"][element[1]]["checked"] = checked
        self.checked_calls.append((element, checked))


def search_site(backend):
    backend.add_page(
        "https://search.example/", title="Search",
        text="Search the web here",
        elements=[
            el("input", "", {"placeholder": "Search", "type": "text"}),
            el("button", "Search",
               {"href": "https://results.example/"}),
        ],
    )
    backend.add_page(
        "https://results.example/", title="Results",
        text="Results for cats and dogs",
        elements=[
            el("a", "Cats article",
               {"href": "https://article.example/"}),
        ],
    )
    backend.add_page(
        "https://article.example/", title="Cats",
        text="All about cats. Cats are independent animals.",
        elements=[el("h1", "All about cats")],
    )


class StubScreen:
    """ScreenObserver stand-in returning one visual element."""

    def __init__(self, confidence=0.9):
        self.confidence = confidence

    def observe(self, origin="browser"):
        return ScreenObservation(
            source="browser", width=800, height=600,
            elements=[VisualElement(
                ref="vis_1", kind="button", label="Canvas Button",
                confidence=self.confidence, source="visual",
                region=VisualRegion(10, 10, 100, 30),
            )],
        )


def make_perception(backend=None, **controller_kw):
    backend = backend or PerceptionBackend()
    controller = BrowserController(
        runtime=AfnanBrowserRuntime(adapter=backend),
        **controller_kw,
    )
    controller.launch()
    return BrowserPerception(controller), controller, backend


class TestUnifiedObservation(unittest.TestCase):
    def test_dom_fallback_observation(self):
        backend = PerceptionBackend()
        search_site(backend)
        perception, controller, _ = make_perception(backend)
        controller.new_tab("https://search.example/")
        observed = perception.observe()
        self.assertEqual(observed["kind"], "browser_observation")
        self.assertEqual(observed["url"], "https://search.example/")
        roles = {e["element_id"]: e for e in observed["elements"]}
        self.assertTrue(roles)
        first = observed["elements"][0]
        self.assertIn("bbox", first)
        self.assertIn("confidence", first)
        self.assertIn("frame", first)
        self.assertEqual(first["tab_id"], observed["tab_id"])
        self.assertFalse(observed["loading"])
        controller.shutdown()

    def test_accessibility_tree_is_preferred_and_actionable(self):
        backend = PerceptionBackend()
        backend.add_page(
            "https://ax.example/", title="AX",
            text="AX page",
            elements=[el("button", "AX Login", {"role": "button"})],
            ax={
                "role": "document", "name": "",
                "children": [
                    {"role": "button", "name": "AX Login",
                     "children": []},
                    {"role": "heading", "name": "Welcome",
                     "children": []},
                ],
            },
        )
        perception, controller, _ = make_perception(backend)
        controller.new_tab("https://ax.example/")
        observed = perception.observe()
        ax_elements = [
            e for e in observed["elements"]
            if e["source"] == "accessibility"
        ]
        self.assertTrue(ax_elements)
        button = next(
            e for e in ax_elements if e["role"] == "button"
        )
        self.assertEqual(button["accessible_name"], "AX Login")
        self.assertTrue(button["actionable"])  # ref resolved
        self.assertGreaterEqual(button["confidence"], 0.9)
        controller.shutdown()


class TestSemanticLocate(unittest.TestCase):
    def setUp(self):
        self.backend = PerceptionBackend()
        search_site(self.backend)
        self.perception, self.controller, _ = make_perception(
            self.backend
        )
        self.controller.new_tab("https://search.example/")

    def tearDown(self):
        self.controller.shutdown()

    def test_locate_search_box_and_button(self):
        found = self.perception.locate("Search box")
        best = found["candidates"][0]
        self.assertEqual(best["tier"], "actionable")
        self.assertIsNotNone(best["element_id"])
        found = self.perception.locate("Search button")
        self.assertEqual(found["candidates"][0]["role"], "button")

    def test_low_confidence_withholds_element_id(self):
        found = self.perception.locate(
            "Quantum flux capacitor control"
        )
        best = found["candidates"][0]
        self.assertEqual(best["tier"], "low")
        self.assertIsNone(best["element_id"])
        self.assertFalse(best["actionable"])


class TestComputerActions(unittest.TestCase):
    def setUp(self):
        self.backend = PerceptionBackend()
        search_site(self.backend)
        self.perception, self.controller, _ = make_perception(
            self.backend
        )
        self.controller.new_tab("https://search.example/")

    def tearDown(self):
        self.controller.shutdown()

    def test_search_workflow_type_submit_verify(self):
        box = self.perception.locate("Search box")["candidates"][0]
        typed = self.perception.act(
            "type", element_id=box["element_id"], text="cats"
        )
        self.assertEqual(typed["result"]["element"]["value"], "cats")
        button = self.perception.locate("Search button")[
            "candidates"
        ][0]
        clicked = self.perception.act(
            "click", element_id=button["element_id"],
            expect_text="Results for cats",
        )
        self.assertTrue(clicked["page_changed"])
        self.assertTrue(clicked["verified"])
        self.assertEqual(
            clicked["after"]["url"], "https://results.example/"
        )

    def test_click_verifies_navigation_and_extracts(self):
        button = self.perception.locate("Search button")[
            "candidates"
        ][0]
        self.perception.act("click", element_id=button["element_id"])
        link = self.perception.locate("Cats article")["candidates"][0]
        opened = self.perception.act(
            "click", element_id=link["element_id"],
            expect_url_contains="article.example",
        )
        self.assertTrue(opened["verified"])
        self.assertIn(
            "All about cats", opened["after_observation"]["text"]
        )

    def test_stale_element_is_refused(self):
        box = self.perception.locate("Search box")["candidates"][0]
        self.controller.navigate("https://results.example/")
        with self.assertRaises(BrowserException) as caught:
            self.perception.act(
                "type", element_id=box["element_id"], text="x"
            )
        self.assertEqual(
            caught.exception.error.code,
            BrowserErrorCode.STALE_ELEMENT,
        )

    def test_unknown_element_id_is_stale(self):
        with self.assertRaises(BrowserException) as caught:
            self.perception.act("click", element_id="pe_999")
        self.assertEqual(
            caught.exception.error.code,
            BrowserErrorCode.STALE_ELEMENT,
        )

    def test_keyboard_scroll_focus_hover_double_click(self):
        box = self.perception.locate("Search box")["candidates"][0]
        focused = self.perception.act(
            "focus", element_id=box["element_id"]
        )
        self.assertEqual(focused["result"]["action"], "focus")
        self.assertTrue(self.backend.focused)
        keyed = self.perception.act(
            "press_key", element_id=box["element_id"], key="Enter"
        )
        self.assertEqual(keyed["result"]["action"], "press_key")
        hot = self.perception.act("hotkey", key="Control+A")
        self.assertEqual(hot["result"]["key"], "Control+A")
        scrolled = self.perception.act("scroll", dy=300)
        self.assertEqual(scrolled["result"]["action"], "scroll")
        button = self.perception.locate("Search button")[
            "candidates"
        ][0]
        hovered = self.perception.act(
            "hover", element_id=button["element_id"]
        )
        self.assertEqual(hovered["result"]["action"], "hover")
        doubled = self.perception.act(
            "double_click", element_id=button["element_id"]
        )
        self.assertEqual(
            doubled["result"]["action"], "double_click"
        )
        self.assertIn(
            ("click", 140.0, 20.0, 2), self.backend.mouse_events
        )

    def test_check_and_uncheck(self):
        backend = PerceptionBackend()
        backend.add_page(
            "https://form.example/", title="Form",
            text="Preferences",
            elements=[
                el("input", "", {"type": "checkbox",
                                 "aria-label": "Subscribe"}),
            ],
        )
        perception, controller, _ = make_perception(backend)
        controller.new_tab("https://form.example/")
        box = perception.locate("Subscribe checkbox")["candidates"][0]
        checked = perception.act(
            "check", element_id=box["element_id"]
        )
        self.assertTrue(checked["result"]["checked"])
        unchecked = perception.act(
            "uncheck", element_id=box["element_id"]
        )
        self.assertFalse(unchecked["result"]["checked"])
        controller.shutdown()


class TestMultiTabSafety(unittest.TestCase):
    def test_wrong_tab_action_is_refused(self):
        backend = PerceptionBackend()
        search_site(backend)
        perception, controller, _ = make_perception(backend)
        tab_one = controller.new_tab("https://search.example/")
        box = perception.locate("Search box")["candidates"][0]
        tab_two = controller.new_tab("https://results.example/")
        with self.assertRaises(BrowserException) as caught:
            perception.act(
                "type", element_id=box["element_id"],
                tab_id=tab_two["tab_id"], text="nope",
            )
        self.assertEqual(
            caught.exception.error.code,
            BrowserErrorCode.INVALID_TAB,
        )
        # Acting in the element's own tab still works.
        typed = perception.act(
            "type", element_id=box["element_id"],
            tab_id=tab_one["tab_id"], text="cats",
        )
        self.assertEqual(typed["tab_id"], tab_one["tab_id"])
        controller.shutdown()


class TestConfidenceGating(unittest.TestCase):
    def _login_page(self, backend):
        backend.add_page(
            "https://login.example/", title="Login",
            text="Welcome",
            elements=[
                el("a", "Login",
                   {"href": "https://home.example/"}),
            ],
        )
        backend.add_page(
            "https://home.example/", title="Home",
            text="Welcome home", elements=[],
        )

    def test_uncertain_target_needs_approval(self):
        backend = PerceptionBackend()
        self._login_page(backend)
        perception, controller, _ = make_perception(backend)
        controller.new_tab("https://login.example/")
        found = perception.locate("Login button")
        best = found["candidates"][0]
        self.assertEqual(best["tier"], "uncertain")
        with self.assertRaises(BrowserException) as caught:
            perception.act("click", element_id=best["element_id"])
        self.assertIn(
            caught.exception.error.code,
            (
                BrowserErrorCode.APPROVAL_REQUIRED,
                BrowserErrorCode.APPROVAL_DENIED,
            ),
        )
        controller.shutdown()

    def test_uncertain_target_proceeds_once_approved(self):
        backend = PerceptionBackend()
        self._login_page(backend)
        gate = ApprovalGate(approver=lambda request: True)
        perception, controller, _ = make_perception(
            backend, approval_gate=gate
        )
        controller.new_tab("https://login.example/")
        found = perception.locate("Login button")
        clicked = perception.act(
            "click",
            element_id=found["candidates"][0]["element_id"],
            expect_url_contains="home.example",
        )
        self.assertTrue(clicked["verified"])
        controller.shutdown()


class TestVisualFallback(unittest.TestCase):
    def _canvas_page(self, backend):
        backend.add_page(
            "https://canvas.example/", title="Canvas",
            text="A canvas app", elements=[],
        )

    def test_visual_click_when_no_dom_elements(self):
        backend = PerceptionBackend()
        self._canvas_page(backend)
        perception, controller, _ = make_perception(backend)
        perception.screen_observer = StubScreen(confidence=0.9)
        controller.new_tab("https://canvas.example/")
        observed = perception.observe()
        visual = [
            e for e in observed["elements"]
            if e["source"] == "visual"
        ]
        self.assertEqual(len(visual), 1)
        clicked = perception.act(
            "click", element_id=visual[0]["element_id"]
        )
        self.assertEqual(clicked["result"]["action"], "click_at")
        self.assertIn(
            ("click", 60.0, 25.0, 1), backend.mouse_events
        )
        controller.shutdown()

    def test_low_confidence_visual_click_needs_approval(self):
        backend = PerceptionBackend()
        self._canvas_page(backend)
        perception, controller, _ = make_perception(backend)
        perception.screen_observer = StubScreen(confidence=0.6)
        controller.new_tab("https://canvas.example/")
        observed = perception.observe()
        visual = observed["elements"][0]
        with self.assertRaises(BrowserException) as caught:
            perception.act(
                "click", element_id=visual["element_id"]
            )
        self.assertEqual(
            caught.exception.error.code,
            BrowserErrorCode.APPROVAL_REQUIRED,
        )
        self.assertEqual(backend.mouse_events, [])
        controller.shutdown()


class TestPerceptionTools(unittest.TestCase):
    def test_tools_run_the_workflow(self):
        backend = PerceptionBackend()
        search_site(backend)
        controller = BrowserController(
            runtime=AfnanBrowserRuntime(adapter=backend)
        )
        registry = ToolRegistry()
        for tool in create_browser_tools(controller):
            registry.register(tool)
        registry.execute("browser_launch", {})
        tab = registry.execute(
            "browser_new_tab", {"url": "https://search.example/"}
        )
        self.assertTrue(tab.success)
        perceived = registry.execute("browser_perceive", {})
        self.assertTrue(perceived.success)
        located = registry.execute(
            "browser_locate", {"description": "Search box"}
        )
        element_id = located.output["candidates"][0]["element_id"]
        typed = registry.execute(
            "browser_computer_act",
            {"action": "type", "element_id": element_id,
             "text": "cats"},
        )
        self.assertTrue(typed.success)
        self.assertEqual(
            typed.output["result"]["element"]["value"], "cats"
        )
        controller.shutdown()

    def test_perception_is_recorded_in_agent_state(self):
        backend = PerceptionBackend()
        search_site(backend)
        agent, _ = make_agent(
            [
                '{"goal": "look", "steps": [{"step_id": "s1", '
                '"description": "Perceive the page", '
                '"tool_name": "browser_perceive", "arguments": {}, '
                '"expected_result": "page perceived"}]}'
            ],
            backend=backend,
        )
        agent.browser.launch()
        agent.browser.new_tab("https://search.example/")
        result = agent.orchestrator.run("look")
        self.assertTrue(result.state.observations)
        summaries = " ".join(
            o.text for o in result.state.observations
        )
        self.assertIn("Perceived page", summaries)


if __name__ == "__main__":
    unittest.main()

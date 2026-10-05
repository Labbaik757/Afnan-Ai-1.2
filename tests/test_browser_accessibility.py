"""Accessibility tree + semantic locator tests.

The browser's accessibility tree is the preferred source; the
DOM-derived tree is the fallback; semantic descriptions rank
candidates with confidence scores, and low-confidence guesses
never expose an actionable element reference.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai.agent import AfnanAgent
from afnan_ai.browser import BrowserController
from afnan_ai.browser.backend import BrowserBackend
from afnan_ai.browser.semantics import rank
from afnan_ai.llm.base import LLMProvider


class FakeElement:
    def __init__(self, tag, attrs=None, text=""):
        self.tag = tag
        self.attrs = dict(attrs or {})
        self.text = text
        self.value = ""
        self.clicked = 0

    @property
    def editable(self):
        return self.tag in ("input", "textarea", "select")


SNAPSHOT = {
    "role": "WebArea", "name": "Docs home",
    "children": [
        {"role": "heading", "name": "Welcome to the docs", "level": 1},
        {"role": "textbox", "name": "Search documentation"},
        {"role": "button", "name": "Sign in"},
        {"role": "navigation", "name": "", "children": [
            {"role": "link", "name": "Home"},
            {"role": "link", "name": "Next page"},
        ]},
        {"role": "checkbox", "name": "Remember me"},
    ],
}


class FakePage:
    def __init__(self, with_snapshot=True):
        self.url = "https://docs.example/"
        self.title = "Docs home"
        self.with_snapshot = with_snapshot
        self.elements = [
            FakeElement("input", {"id": "q", "type": "search",
                                  "placeholder": "Search documentation"}),
            FakeElement("button", {"id": "signin"}, text="Sign in"),
            FakeElement("a", {"id": "next", "href": "/page2"},
                        text="Next page"),
            FakeElement("input", {"id": "remember", "type": "checkbox"}),
        ]


class FakeBackend(BrowserBackend):
    name = "fake-a11y"

    def __init__(self, with_snapshot=True):
        self.page = FakePage(with_snapshot)
        self.started = False

    def start(self, browser, headless):
        self.started = True

    def connect(self, endpoint):
        self.started = True

    def stop(self):
        self.started = False

    def new_page(self):
        return self.page

    def close_page(self, handle):
        pass

    def goto(self, handle, url):
        handle.url = url

    def go_back(self, handle):
        pass

    def go_forward(self, handle):
        pass

    def reload(self, handle):
        pass

    def page_url(self, handle):
        return handle.url

    def page_title(self, handle):
        return handle.title

    def query_elements(self, handle, locator, limit):
        text = locator.get("text", "")
        sel = locator.get("selector", "")
        found = []
        for el in handle.elements:
            if sel.startswith("#") and el.attrs.get("id") == sel[1:]:
                found.append(el)
            elif text and text in el.text:
                found.append(el)
        return found[:limit]

    def element_info(self, handle, element):
        return {
            "tag": element.tag,
            "text": element.text,
            "attributes": dict(element.attrs),
            "visible": True,
            "enabled": True,
            "editable": element.editable,
            "value": element.value,
        }

    def click_element(self, handle, element, timeout_ms):
        element.clicked += 1

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

    def page_text(self, handle):
        return "Welcome to the docs"

    def interactive_elements(self, handle, limit):
        return list(handle.elements)[:limit]

    def wait_for(self, handle, spec, timeout_ms):
        pass

    def screenshot(self, handle):
        return b"\x89PNG fake"

    def accessibility_snapshot(self, handle):
        if not handle.with_snapshot:
            from afnan_ai.browser.base import (
                BrowserErrorCode, BrowserException,
            )
            raise BrowserException(
                "no accessibility tree here",
                code=BrowserErrorCode.OPERATION_FAILED,
            )
        return SNAPSHOT


def make_controller(with_snapshot=True):
    backend = FakeBackend(with_snapshot)
    controller = BrowserController(backend=backend)
    controller.launch()
    controller.new_tab("https://docs.example/")
    return controller, backend


class TestAccessibilityTree(unittest.TestCase):
    def test_snapshot_preferred_and_normalized(self):
        controller, _ = make_controller(with_snapshot=True)
        tree = controller.accessibility_tree()
        self.assertEqual(tree["source"], "accessibility")
        roles = {n["node_id"]: n for n in tree["nodes"]}
        headings = [n for n in tree["nodes"] if n["role"] == "heading"]
        self.assertEqual(headings[0]["name"], "Welcome to the docs")
        self.assertEqual(headings[0]["level"], 1)
        names = {n["name"] for n in tree["nodes"]}
        self.assertIn("Sign in", names)
        self.assertIn("Next page", names)
        self.assertTrue(all(n["tab_id"] == "tab_1" for n in tree["nodes"]))
        # nested children are flattened with depth
        home = [n for n in tree["nodes"] if n["name"] == "Home"][0]
        self.assertEqual(home["depth"], 2)

    def test_dom_derivation_fallback(self):
        controller, _ = make_controller(with_snapshot=False)
        tree = controller.accessibility_tree()
        self.assertEqual(tree["source"], "dom")
        by_name = {n["name"]: n for n in tree["nodes"]}
        self.assertEqual(
            by_name["Search documentation"]["role"], "searchbox"
        )
        self.assertEqual(by_name["Sign in"]["role"], "button")
        self.assertEqual(by_name["Next page"]["role"], "link")
        # derived nodes are actionable: they carry live refs
        self.assertTrue(by_name["Sign in"]["element_ref"])

    def test_derived_nodes_clickable_via_ref(self):
        controller, backend = make_controller(with_snapshot=False)
        tree = controller.accessibility_tree()
        signin = [n for n in tree["nodes"] if n["name"] == "Sign in"][0]
        controller.click({"ref": signin["element_ref"]})
        el = next(e for e in backend.page.elements
                  if e.attrs.get("id") == "signin")
        self.assertEqual(el.clicked, 1)


class TestSemanticLocator(unittest.TestCase):
    def setUp(self):
        controller, _ = make_controller(with_snapshot=True)
        self.nodes = controller.accessibility_tree()["nodes"]

    def test_login_button(self):
        matches = rank("Login button", self.nodes)
        self.assertEqual(matches[0]["name"], "Sign in")
        self.assertGreaterEqual(matches[0]["confidence"], 0.8)
        self.assertEqual(matches[0]["tier"], "actionable")

    def test_search_field(self):
        matches = rank("Search field", self.nodes)
        self.assertEqual(matches[0]["name"], "Search documentation")
        self.assertIn(matches[0]["role"], ("searchbox", "textbox"))
        self.assertGreaterEqual(matches[0]["confidence"], 0.8)

    def test_next_page_link(self):
        matches = rank("Next page link", self.nodes)
        self.assertEqual(matches[0]["name"], "Next page")
        self.assertEqual(matches[0]["role"], "link")

    def test_low_confidence_guess(self):
        matches = rank("Quantum banana hammock", self.nodes)
        self.assertLess(matches[0]["confidence"], 0.5)
        self.assertEqual(matches[0]["tier"], "low")


class _QueueLLM(LLMProvider):
    name = "queue"
    display_name = "Queue"
    model = "queue-1"

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages):
        if not self.replies:
            raise AssertionError("QueueLLM ran out of replies")
        return self.replies.pop(0)


class TestSemanticToolGuard(unittest.TestCase):
    def test_low_confidence_ref_withheld(self):
        backend = FakeBackend(with_snapshot=False)
        controller = BrowserController(backend=backend)
        agent = AfnanAgent(
            llm_provider=_QueueLLM(["{}"]), browser_controller=controller,
            enable_screen_tools=False,
        )
        agent.execute_tool("browser_launch", {})
        agent.execute_tool(
            "browser_navigate", {"url": "https://docs.example/"}
        )
        result = agent.execute_tool(
            "browser_find_semantic",
            {"description": "Quantum banana hammock"},
        )
        self.assertTrue(result.success)
        self.assertTrue(result.output["uncertain"])
        self.assertIsNone(result.output["matches"][0]["element_ref"])

        good = agent.execute_tool(
            "browser_find_semantic", {"description": "Login button"}
        )
        self.assertFalse(good.output["uncertain"])
        self.assertIsNotNone(good.output["matches"][0]["element_ref"])


if __name__ == "__main__":
    unittest.main()

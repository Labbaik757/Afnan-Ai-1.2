"""Multi-tab task management, web search → open → extract, and
page content extraction tests, against a fake web.

Covers: tab purposes and wrong-tab safety, AgentState tab
association, structured search results (with redirect
unwrapping), opening results into purpose-tagged tabs, clean
content extraction (boilerplate filtering, chunking, table
structure, secret redaction), failure handling, and a full
orchestrated search-to-extraction task.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai.agent import AfnanAgent
from afnan_ai.browser import BrowserController, create_browser_tools
from afnan_ai.browser.backend import BrowserBackend
from afnan_ai.browser.extraction import clean_document
from afnan_ai.llm.base import LLMProvider
from afnan_ai.orchestrator import OrchestrationStatus


# ----------------------------------------------------------------------
# Fake web
# ----------------------------------------------------------------------

class FakeElement:
    def __init__(self, tag, attrs=None, text="", selectors=()):
        self.tag = tag
        self.attrs = dict(attrs or {})
        self.text = text
        self.value = ""
        self.selectors = set(selectors)
        self.clicked = 0

    @property
    def editable(self):
        return self.tag in ("input", "textarea", "select")


ARTICLE_DOC = {
    "title": "Python Guide",
    "url": "https://example.com/python-guide",
    "headings": [
        {"level": 1, "text": "Python Guide"},
        {"level": 2, "text": "Getting started"},
    ],
    "paragraphs": [
        "Home",  # navigation fragment — filtered
        "Python is a programming language known for readable "
        "syntax and a large ecosystem of libraries.",
        "All rights reserved by Example Corp.",  # boilerplate
        "You can install Python from the official website and "
        "start writing scripts within minutes.",
        "Your session token: abcdef123456 should stay private "
        "and never be shared with third parties.",
    ],
    "lists": [["Readable syntax", "Huge ecosystem", "Great community"]],
    "links": [
        {"text": "Official site",
         "url": "https://www.python.org/?session_id=sekrit99"},
        {"text": "", "url": "https://example.com/empty"},
        {"text": "Docs", "url": "https://docs.python.org"},
    ],
    "tables": [{
        "caption": "Versions",
        "rows": [["Version", "Year"], ["3.12", "2023"], ["3.13", "2024"]],
    }],
    "text": "Python Guide full text",
}


class FakeWebPage:
    def __init__(self, kind):
        self.kind = kind
        self.history = ["about:blank"]
        self.index = 0
        self.elements = []
        self.clicked_ids = []
        self._load("about:blank")

    @property
    def url(self):
        return self.history[self.index]

    @property
    def title(self):
        return {
            "serp": "DuckDuckGo",
            "article": "Python Guide",
            "other": "Page",
            "blank": "",
        }[self.kind]

    @property
    def text(self):
        if self.kind == "serp":
            return "Search results for your query"
        if self.kind == "article":
            return "Python is a programming language article"
        return "Some page"

    def goto(self, url):
        self.history = self.history[: self.index + 1] + [url]
        self.index += 1
        self._load(url)

    def _load(self, url):
        if "duckduckgo.com" in url:
            self.kind = "serp"
            self.elements = [
                FakeElement("a", {
                    "href": "//duckduckgo.com/l/?uddg=https%3A%2F%2F"
                            "example.com%2Fpython-guide&rut=x",
                }, text="Python Guide — Example",
                    selectors=("a.result__a", ".result__a")),
                FakeElement("a", {
                    "href": "https://www.python.org/",
                }, text="Welcome to Python.org",
                    selectors=("a.result__a", ".result__a")),
                FakeElement("div", {}, text="Python is a programming "
                    "language known for readable syntax.",
                    selectors=(".result__snippet",)),
                FakeElement("div", {}, text="The official home of "
                    "the Python programming language.",
                    selectors=(".result__snippet",)),
            ]
        elif "example.com/python-guide" in url:
            self.kind = "article"
            self.elements = [
                FakeElement("a", {"href": "https://www.python.org/"},
                            text="Official site"),
            ]
        elif url == "about:blank":
            self.kind = "blank"
            self.elements = []
        else:
            self.kind = "other"
            self.elements = [
                FakeElement("button", {"id": "act"}, text="Act"),
            ]


class FakeWebBackend(BrowserBackend):
    name = "fake-web"

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
        page = FakeWebPage("blank")
        self.pages.append(page)
        return page

    def close_page(self, handle):
        self.pages.remove(handle)

    def goto(self, handle, url):
        handle.goto(url)

    def go_back(self, handle):
        if handle.index > 0:
            handle.index -= 1

    def go_forward(self, handle):
        if handle.index < len(handle.history) - 1:
            handle.index += 1

    def reload(self, handle):
        pass

    def page_url(self, handle):
        return handle.url

    def page_title(self, handle):
        return handle.title

    def list_pages(self):
        return list(self.pages)

    def query_elements(self, handle, locator, limit):
        sel = locator.get("selector", "")
        text = locator.get("text", "")
        found = []
        for el in handle.elements:
            if sel and sel in el.selectors:
                found.append(el)
            elif sel.startswith("#") and el.attrs.get("id") == sel[1:]:
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
        handle.clicked_ids.append(element.attrs.get("id"))

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
        return handle.text

    def interactive_elements(self, handle, limit):
        return list(handle.elements)[:limit]

    def wait_for(self, handle, spec, timeout_ms):
        pass

    def screenshot(self, handle):
        return b"\x89PNG fake"

    def page_content(self, handle):
        if handle.kind == "article":
            return dict(ARTICLE_DOC)
        return {}


def make_controller():
    backend = FakeWebBackend()
    controller = BrowserController(backend=backend)
    create_browser_tools(controller)  # creates the research session
    controller.launch()
    return controller, backend


# ----------------------------------------------------------------------
# Multi-tab task management
# ----------------------------------------------------------------------

class TestTabTaskManager(unittest.TestCase):
    def test_purposes_lifecycle(self):
        controller, backend = make_controller()
        news = controller.new_tab(
            "https://news.example/", purpose="news research"
        )
        docs = controller.new_tab(
            "https://example.com/python-guide", purpose="docs"
        )
        self.assertEqual(news["purpose"], "news research")
        tabs = controller.list_tabs()
        by_id = {t["tab_id"]: t for t in tabs}
        self.assertEqual(by_id[news["tab_id"]]["purpose"], "news research")
        self.assertEqual(by_id[docs["tab_id"]]["purpose"], "docs")
        self.assertTrue(by_id[docs["tab_id"]]["active"])

        controller.set_tab_purpose(news["tab_id"], "pricing research")
        self.assertEqual(
            controller.tab_purpose(news["tab_id"]), "pricing research"
        )
        controller.close_tab(news["tab_id"])
        tabs = controller.list_tabs()
        self.assertNotIn(news["tab_id"], {t["tab_id"] for t in tabs})

    def test_actions_hit_only_the_addressed_tab(self):
        controller, backend = make_controller()
        tab_a = controller.new_tab("https://a.example/")
        tab_b = controller.new_tab("https://b.example/")
        controller.click({"selector": "#act"}, tab_id=tab_a["tab_id"])
        page_a, page_b = backend.pages
        self.assertEqual(page_a.clicked_ids, ["act"])
        self.assertEqual(page_b.clicked_ids, [])


# ----------------------------------------------------------------------
# Extraction (unit)
# ----------------------------------------------------------------------

class TestExtraction(unittest.TestCase):
    def test_clean_document(self):
        content = clean_document(dict(ARTICLE_DOC))
        self.assertTrue(content["content_found"])
        self.assertTrue(content["structured"])
        paragraphs = " ".join(content["paragraphs"])
        self.assertIn("readable", paragraphs)
        self.assertNotIn("All rights reserved", paragraphs)
        # secrets redacted from text and URLs
        self.assertNotIn("abcdef123456", paragraphs)
        self.assertNotIn("sekrit99", json.dumps(content["links"]))
        # table structure preserved
        table = content["tables"][0]
        self.assertEqual(table["rows"][0], ["Version", "Year"])
        self.assertEqual(table["rows"][2], ["3.13", "2024"])
        self.assertEqual(table["column_count"], 2)
        # empty-text links dropped
        self.assertEqual(len(content["links"]), 2)

    def test_chunking_bounds_content(self):
        doc = dict(ARTICLE_DOC)
        doc["paragraphs"] = [
            f"Paragraph {i} " + "with plenty of useful content. " * 12
            for i in range(6)
        ]
        content = clean_document(doc, max_chars=500)
        self.assertGreater(content["chunk_count"], 1)
        for chunk in content["chunks"]:
            self.assertLessEqual(chunk["char_count"], 500)
            self.assertEqual(
                chunk["token_estimate"], chunk["char_count"] // 4
            )

    def test_empty_page_is_honest(self):
        content = clean_document({"title": "Blank", "url": "x"})
        self.assertFalse(content["content_found"])
        self.assertEqual(content["chunk_count"], 0)


# ----------------------------------------------------------------------
# Search → open → extract
# ----------------------------------------------------------------------

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


def step(step_id, tool, arguments, expected):
    return {
        "step_id": step_id, "description": f"{tool} step",
        "tool_name": tool, "arguments": arguments,
        "expected_result": expected,
    }


def plan_json(goal, steps):
    return json.dumps({"goal": goal, "steps": steps})


def make_agent(replies):
    backend = FakeWebBackend()
    controller = BrowserController(backend=backend)
    agent = AfnanAgent(
        llm_provider=QueueLLM(replies), browser_controller=controller,
        enable_screen_tools=False,
    )
    return agent, backend


class TestResearchWorkflow(unittest.TestCase):
    def test_search_extracts_structured_results(self):
        controller, backend = make_controller()
        controller.new_tab()
        research = controller.research_session
        out = research.search("python guide")
        self.assertEqual(out["count"], 2)
        first = out["results"][0]
        self.assertEqual(first["rank"], 1)
        self.assertIn("Python Guide", first["title"])
        # engine redirect unwrapped to the real destination
        self.assertEqual(first["url"], "https://example.com/python-guide")
        self.assertIn("readable syntax", first["snippet"])
        self.assertEqual(first["source"], "duckduckgo")

    def test_open_result_extracts_and_tags_tab(self):
        controller, backend = make_controller()
        controller.new_tab()
        research = controller.research_session
        research.search("python guide")
        opened = research.open_result(0)
        self.assertIn("Search result:", opened["purpose"])
        content = opened["content"]
        self.assertTrue(content["content_found"])
        self.assertEqual(content["counts"]["tables"], 1)
        blob = json.dumps(opened)
        self.assertNotIn("abcdef123456", blob)
        self.assertNotIn("sekrit99", blob)
        tabs = controller.list_tabs()
        self.assertTrue(
            any("python-guide" in t["url"] for t in tabs)
        )

    def test_search_failures_are_structured(self):
        controller, backend = make_controller()
        controller.new_tab("https://a.example/")
        research = controller.research_session
        from afnan_ai.browser.base import BrowserException
        with self.assertRaises(BrowserException):
            research.open_result(0)  # nothing searched yet
        out = None
        with self.assertRaises(BrowserException):
            research.search("")  # empty query
        # searching while on a non-SERP start still navigates to
        # the engine first, so it succeeds; a page that yields no
        # results is the failure case:
        research.search("python guide")
        with self.assertRaises(BrowserException):
            research.open_result(99)  # out of range

    def test_orchestrated_search_to_extraction(self):
        plan = plan_json("Research python", [
            step("s1", "browser_launch", {}, "Browser is running"),
            step("s2", "browser_search", {"query": "python guide"},
                 "Duckduckgo search results for query"),
            step("s3", "browser_open_result", {"index": 0},
                 "Search result article opened"),
            step("s4", "browser_extract_content", {},
                 "Python article content extracted"),
        ])
        agent, backend = make_agent([plan])
        result = agent.orchestrator.run("Research python")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        state_blob = result.state.to_json()
        self.assertIn("search_results", state_blob)
        self.assertNotIn("abcdef123456", state_blob)
        self.assertNotIn("sekrit99", state_blob)
        tabs = agent.browser.list_tabs()
        self.assertTrue(
            any(t["purpose"].startswith("Search result:") for t in tabs)
        )

    def test_tab_purposes_recorded_in_state(self):
        plan = plan_json("Parallel research", [
            step("s1", "browser_launch", {}, "Browser is running"),
            step("s2", "browser_new_tab",
                 {"url": "https://example.com/python-guide",
                  "purpose": "docs research"},
                 "New tab opened at python guide with purpose"),
            step("s3", "browser_list_tabs", {},
                 "Tabs listed with docs research purpose"),
        ])
        agent, backend = make_agent([plan])
        result = agent.orchestrator.run("Parallel research")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertIn("docs research", result.state.to_json())


if __name__ == "__main__":
    unittest.main()

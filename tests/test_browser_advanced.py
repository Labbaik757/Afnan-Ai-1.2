"""Advanced browser capabilities: SPA awareness, pagination /
infinite scroll, CAPTCHA human_required handling, the download
manager, and the history/session manager.

Everything runs against a fake dynamic backend — no real browser,
network or downloads.  Security invariant checked throughout: no
secret ever lands in history, state or results, and a CAPTCHA is
detected and handed to a human, never solved or bypassed.
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from afnan_ai.agent import AfnanAgent
from afnan_ai.browser.backend import BrowserBackend
from afnan_ai.browser.base import BrowserErrorCode, BrowserException
from afnan_ai.browser.challenge import ChallengeDetector
from afnan_ai.browser.controller import BrowserController
from afnan_ai.browser.pagination import Paginator
from afnan_ai.browser.tools import create_browser_tools
from afnan_ai.llm.base import LLMProvider
from afnan_ai.tools.base import ToolExecutionError
from afnan_ai.tools.registry import ToolRegistry

# ----------------------------------------------------------------------
# Fake dynamic backend
# ----------------------------------------------------------------------


def el(tag, text="", attrs=None, **kw):
    element = {
        "tag": tag,
        "text": text,
        "attributes": attrs or {},
        "visible": True,
        "enabled": True,
        "editable": tag in ("input", "textarea"),
        "value": "",
        "options": [],
    }
    element.update(kw)
    return element


class DynamicBackend(BrowserBackend):
    """A fake SPA backend: probe states evolve, scrolls append item
    batches, Next links switch pages, downloads are pre-seeded."""

    name = "dynamic-fake"

    def __init__(self):
        self.started = False
        self.pages = {}          # url -> page dict
        self.handles = {}        # handle id -> page dict
        self._next_handle = 0
        self.current = None
        self.crashed = False
        self.download_list = []
        self.probe_supported = True

    # -- page fixtures -----------------------------------------------------

    def add_page(self, url, *, title="", text="", elements=(),
                 probe_plan=None, probe_cycle=None, batches=None,
                 batch_size=0):
        self.pages[url] = {
            "url": url,
            "title": title,
            "text": text,
            "elements": [dict(e) for e in elements],
            "probe_plan": list(probe_plan or []),
            "probe_cycle": list(probe_cycle or []),
            "probe_idx": 0,
            "batches": list(batches or []),
            "batch_size": batch_size,
            "scrolls": 0,
        }
        return self.pages[url]

    # -- BrowserBackend interface ------------------------------------------

    def start(self, browser, headless):
        self.started = True

    def connect(self, endpoint):
        self.started = True

    def reload(self, handle):
        pass

    def stop(self):
        self.started = False

    def _page(self, handle):
        if self.crashed:
            raise RuntimeError("renderer crashed")
        return self.handles[handle]

    def new_page(self):
        self._next_handle += 1
        handle = f"h{self._next_handle}"
        self.handles[handle] = {
            "url": "about:blank", "title": "", "text": "",
            "elements": [], "probe_plan": [], "probe_cycle": [],
            "probe_idx": 0, "batches": [], "batch_size": 0,
            "scrolls": 0,
        }
        return handle

    def close_page(self, handle):
        self.handles.pop(handle, None)

    def goto(self, handle, url):
        if self.crashed:
            raise RuntimeError("renderer crashed")
        if url not in self.pages:
            raise RuntimeError(f"no page at {url}")
        self.handles[handle] = self.pages[url]

    def go_back(self, handle):
        pass

    def go_forward(self, handle):
        pass

    def page_url(self, handle):
        return self._page(handle)["url"]

    def page_title(self, handle):
        return self._page(handle)["title"]

    def page_text(self, handle):
        return self._page(handle)["text"]

    def interactive_elements(self, handle, limit):
        page = self._page(handle)
        return [
            (page["url"], i)
            for i, e in enumerate(page["elements"])
            if e["tag"] in ("a", "button", "input", "select", "textarea")
        ][:limit]

    def element_info(self, handle, el_handle):
        page = self._page(handle)
        return dict(page["elements"][el_handle[1]])

    def query_elements(self, handle, locator, limit):
        page = self._page(handle)
        found = []
        for i, e in enumerate(page["elements"]):
            if "selector" in locator:
                sel = locator["selector"]
                classes = (e["attributes"].get("class") or "").split()
                if sel.startswith(".") and sel[1:] in classes:
                    found.append((page["url"], i))
                elif sel == e["tag"]:
                    found.append((page["url"], i))
            elif "text" in locator:
                if e["text"] == locator["text"]:
                    found.append((page["url"], i))
            elif "role" in locator:
                if e["attributes"].get("role") == locator["role"] and (
                    e["text"] == locator.get("name")
                    or e["attributes"].get("aria-label")
                    == locator.get("name")
                ):
                    found.append((page["url"], i))
        return found

    def click_element(self, handle, el_handle, timeout_ms):
        page = self._page(handle)
        element = page["elements"][el_handle[1]]
        href = element["attributes"].get("href")
        if href and href in self.pages:
            self.handles[handle] = self.pages[href]

    def fill_element(self, handle, el_handle, text, timeout_ms):
        page = self._page(handle)
        page["elements"][el_handle[1]]["value"] = text

    def clear_element(self, handle, el_handle, timeout_ms):
        page = self._page(handle)
        page["elements"][el_handle[1]]["value"] = ""

    def scroll_page(self, handle, dx, dy):
        page = self._page(handle)
        page["scrolls"] += 1
        if page["batches"]:
            page["elements"].extend(page["batches"].pop(0))
        elif page["batch_size"]:
            start = len(page["elements"]) + 1
            page["elements"].extend(
                el("a", f"Generated item {n}",
                   {"class": "item",
                    "href": f"https://gen.example/{n}"})
                for n in range(start, start + page["batch_size"])
            )

    def page_probe(self, handle):
        if not self.probe_supported:
            raise RuntimeError("probe not supported by this driver")
        page = self._page(handle)
        plan = page["probe_plan"] or page["probe_cycle"]
        if not plan:
            return {
                "url": page["url"], "title": page["title"],
                "ready_state": "complete", "frameworks": [],
                "text_length": len(page["text"]),
                "element_count": len(page["elements"]),
                "content_hash": f"h-{page['text']}",
            }
        idx = page["probe_idx"]
        page["probe_idx"] = idx + 1
        if page["probe_plan"]:
            # scripted async states; the last one repeats (settled)
            state = page["probe_plan"][
                min(idx, len(page["probe_plan"]) - 1)
            ]
        else:
            cycle = page["probe_cycle"]
            state = cycle[idx % len(cycle)]
        return dict(state)

    def downloads(self):
        return [dict(d) for d in self.download_list]


def probe_state(url="https://spa.example/", text_len=100, hash_="h1",
                frameworks=("react",), title="App"):
    return {
        "url": url, "title": title, "ready_state": "complete",
        "frameworks": list(frameworks), "text_length": text_len,
        "element_count": 10, "content_hash": hash_,
    }


def make_controller(backend=None, **kw):
    backend = backend or DynamicBackend()
    controller = BrowserController(backend=backend, **kw)
    controller.launch()
    controller.new_tab()
    return controller, backend


# ----------------------------------------------------------------------
# 1. SPA awareness
# ----------------------------------------------------------------------


class TestSpaAwareness(unittest.TestCase):
    def test_wait_for_stable_after_async_rendering(self):
        backend = DynamicBackend()
        loading = probe_state(text_len=10, hash_="loading")
        rendering = probe_state(text_len=60, hash_="rendering")
        final = probe_state(text_len=120, hash_="final")
        backend.add_page(
            "https://spa.example/", title="App",
            text="Dashboard widgets ready",
            elements=[el("a", "Reports",
                         {"href": "https://spa.example/reports"})],
            probe_plan=[loading, rendering, final],
        )
        controller, _ = make_controller(backend)
        controller.navigate("https://spa.example/")
        result = controller.wait_for_stable(timeout_ms=2000)
        self.assertTrue(result["stable"])
        self.assertEqual(result["content_hash"], "final")
        self.assertIn("react", result["frameworks"])
        self.assertGreater(result["probes"], 2)

    def test_never_stable_page_times_out(self):
        backend = DynamicBackend()
        backend.add_page(
            "https://spa.example/live", title="Live",
            text="Live ticker",
            probe_cycle=[probe_state(hash_="a"), probe_state(hash_="b")],
        )
        controller, _ = make_controller(backend)
        controller.navigate("https://spa.example/live")
        with self.assertRaises(BrowserException) as ctx:
            controller.wait_for_stable(timeout_ms=300)
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.TIMEOUT
        )

    def test_probe_falls_back_to_page_reads(self):
        backend = DynamicBackend()
        backend.probe_supported = False
        backend.add_page(
            "https://plain.example/", title="Plain",
            text="Static content here",
            elements=[el("a", "About",
                         {"href": "https://plain.example/about"})],
        )
        controller, _ = make_controller(backend)
        controller.navigate("https://plain.example/")
        probe = controller.spa_state()
        self.assertEqual(probe["title"], "Plain")
        self.assertEqual(probe["text_length"], len("Static content here"))
        result = controller.wait_for_stable(timeout_ms=1000)
        self.assertTrue(result["stable"])


# ----------------------------------------------------------------------
# 2. Infinite scroll / pagination
# ----------------------------------------------------------------------


def item(n, prefix="Item"):
    return el("a", f"{prefix} {n}",
              {"class": "item", "href": f"https://shop.example/{n}"})


class TestPagination(unittest.TestCase):
    def _scroll_site(self, backend):
        backend.add_page(
            "https://shop.example/feed", title="Feed",
            text="Feed",
            elements=[item(n) for n in range(1, 4)],
            batches=[
                [item(n) for n in (2, 3, 4, 5)],   # 2 duplicates
                [item(n) for n in (6, 7)],
            ],
        )

    def test_infinite_scroll_collects_and_dedupes(self):
        backend = DynamicBackend()
        self._scroll_site(backend)
        controller, _ = make_controller(backend)
        controller.navigate("https://shop.example/feed")
        result = Paginator(controller).collect(
            mode="scroll", item_selector=".item",
            max_items=50, max_pages=10,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 7)
        self.assertGreater(result["duplicates_skipped"], 0)
        self.assertEqual(result["stopped_reason"], "exhausted")

    def test_max_items_limit_terminates_collection(self):
        backend = DynamicBackend()
        backend.add_page(
            "https://shop.example/endless", title="Endless",
            text="Endless", elements=[item(1)],
            batch_size=5,
        )
        controller, _ = make_controller(backend)
        controller.navigate("https://shop.example/endless")
        result = Paginator(controller).collect(
            mode="scroll", item_selector=".item",
            max_items=8, max_pages=100,
        )
        self.assertEqual(result["count"], 8)
        self.assertEqual(result["stopped_reason"], "max_items")

    def test_max_pages_limit_terminates_collection(self):
        backend = DynamicBackend()
        backend.add_page(
            "https://shop.example/endless2", title="Endless",
            text="Endless", elements=[item(1)],
            batch_size=5,
        )
        controller, _ = make_controller(backend)
        controller.navigate("https://shop.example/endless2")
        result = Paginator(controller).collect(
            mode="scroll", item_selector=".item",
            max_items=500, max_pages=3,
        )
        self.assertEqual(result["pages_visited"], 3)
        self.assertEqual(result["stopped_reason"], "max_pages")

    def test_next_page_pagination(self):
        backend = DynamicBackend()
        backend.add_page(
            "https://shop.example/list?page=1", title="List 1",
            text="List page 1",
            elements=[
                item(1), item(2),
                el("a", "Next",
                   {"href": "https://shop.example/list?page=2"}),
            ],
        )
        backend.add_page(
            "https://shop.example/list?page=2", title="List 2",
            text="List page 2",
            elements=[item(3), item(4)],
        )
        controller, _ = make_controller(backend)
        controller.navigate("https://shop.example/list?page=1")
        result = Paginator(controller).collect(
            mode="next", item_selector=".item",
            max_items=50, max_pages=10,
        )
        self.assertTrue(result["success"])
        self.assertEqual(
            [i["text"] for i in result["items"]],
            ["Item 1", "Item 2", "Item 3", "Item 4"],
        )
        self.assertEqual(result["pages_visited"], 2)
        self.assertEqual(result["stopped_reason"], "next_unavailable")


# ----------------------------------------------------------------------
# 3. CAPTCHA detection — human_required, never bypassed
# ----------------------------------------------------------------------


class TestChallengeDetection(unittest.TestCase):
    def _captcha_site(self, backend):
        backend.add_page(
            "https://site.example/challenge", title="Security Check",
            text="Verify you are human. I'm not a robot.",
            elements=[el("button", "Verify", {"id": "verify"})],
        )
        backend.add_page(
            "https://site.example/home", title="Home",
            text="Welcome home, everything is fine.",
            elements=[el("button", "Continue", {"id": "go"})],
        )

    def test_detector_flags_challenge_page_only(self):
        detector = ChallengeDetector()
        hit = detector.detect({
            "url": "https://x.example/challenge",
            "title": "Security Check",
            "text": "Verify you are human",
            "elements": [],
        })
        self.assertTrue(hit["detected"])
        self.assertTrue(hit["human_required"])
        miss = detector.detect({
            "url": "https://x.example/home", "title": "Home",
            "text": "Welcome home", "elements": [],
        })
        self.assertFalse(miss["detected"])

    def test_click_on_challenge_page_is_human_required(self):
        backend = DynamicBackend()
        self._captcha_site(backend)
        controller, _ = make_controller(backend)
        controller.navigate("https://site.example/challenge")
        found = controller.find_elements({"selector": "button"})
        with self.assertRaises(BrowserException) as ctx:
            controller.click({"ref": found[0]["ref"]})
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.HUMAN_REQUIRED
        )
        self.assertIn(
            "human_required", ctx.exception.error.message
        )

    def test_normal_page_actions_unaffected(self):
        backend = DynamicBackend()
        self._captcha_site(backend)
        controller, _ = make_controller(backend)
        controller.navigate("https://site.example/home")
        found = controller.find_elements({"selector": "button"})
        result = controller.click({"ref": found[0]["ref"]})
        self.assertEqual(result["action"], "click")

    def test_guard_can_be_disabled_explicitly(self):
        backend = DynamicBackend()
        self._captcha_site(backend)
        controller, _ = make_controller(backend, challenge_guard=False)
        controller.navigate("https://site.example/challenge")
        found = controller.find_elements({"selector": "button"})
        result = controller.click({"ref": found[0]["ref"]})
        self.assertEqual(result["action"], "click")


# ----------------------------------------------------------------------
# 4. Download manager
# ----------------------------------------------------------------------


class TestDownloadManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.png = root / "report.png"
        self.png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"data" * 10)
        self.empty = root / "empty.pdf"
        self.empty.write_bytes(b"")
        self.corrupt = root / "fake.png"
        self.corrupt.write_bytes(b"not a png at all")
        self.exe = root / "setup.exe"
        self.exe.write_bytes(b"MZ" + b"\x00" * 20)

    def tearDown(self):
        self.tmp.cleanup()

    def _controller(self, records):
        backend = DynamicBackend()
        backend.download_list = records
        controller, _ = make_controller(backend)
        return controller

    def test_completed_download_verified(self):
        controller = self._controller([{
            "id": "dl_1", "url": "https://files.example/report.png",
            "filename": "report.png", "state": "completed",
            "path": str(self.png), "failure": None,
        }])
        record = controller.download_manager.list_downloads()[0]
        self.assertEqual(record["integrity"], "ok")
        self.assertEqual(record["file_type"], "png")
        self.assertGreater(record["size_bytes"], 0)
        self.assertFalse(record["unsafe"])

    def test_failed_and_corrupt_downloads_reported_honestly(self):
        controller = self._controller([
            {
                "id": "dl_1", "url": "https://files.example/a.zip",
                "filename": "a.zip", "state": "failed",
                "path": None, "failure": "network error",
            },
            {
                "id": "dl_2", "url": "https://files.example/fake.png",
                "filename": "fake.png", "state": "completed",
                "path": str(self.corrupt), "failure": None,
            },
            {
                "id": "dl_3", "url": "https://files.example/empty.pdf",
                "filename": "empty.pdf", "state": "completed",
                "path": str(self.empty), "failure": None,
            },
        ])
        records = controller.download_manager.list_downloads()
        self.assertEqual(records[0]["integrity"], "failed")
        self.assertEqual(records[1]["integrity"], "corrupt")
        self.assertEqual(records[2]["integrity"], "empty_file")

    def test_unsafe_download_flagged_and_never_opened(self):
        controller = self._controller([{
            "id": "dl_9", "url": "https://files.example/setup.exe",
            "filename": "setup.exe", "state": "completed",
            "path": str(self.exe), "failure": None,
        }])
        manager = controller.download_manager
        record = manager.list_downloads()[0]
        self.assertTrue(record["unsafe"])
        self.assertTrue(record["requires_approval"])
        # the manager has no way to open/execute anything
        self.assertFalse(hasattr(manager, "open"))
        self.assertFalse(hasattr(manager, "execute"))
        self.assertFalse(hasattr(manager, "run_file"))

    def test_wait_for_download_timeout_and_secret_redaction(self):
        controller = self._controller([{
            "id": "dl_5",
            "url": "https://files.example/x.zip?token=sekrit123",
            "filename": "x.zip", "state": "started",
            "path": None, "failure": None,
        }])
        manager = controller.download_manager
        with self.assertRaises(BrowserException) as ctx:
            manager.wait_for("dl_5", timeout_ms=150)
        self.assertEqual(ctx.exception.code, BrowserErrorCode.TIMEOUT)
        record = manager.list_downloads()[0]
        self.assertNotIn("sekrit123", record["url"])


# ----------------------------------------------------------------------
# 5. History / session manager
# ----------------------------------------------------------------------


class TestSessionManager(unittest.TestCase):
    def _site(self, backend):
        backend.add_page(
            "https://a.example/", title="A", text="Page A",
            elements=[el("a", "to B", {"href": "https://b.example/"})],
        )
        backend.add_page(
            "https://b.example/", title="B", text="Page B", elements=[],
        )

    def test_history_records_navigation_with_purposes(self):
        backend = DynamicBackend()
        self._site(backend)
        controller, _ = make_controller(backend)
        controller.navigate("https://a.example/")
        tab2 = controller.new_tab(
            "https://b.example/", purpose="price check"
        )
        history = controller.session_manager.history()
        actions = [(h["action"], h["tab_id"]) for h in history]
        self.assertIn(("navigate", "tab_1"), actions)
        self.assertIn(("new_tab", tab2["tab_id"]), actions)
        entry = [h for h in history if h["tab_id"] == tab2["tab_id"]][-1]
        self.assertEqual(entry["purpose"], "price check")

    def test_history_redacts_tokens_in_urls(self):
        backend = DynamicBackend()
        backend.add_page(
            "https://a.example/?api_key=sekrit999", title="A",
            text="Page A", elements=[],
        )
        controller, _ = make_controller(backend)
        controller.navigate("https://a.example/?api_key=sekrit999")
        history = controller.session_manager.history()
        self.assertNotIn("sekrit999", json.dumps(history))

    def test_save_load_restore_recovers_context(self):
        backend = DynamicBackend()
        self._site(backend)
        controller, _ = make_controller(backend)
        controller.navigate("https://a.example/")
        controller.new_tab("https://b.example/", purpose="research")
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "session.json")
            saved = controller.session_manager.save(path)
            self.assertEqual(saved["tabs"], 2)

            backend2 = DynamicBackend()
            self._site(backend2)
            # one saved page is gone in the new session world
            del backend2.pages["https://b.example/"]
            controller2, _ = make_controller(backend2)
            loaded = controller2.session_manager.load(path)
            self.assertEqual(len(loaded["tabs"]), 2)
            result = controller2.session_manager.restore(path)
            self.assertEqual(len(result["restored_tabs"]), 1)
            self.assertEqual(len(result["failed"]), 1)
            self.assertEqual(result["history_entries_recovered"], 4)


# ----------------------------------------------------------------------
# Tools + orchestrated end-to-end
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


def tool_by_name(controller, name):
    registry = ToolRegistry()
    for tool in create_browser_tools(controller):
        registry.register(tool)
    return registry.get(name)


class TestAdvancedTools(unittest.TestCase):
    def test_new_tools_registered(self):
        controller, _ = make_controller(DynamicBackend())
        names = {t.name for t in create_browser_tools(controller)}
        for expected in (
            "browser_wait_for_stable", "browser_collect_items",
            "browser_check_challenge", "browser_downloads",
            "browser_session",
        ):
            self.assertIn(expected, names)

    def test_check_challenge_tool_reports_human_required(self):
        backend = DynamicBackend()
        backend.add_page(
            "https://site.example/challenge", title="Security Check",
            text="Verify you are human. I'm not a robot.",
            elements=[el("button", "Verify", {"id": "v"})],
        )
        controller, _ = make_controller(backend)
        controller.navigate("https://site.example/challenge")
        tool = tool_by_name(controller, "browser_check_challenge")
        registry = ToolRegistry()
        for t in create_browser_tools(controller):
            registry.register(t)
        result = registry.execute("browser_check_challenge", {})
        self.assertTrue(result.success)
        self.assertEqual(result.output["status"], "human_required")
        self.assertTrue(result.output["human_required"])

    def test_downloads_tool_lists_records(self):
        backend = DynamicBackend()
        backend.download_list = [{
            "id": "dl_1", "url": "https://f.example/a.pdf",
            "filename": "a.pdf", "state": "completed",
            "path": None, "failure": None,
        }]
        controller, _ = make_controller(backend)
        registry = ToolRegistry()
        for t in create_browser_tools(controller):
            registry.register(t)
        result = registry.execute("browser_downloads", {})
        self.assertTrue(result.success)
        self.assertEqual(result.output["count"], 1)


class TestEndToEnd(unittest.TestCase):
    def test_spa_collect_and_session_task(self):
        backend = DynamicBackend()
        final = probe_state(
            url="https://spa.example/app", text_len=120, hash_="final"
        )
        backend.add_page(
            "https://spa.example/app", title="App",
            text="Dashboard loaded with widgets and reports",
            elements=[item(n) for n in range(1, 4)],
            probe_plan=[probe_state(hash_="boot"), final],
            batches=[[item(4), item(5)]],
        )
        controller, _ = make_controller(backend)
        agent = AfnanAgent(
            llm_provider=QueueLLM([
                plan_json("Research the SPA", [
                    step("s1", "browser_navigate",
                         {"url": "https://spa.example/app"},
                         "Dashboard loaded widgets reports"),
                    step("s2", "browser_wait_for_stable",
                         {"timeout_ms": 2000},
                         "Page stable frameworks react"),
                    step("s3", "browser_collect_items",
                         {"mode": "scroll", "item_selector": ".item",
                          "max_items": 50, "max_pages": 5},
                         "Collected 5 items"),
                ]),
            ]),
            browser_controller=controller,
        )
        result = agent.orchestrator.run("Research the SPA dashboard")
        self.assertEqual(result.status, "completed", result.error)
        state = agent.orchestrator.state
        kinds = [
            o.metadata.get("observation", {}).get("type")
            for o in state.observations
        ]
        self.assertIn("spa_state", kinds)
        self.assertIn("pagination", kinds)

    def test_captcha_task_pauses_as_human_required(self):
        backend = DynamicBackend()
        backend.add_page(
            "https://site.example/challenge", title="Security Check",
            text="Verify you are human. I'm not a robot.",
            elements=[el("button", "Verify", {"id": "verify"})],
        )
        controller, _ = make_controller(backend)
        blocked_plan = plan_json("Open the dashboard", [
            step("s1", "browser_navigate",
                 {"url": "https://site.example/challenge"},
                 "Verify you are human robot"),
            step("s2", "browser_click",
                 {"selector": "button"},
                 "Dashboard opened"),
        ])
        agent = AfnanAgent(
            llm_provider=QueueLLM([blocked_plan] * 4),
            browser_controller=controller,
        )
        result = agent.orchestrator.run("Open the dashboard")
        self.assertNotEqual(result.status, "completed")
        dump = json.dumps(
            agent.orchestrator.state.to_dict(), default=str
        )
        self.assertIn("human_required", dump)

    def test_browser_crash_is_structured_failure_not_crash(self):
        backend = DynamicBackend()
        backend.add_page(
            "https://ok.example/", title="OK", text="All good",
            elements=[],
        )
        controller, _ = make_controller(backend)
        controller.navigate("https://ok.example/")
        backend.crashed = True
        with self.assertRaises(BrowserException) as ctx:
            controller.navigate("https://ok.example/")
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.NAVIGATION_FAILED
        )


if __name__ == "__main__":
    unittest.main()

"""ChromiumAdapter tests.

The adapter is covered at three levels:

* a fake CDP connection (no browser process) for the adapter's
  own logic — lifecycle, navigation, elements, screenshots,
  accessibility tree conversion, profile isolation;
* an engine-isolation scan proving no Playwright/CDP leakage
  outside the adapter layer;
* real-Chromium integration tests (launch, profiles, tabs,
  screenshots, restart, crash recovery, persistence) that run
  only when a Chromium-family executable is installed, plus a
  Playwright-vs-Chromium comparison that runs when both
  engines are available.  Both compare the *normalized*
  controller observation, which is what the agent consumes.
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from afnan_ai.browser.base import BrowserErrorCode, BrowserException
from afnan_ai.browser.chromium_adapter import (
    ChromiumAdapter,
    _CDPConnection,
    _CDPError,
)
from afnan_ai.browser.controller import BrowserController
from afnan_ai.browser.runtime import AfnanBrowserRuntime

from test_browser_advanced import el
from test_browser_runtime import FakeBrowserAdapter


# ----------------------------------------------------------------------
# Fake CDP plumbing
# ----------------------------------------------------------------------


class FakeWS:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent: list[str] = []

    def send_text(self, text):
        self.sent.append(text)

    def recv_text(self, timeout):
        return self.messages.pop(0) if self.messages else None

    def close(self):
        pass


class TestCDPConnection(unittest.TestCase):
    def test_call_matches_response_and_stashes_events(self):
        ws = FakeWS([
            '{"method": "Page.frameNavigated", "params": {}}',
            '{"id": 1, "result": {"ok": true}}',
        ])
        cdp = _CDPConnection(ws)
        result = cdp.call("Runtime.evaluate", {"expression": "1"})
        self.assertEqual(result, {"ok": True})
        events = cdp.drain_events()
        self.assertEqual(events[0]["method"], "Page.frameNavigated")

    def test_call_raises_cdp_error(self):
        ws = FakeWS(['{"id": 1, "error": {"message": "nope"}}'])
        cdp = _CDPConnection(ws)
        with self.assertRaises(_CDPError):
            cdp.call("Bogus.method")


class FakeCDPConnection:
    """Scripted stand-in for a real Chromium CDP connection."""

    def __init__(self):
        self.url = "about:blank"
        self.title = "Example Title"
        self.closed = False
        self._load_pending = False

    def call(self, method, params=None, *, session_id=None, timeout=30.0):
        params = params or {}
        if method == "Browser.getVersion":
            return {"product": "HeadlessChrome/120.0"}
        if method == "Target.createTarget":
            return {"targetId": "target-1"}
        if method == "Target.attachToTarget":
            return {"sessionId": "session-1"}
        if method == "Target.getTargets":
            return {"targetInfos": [{
                "targetId": "target-1", "type": "page",
                "url": self.url,
            }]}
        if method == "Page.navigate":
            self.url = params["url"]
            self._load_pending = True
            return {}
        if method == "Runtime.evaluate":
            return self._evaluate(params.get("expression", ""))
        if method == "Runtime.getProperties":
            return {"result": [
                {"name": "0", "value": {"objectId": "el-1"}},
                {"name": "length", "value": {"value": 1}},
            ]}
        if method == "Runtime.callFunctionOn":
            declaration = params.get("functionDeclaration", "")
            if "getBoundingClientRect" in declaration:
                return {"result": {"value": {
                    "tag": "button", "text": "Go",
                    "attributes": {}, "editable": False,
                    "value": "", "visible": True, "enabled": True,
                }}}
            return {"result": {"value": True}}
        if method == "Page.captureScreenshot":
            return {"data": base64.b64encode(
                b"\x89PNG\r\n\x1a\nfake").decode()}
        if method == "Accessibility.getFullAXTree":
            return {"nodes": [
                {"nodeId": "1", "role": {"value": "RootWebArea"},
                 "name": {"value": "Example"}, "childIds": ["2"]},
                {"nodeId": "2", "parentId": "1",
                 "role": {"value": "button"},
                 "name": {"value": "Go"}},
            ]}
        return {}

    def _evaluate(self, expression):
        if "frameworks" in expression:
            return {"result": {"value": {
                "url": self.url, "title": self.title,
                "ready_state": "complete", "frameworks": [],
                "text_length": 15, "element_count": 4,
                "content_hash": "7",
            }}}
        if "querySelectorAll('p')" in expression:
            return {"result": {"value": {
                "title": self.title, "url": self.url,
                "headings": [], "paragraphs": ["Hello from page"],
                "lists": [], "links": [], "tables": [],
                "text": "Hello from page",
            }}}
        if "location.href" in expression:
            value = self.url
        elif "document.title" in expression:
            value = self.title
        elif "document.body" in expression and "innerText" in expression:
            value = "Hello from page"
        elif "querySelectorAll" in expression:
            return {"result": {"objectId": "arr-1", "type": "object"}}
        else:
            value = None
        return {"result": {"value": value}}

    def poll(self, timeout=0.05):
        pass

    def drain_events(self):
        if self._load_pending:
            self._load_pending = False
            return [{
                "method": "Page.loadEventFired",
                "sessionId": "session-1",
                "params": {},
            }]
        return []

    def close(self):
        self.closed = True


def fake_chromium_adapter(**kwargs) -> ChromiumAdapter:
    """A ChromiumAdapter whose process/CDP layers are faked."""
    adapter = ChromiumAdapter(executable_path="/bin/true", **kwargs)
    adapter._spawn_browser_process = lambda state: "ws://fake"
    adapter._open_cdp = lambda url: FakeCDPConnection()
    return adapter


# ----------------------------------------------------------------------
# Adapter logic against the fake CDP connection
# ----------------------------------------------------------------------


class TestChromiumAdapterFakeCDP(unittest.TestCase):
    def setUp(self):
        self.adapter = fake_chromium_adapter()
        self.adapter.start("chromium", True)

    def tearDown(self):
        self.adapter.stop()

    def test_navigation_observation_and_screenshot(self):
        adapter = self.adapter
        page = adapter.new_page()
        adapter.goto(page, "https://example.test/")
        self.assertEqual(adapter.page_url(page), "https://example.test/")
        self.assertEqual(adapter.page_title(page), "Example Title")
        self.assertEqual(adapter.page_text(page), "Hello from page")
        probe = adapter.page_probe(page)
        self.assertEqual(probe["ready_state"], "complete")
        shot = adapter.screenshot(page)
        self.assertTrue(shot.startswith(b"\x89PNG"))
        self.assertEqual(len(adapter.list_pages()), 1)

    def test_elements_and_accessibility_tree(self):
        adapter = self.adapter
        page = adapter.new_page()
        elements = adapter.interactive_elements(page, 5)
        self.assertEqual(len(elements), 1)
        info = adapter.element_info(page, elements[0])
        self.assertEqual(info["tag"], "button")
        self.assertEqual(info["text"], "Go")
        self.assertTrue(info["visible"])
        adapter.click_element(page, elements[0], 1000)
        tree = adapter.accessibility_snapshot(page)
        self.assertEqual(tree["role"], "rootwebarea")
        self.assertEqual(tree["children"][0]["role"], "button")
        self.assertEqual(tree["children"][0]["name"], "Go")

    def test_capabilities_are_afnan_level(self):
        caps = self.adapter.capabilities()
        self.assertTrue(caps["tabs"])
        self.assertTrue(caps["accessibility"])
        self.assertTrue(caps["persistent_profiles"])
        self.assertFalse(caps["downloads"])

    def test_firefox_is_not_a_chromium(self):
        adapter = fake_chromium_adapter()
        with self.assertRaises(BrowserException) as caught:
            adapter.start("firefox", True)
        self.assertEqual(
            caught.exception.error.code,
            BrowserErrorCode.BROWSER_UNAVAILABLE,
        )

    def test_operations_before_start_are_structured(self):
        adapter = fake_chromium_adapter()
        with self.assertRaises(BrowserException) as caught:
            adapter.new_page()
        self.assertEqual(
            caught.exception.error.code,
            BrowserErrorCode.BROWSER_NOT_STARTED,
        )

    def test_profiles_are_isolated_processes(self):
        adapter = self.adapter
        spawns: list[str] = []
        original = adapter._spawn_browser_process

        def counting_spawn(state):
            spawns.append(state.name)
            return original(state)

        adapter._spawn_browser_process = counting_spawn
        with tempfile.TemporaryDirectory() as tmp:
            adapter.create_profile_context(
                "work", {"user_data_dir": tmp}
            )
            adapter.set_active_profile("work")
            self.assertEqual(adapter._active_profile, "work")
            page = adapter.new_page()
            self.assertEqual(page.profile, "work")
        self.assertIn("work", spawns)
        with self.assertRaises(BrowserException) as caught:
            adapter.set_active_profile("missing")
        self.assertEqual(
            caught.exception.error.code,
            BrowserErrorCode.PROFILE_ERROR,
        )


class TestExecutableDiscovery(unittest.TestCase):
    def test_env_override_wins(self):
        with tempfile.NamedTemporaryFile() as handle:
            old = os.environ.get("AFNAN_CHROMIUM_EXECUTABLE")
            os.environ["AFNAN_CHROMIUM_EXECUTABLE"] = handle.name
            try:
                found = ChromiumAdapter.find_executable("chromium")
            finally:
                if old is None:
                    os.environ.pop("AFNAN_CHROMIUM_EXECUTABLE", None)
                else:
                    os.environ["AFNAN_CHROMIUM_EXECUTABLE"] = old
            self.assertEqual(found, handle.name)

    def test_missing_executable_is_structured(self):
        adapter = ChromiumAdapter()
        original = ChromiumAdapter.find_executable
        ChromiumAdapter.find_executable = staticmethod(
            lambda browser="chromium": None
        )
        try:
            with self.assertRaises(BrowserException) as caught:
                adapter.start("chromium", True)
        finally:
            ChromiumAdapter.find_executable = original
        self.assertEqual(
            caught.exception.error.code,
            BrowserErrorCode.BROWSER_UNAVAILABLE,
        )


# ----------------------------------------------------------------------
# Runtime-level additions (health, tab tasks, permissions, events)
# ----------------------------------------------------------------------


class TestRuntimeHealthAndPermissions(unittest.TestCase):
    def setUp(self):
        self.adapter = FakeBrowserAdapter()
        self.adapter.add_page(
            "https://a.example/", title="A", text="page a",
            elements=[],
        )
        self.runtime = AfnanBrowserRuntime(adapter=self.adapter)
        self.runtime.start()
        self.handle = self.runtime.new_page()
        self.runtime.goto(self.handle, "https://a.example/")

    def tearDown(self):
        self.runtime.stop()

    def test_health_reports_alive_session_and_active_tab(self):
        health = self.runtime.health()
        self.assertTrue(health["alive"])
        self.assertEqual(health["session_status"], "running")
        self.assertEqual(health["adapter"], "fake")
        self.assertIsNotNone(health["active_tab"])
        self.assertTrue(health["page_responsive"])
        self.assertEqual(health["tab_count"], 1)

    def test_browser_ready_event_is_emitted(self):
        types = [e["type"] for e in self.runtime.events()]
        self.assertIn("browser_started", types)
        self.assertIn("browser_ready", types)

    def test_tab_task_association(self):
        tab_id = self.runtime.session.tabs[0].tab_id
        self.assertTrue(
            self.runtime.assign_tab_task(tab_id, "task-42")
        )
        record = self.runtime.tab_record(tab_id)
        self.assertEqual(record["task_id"], "task-42")
        self.assertEqual(record["window_id"], "main")
        self.assertFalse(self.runtime.assign_tab_task("rt_99", "x"))

    def test_crash_emits_browser_crashed_event(self):
        self.adapter.crash()
        self.assertFalse(self.runtime.check_health())
        types = [e["type"] for e in self.runtime.events()]
        self.assertIn("browser_crashed", types)
        self.assertEqual(self.runtime.session.status, "crashed")
        self.assertFalse(self.runtime.health()["alive"])

    def test_permission_levels_fail_safe_by_default(self):
        runtime = self.runtime
        self.assertTrue(runtime.check_permission("navigate"))
        self.assertTrue(
            runtime.check_permission("click", "sensitive")
        )
        self.assertFalse(
            runtime.check_permission("pay", "approval_required")
        )
        runtime.set_permission_checker(
            lambda action, level, ctx: level != "destructive"
        )
        self.assertFalse(
            runtime.check_permission("delete", "destructive")
        )
        self.assertTrue(
            runtime.check_permission("pay", "approval_required")
        )
        with self.assertRaises(BrowserException):
            runtime.check_permission("x", "bogus-level")


# ----------------------------------------------------------------------
# Adapter comparison: same normalized controller results
# ----------------------------------------------------------------------


def _controller_observation(adapter) -> dict:
    controller = BrowserController(
        runtime=AfnanBrowserRuntime(adapter=adapter)
    )
    controller.launch(browser="chromium")
    tab_id = controller.new_tab("https://example.test/")["tab_id"]
    observation = controller.observe(tab_id)
    controller.shutdown()
    return {
        "url": observation["url"],
        "title": observation["title"],
        "text": observation["text"].strip(),
        "element_tags": [
            e["tag"] for e in observation["elements"]
        ],
    }


class TestAdapterComparison(unittest.TestCase):
    def test_fake_and_chromium_adapters_normalize_the_same(self):
        fake = FakeBrowserAdapter()
        fake.add_page(
            "https://example.test/", title="Example Title",
            text="Hello from page",
            elements=[el("button", text="Go")],
        )
        chromium_result = _controller_observation(
            fake_chromium_adapter()
        )
        fake_result = _controller_observation(fake)
        self.assertEqual(chromium_result, fake_result)

    @unittest.skipUnless(
        ChromiumAdapter.find_executable(),
        "no Chromium-family executable installed",
    )
    def test_playwright_and_chromium_agree_when_both_available(self):
        from afnan_ai.browser.backend import PlaywrightAdapter

        try:
            playwright_result = _controller_observation(
                PlaywrightAdapter()
            )
        except BrowserException:
            self.skipTest("Playwright engine not available")
        chromium_result = _controller_observation(ChromiumAdapter())
        self.assertEqual(chromium_result["title"], "Example Title")
        self.assertEqual(playwright_result["title"], "Example Title")
        self.assertEqual(
            chromium_result["url"], playwright_result["url"]
        )


# ----------------------------------------------------------------------
# Real Chromium integration (only when a binary exists)
# ----------------------------------------------------------------------

PAGE_ONE = (
    "data:text/html,<title>One</title>"
    "<h1>Hello One</h1><button>Go</button>"
)


@unittest.skipUnless(
    ChromiumAdapter.find_executable(),
    "no Chromium-family executable installed",
)
class TestChromiumIntegration(unittest.TestCase):
    def setUp(self):
        self.adapter = ChromiumAdapter()
        self.adapter.start("chromium", True)

    def tearDown(self):
        self.adapter.stop()

    def test_launch_navigate_tabs_screenshot_restart(self):
        adapter = self.adapter
        page = adapter.new_page()
        adapter.goto(page, PAGE_ONE)
        self.assertEqual(adapter.page_title(page), "One")
        self.assertIn("Hello One", adapter.page_text(page))
        self.assertTrue(
            adapter.screenshot(page).startswith(b"\x89PNG")
        )
        page2 = adapter.new_page()
        adapter.goto(
            page2, "data:text/html,<title>Two</title><p>second</p>"
        )
        self.assertEqual(len(adapter.list_pages()), 2)
        adapter.close_page(page2)
        self.assertEqual(len(adapter.list_pages()), 1)
        adapter.restart()
        self.assertTrue(adapter.is_alive())
        page3 = adapter.new_page()
        adapter.goto(page3, PAGE_ONE)
        self.assertEqual(adapter.page_title(page3), "One")

    def test_profile_storage_persists_across_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = ChromiumAdapter(user_data_dir=tmp)
            adapter.start("chromium", True)
            page = adapter.new_page()
            adapter.goto(page, PAGE_ONE)
            adapter._evaluate(
                page, "localStorage.setItem('afnan', 'kept')"
            )
            adapter.stop()
            adapter.start("chromium", True)
            page = adapter.new_page()
            adapter.goto(page, PAGE_ONE)
            kept = adapter._evaluate(
                page, "localStorage.getItem('afnan')"
            )
            self.assertEqual(kept, "kept")
            adapter.stop()

    def test_runtime_crash_recovery(self):
        runtime = AfnanBrowserRuntime(adapter=ChromiumAdapter())
        runtime.start()
        handle = runtime.new_page()
        runtime.goto(handle, PAGE_ONE)
        process = runtime.adapter._profiles["default"].process
        process.kill()
        self.assertFalse(runtime.check_health())
        self.assertEqual(runtime.session.status, "crashed")
        result = runtime.recover()
        self.assertTrue(result.success)
        self.assertEqual(
            len(result.data["recoverable_tabs"]), 1
        )
        self.assertTrue(runtime.check_health())
        runtime.stop()


# ----------------------------------------------------------------------
# Engine isolation (requirement: no engine leakage)
# ----------------------------------------------------------------------


class TestEngineIsolation(unittest.TestCase):
    def test_playwright_imports_live_only_in_its_adapter(self):
        root = (
            Path(__file__).resolve().parents[1] / "afnan_ai"
        )
        offenders = []
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if "import playwright" in source or (
                "from playwright" in source
            ):
                if path.name != "backend.py":
                    offenders.append(str(path.relative_to(root)))
        self.assertEqual(offenders, [])

    def test_core_agent_has_no_engine_references(self):
        root = (
            Path(__file__).resolve().parents[1] / "afnan_ai"
        )
        for name in ("agent.py", "planner.py", "executor.py",
                     "verifier.py", "orchestrator.py"):
            source = (root / name).read_text(encoding="utf-8")
            self.assertNotIn("ChromiumAdapter", source, name)
            self.assertNotIn("playwright", source.lower(), name)


if __name__ == "__main__":
    unittest.main()

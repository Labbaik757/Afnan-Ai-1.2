"""Afnan Browser Runtime tests.

The runtime (lifecycle, sessions, persistent profiles, events,
capabilities, crash recovery) is tested against a purpose-built
FakeBrowserAdapter — no real browser is launched.  Controller
and agent integration run on the same fake, proving the stack
above the adapter never touches engine types.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from afnan_ai.browser.base import BrowserErrorCode, BrowserException
from afnan_ai.browser.controller import BrowserController
from afnan_ai.browser.engine import BrowserEngineAdapter
from afnan_ai.browser.models import BrowserSession, BrowserTab
from afnan_ai.browser.runtime import AfnanBrowserRuntime

from test_browser_advanced import QueueLLM, el, plan_json, step


class FakeBrowserAdapter(BrowserEngineAdapter):
    """In-memory engine adapter (the AfnanChromiumAdapter stand-in)."""

    name = "fake"

    def __init__(self):
        self.pages: dict[str, dict] = {}
        self.handles: dict[str, str] = {}
        self._counter = 0
        self.started = False
        self.crashed = False
        self.contexts: dict[str, dict] = {"default": {}}
        self.context_options: dict[str, dict] = {}

    # -- test helpers ---------------------------------------------------
    def add_page(self, url, title="", text="", elements=None):
        self.pages[url] = {
            "title": title, "text": text,
            "elements": list(elements or []),
        }

    def crash(self):
        self.crashed = True

    # -- engine interface -------------------------------------------------
    def _require(self):
        if self.crashed:
            raise RuntimeError("browser process died")
        if not self.started:
            raise BrowserException(
                "Browser is not running",
                code=BrowserErrorCode.BROWSER_NOT_STARTED,
            )

    def start(self, browser, headless):
        self.started = True

    def connect(self, endpoint):
        self.started = True

    def stop(self):
        self.started = False
        self.handles.clear()

    def is_alive(self):
        return self.started and not self.crashed

    def capabilities(self):
        return {
            "tabs": True, "accessibility": False,
            "screenshots": True, "downloads": False,
            "uploads": True, "persistent_profiles": True,
            "network_observation": False,
        }

    def new_page(self):
        self._require()
        self._counter += 1
        handle = f"page_{self._counter}"
        self.handles[handle] = "about:blank"
        return handle

    def close_page(self, handle):
        self.handles.pop(handle, None)

    def list_pages(self):
        return list(self.handles)

    def goto(self, handle, url):
        self._require()
        if url not in self.pages:
            raise BrowserException(
                f"Navigation failed: no page at {url}",
                code=BrowserErrorCode.NAVIGATION_FAILED,
            )
        self.handles[handle] = url

    def go_back(self, handle):
        self._require()

    def go_forward(self, handle):
        self._require()

    def reload(self, handle):
        self._require()

    def page_url(self, handle):
        return self.handles.get(handle, "")

    def page_title(self, handle):
        page = self.pages.get(self.handles.get(handle, ""), {})
        return page.get("title", "")

    def page_text(self, handle):
        page = self.pages.get(self.handles.get(handle, ""), {})
        return page.get("text", "")

    def interactive_elements(self, handle, limit):
        page = self.pages.get(self.handles.get(handle, ""), {})
        return [
            (self.handles[handle], i)
            for i, _ in enumerate(page.get("elements", []))
        ][:limit]

    def element_info(self, handle, element):
        page = self.pages[self.handles[handle]]
        return dict(page["elements"][element[1]])

    def query_elements(self, handle, locator, limit):
        page = self.pages.get(self.handles.get(handle, ""), {})
        found = []
        for i, spec in enumerate(page.get("elements", [])):
            if "selector" in locator:
                sel = locator["selector"]
                classes = (
                    spec.get("attributes", {}).get("class", "")
                ).split()
                if sel.startswith(".") and sel[1:] in classes:
                    found.append((self.handles[handle], i))
                elif sel == spec.get("tag"):
                    found.append((self.handles[handle], i))
            elif locator.get("text") and (
                locator["text"] in spec.get("text", "")
            ):
                found.append((self.handles[handle], i))
        return found[:limit]

    def click_element(self, handle, element, timeout_ms):
        self._require()

    def fill_element(self, handle, element, text, timeout_ms):
        page = self.pages[self.handles[handle]]
        page["elements"][element[1]]["value"] = text

    def screenshot(self, handle):
        self._require()
        return b"\x89PNG\r\n\x1a\nfake"

    def create_profile_context(self, name, options):
        self.context_options[name] = dict(options or {})
        self.contexts.setdefault(name, {})
        return name

    def set_active_profile(self, name):
        if name not in self.contexts:
            raise BrowserException(
                f"Unknown profile {name!r}",
                code=BrowserErrorCode.PROFILE_ERROR,
            )


class TestRuntimeLifecycle(unittest.TestCase):
    def make_runtime(self, **kw):
        adapter = FakeBrowserAdapter()
        return AfnanBrowserRuntime(adapter=adapter, **kw), adapter

    def test_start_stop_events_and_session(self):
        runtime, _ = self.make_runtime()
        seen = []
        runtime.subscribe(seen.append)
        result = runtime.start()
        self.assertTrue(result.success)
        self.assertEqual(runtime.session.status, "running")
        runtime.stop()
        self.assertEqual(runtime.session.status, "stopped")
        types = [e["type"] for e in seen]
        self.assertIn("browser_started", types)
        self.assertIn("browser_stopped", types)

    def test_tabs_tracked_in_session(self):
        runtime, adapter = self.make_runtime()
        adapter.add_page("https://a.example/", title="A", text="page a")
        runtime.start()
        handle = runtime.new_page()
        runtime.goto(handle, "https://a.example/")
        records = runtime.tab_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["url"], "https://a.example/")
        self.assertEqual(records[0]["title"], "A")
        types = [e["type"] for e in runtime.events()]
        self.assertIn("tab_created", types)
        self.assertIn("navigation_completed", types)
        runtime.close_page(handle)
        self.assertEqual(runtime.tab_records(), [])
        self.assertIn(
            "tab_closed", [e["type"] for e in runtime.events()]
        )

    def test_popup_detected_event(self):
        runtime, adapter = self.make_runtime()
        runtime.start()
        runtime.new_page()
        adapter.handles["popup_1"] = "about:blank"  # engine-side popup
        pages = runtime.list_pages()
        self.assertEqual(len(pages), 2)
        self.assertIn(
            "popup_detected", [e["type"] for e in runtime.events()]
        )

    def test_start_failure_codes(self):
        class Unavailable(FakeBrowserAdapter):
            def start(self, browser, headless):
                raise BrowserException(
                    "no browser", code=BrowserErrorCode.BROWSER_UNAVAILABLE
                )

        runtime = AfnanBrowserRuntime(adapter=Unavailable())
        with self.assertRaises(BrowserException) as ctx:
            runtime.start()
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.BROWSER_UNAVAILABLE
        )

        class Broken(FakeBrowserAdapter):
            def start(self, browser, headless):
                raise BrowserException(
                    "kaboom", code=BrowserErrorCode.OPERATION_FAILED
                )

        runtime = AfnanBrowserRuntime(adapter=Broken())
        with self.assertRaises(BrowserException) as ctx:
            runtime.start()
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.STARTUP_FAILED
        )

    def test_capabilities_merge(self):
        runtime, _ = self.make_runtime()
        caps = runtime.capabilities()
        self.assertTrue(caps["tabs"])
        self.assertTrue(caps["events"])
        self.assertTrue(caps["screenshots"])   # from the adapter
        self.assertFalse(caps["downloads"])    # adapter says no


class TestRuntimeProfilesAndSessions(unittest.TestCase):
    def test_profiles_persist_without_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = AfnanBrowserRuntime(
                adapter=FakeBrowserAdapter(), runtime_dir=tmp
            )
            profile = runtime.create_profile(
                "work",
                {"user_agent": "UA/1.0", "api_token": "sekrit"},
            )
            self.assertNotIn("api_token", profile.preferences)
            self.assertTrue(Path(profile.storage_dir).is_dir())
            saved = Path(tmp, "profiles.json").read_text()
            self.assertNotIn("sekrit", saved)

            runtime2 = AfnanBrowserRuntime(
                adapter=FakeBrowserAdapter(), runtime_dir=tmp
            )
            names = [p["name"] for p in runtime2.list_profiles()]
            self.assertEqual(names, ["work"])
            runtime2.delete_profile("work")
            self.assertEqual(runtime2.list_profiles(), [])

    def test_open_profile_uses_persistent_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = FakeBrowserAdapter()
            runtime = AfnanBrowserRuntime(
                adapter=adapter, runtime_dir=tmp
            )
            runtime.create_profile("work")
            runtime.start()
            runtime.open_profile("work")
            self.assertEqual(runtime.session.profile_id, "work")
            options = adapter.context_options["work"]
            self.assertIn("user_data_dir", options)
            self.assertTrue(options["user_data_dir"].endswith("work"))

    def test_session_persistence_and_redaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = AfnanBrowserRuntime(
                adapter=FakeBrowserAdapter(), runtime_dir=tmp
            )
            runtime.adapter.add_page(
                "https://a.example/?session_id=sekrit",
                title="A", text="x",
            )
            runtime.start()
            handle = runtime.new_page()
            runtime.goto(
                handle, "https://a.example/?session_id=sekrit"
            )
            path = runtime.save_session()
            self.assertIsNotNone(path)
            raw = Path(path).read_text()
            self.assertNotIn("sekrit", raw)

            runtime2 = AfnanBrowserRuntime(
                adapter=FakeBrowserAdapter(), runtime_dir=tmp
            )
            loaded = runtime2.load_session()
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded.tabs), 1)
            self.assertEqual(loaded.tabs[0].title, "A")


class TestCrashRecovery(unittest.TestCase):
    def test_crash_detected_and_recovered(self):
        adapter = FakeBrowserAdapter()
        adapter.add_page("https://a.example/", title="A", text="page a")
        runtime = AfnanBrowserRuntime(adapter=adapter)
        runtime.start()
        handle = runtime.new_page()
        runtime.goto(handle, "https://a.example/")

        adapter.crash()
        self.assertFalse(runtime.check_health())
        self.assertEqual(runtime.session.status, "crashed")
        with self.assertRaises(BrowserException) as ctx:
            runtime.goto(handle, "https://a.example/")
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.BROWSER_CRASHED
        )

        result = runtime.recover()
        self.assertTrue(result.success)
        self.assertEqual(runtime.session.status, "running")
        recoverable = result.data["recoverable_tabs"]
        self.assertEqual(len(recoverable), 1)
        self.assertEqual(
            recoverable[0]["url"], "https://a.example/"
        )
        types = [e["type"] for e in runtime.events()]
        self.assertIn("browser_recovered", types)


class TestModels(unittest.TestCase):
    def test_session_roundtrip(self):
        session = BrowserSession(
            session_id="s1", profile_id="work",
            tabs=[BrowserTab(
                tab_id="rt_1", url="https://a.example/?token=abc",
                title="A",
            )],
        )
        data = session.to_dict()
        self.assertNotIn("token=abc", json.dumps(data))
        restored = BrowserSession.from_dict(data)
        self.assertEqual(restored.session_id, "s1")
        self.assertEqual(restored.tabs[0].title, "A")


class TestControllerOnRuntime(unittest.TestCase):
    def make_controller(self):
        adapter = FakeBrowserAdapter()
        adapter.add_page(
            "https://a.example/", title="A", text="hello page a",
            elements=[el("button", "Go")],
        )
        runtime = AfnanBrowserRuntime(adapter=adapter)
        controller = BrowserController(runtime=runtime)
        return controller, runtime, adapter

    def test_controller_full_flow_and_events(self):
        controller, runtime, _ = self.make_controller()
        controller.launch()
        controller.new_tab("https://a.example/")
        observed = controller.observe()
        self.assertEqual(observed["title"], "A")
        types = [e["type"] for e in controller.events()]
        self.assertIn("browser_started", types)
        self.assertIn("tab_created", types)
        caps = controller.capabilities()
        self.assertEqual(caps["adapter"], "fake")
        self.assertTrue(caps["capabilities"]["semantic_locator"])
        controller.shutdown()

    def test_agent_state_has_no_engine_objects(self):
        controller, runtime, _ = self.make_controller()
        from afnan_ai.agent import AfnanAgent

        agent = AfnanAgent(
            llm_provider=QueueLLM([
                plan_json("Open page A", [
                    step("s1", "browser_navigate",
                         {"url": "https://a.example/"},
                         "hello page a"),
                ]),
            ]),
            browser_controller=controller,
        )
        controller.launch()
        outcome = agent.run_browser_goal("Open page A")
        self.assertEqual(outcome.status, "completed")
        dump = json.dumps(outcome.state.to_dict())
        self.assertNotIn("page_1", dump)  # engine handles never leak
        caps = agent.tools.execute("browser_capabilities", {})
        self.assertTrue(caps.success)
        self.assertEqual(caps.output["adapter"], "fake")


class TestDefaultPersistence(unittest.TestCase):
    """Profiles persist by default through the agent's runtime."""

    def test_config_env_override(self):
        import os as _os

        from afnan_ai.config import AgentConfig

        old = _os.environ.get("AFNAN_BROWSER_RUNTIME_DIR")
        _os.environ["AFNAN_BROWSER_RUNTIME_DIR"] = "/tmp/afnan-rt"
        try:
            config = AgentConfig.from_env()
        finally:
            if old is None:
                _os.environ.pop("AFNAN_BROWSER_RUNTIME_DIR", None)
            else:
                _os.environ["AFNAN_BROWSER_RUNTIME_DIR"] = old
        self.assertEqual(config.browser_runtime_dir, "/tmp/afnan-rt")

    def test_agent_runtime_dir_and_profile_persistence(self):
        from afnan_ai.agent import AfnanAgent

        with tempfile.TemporaryDirectory() as tmp:
            agent = AfnanAgent(
                llm_provider=QueueLLM([]), browser_runtime_dir=tmp
            )
            runtime = agent.get_browser_runtime()
            self.assertEqual(runtime.runtime_dir, Path(tmp))
            runtime.create_profile("work")
            self.assertTrue(Path(tmp, "profiles.json").is_file())

            agent2 = AfnanAgent(
                llm_provider=QueueLLM([]), browser_runtime_dir=tmp
            )
            names = [
                p["name"] for p in agent2.browser.list_profiles()
            ]
            self.assertIn("work", names)

    def test_controller_profile_uses_persistent_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = FakeBrowserAdapter()
            adapter.add_page(
                "https://a.example/", title="A", text="page a",
                elements=[],
            )
            runtime = AfnanBrowserRuntime(
                adapter=adapter, runtime_dir=tmp
            )
            controller = BrowserController(runtime=runtime)
            controller.launch()
            controller.new_tab("https://a.example/")
            controller.create_profile("work")
            profile = runtime.get_profile("work")
            self.assertTrue(profile.storage_dir.endswith("work"))
            self.assertTrue(Path(profile.storage_dir).is_dir())
            controller.select_profile("work")
            options = adapter.context_options["work"]
            self.assertTrue(
                options["user_data_dir"].endswith("work")
            )
            controller.shutdown()

            # A fresh controller on the same runtime dir knows
            # the profile without re-creating it.
            runtime2 = AfnanBrowserRuntime(
                adapter=FakeBrowserAdapter(), runtime_dir=tmp
            )
            controller2 = BrowserController(runtime=runtime2)
            names = [
                p["name"] for p in controller2.list_profiles()
            ]
            self.assertIn("work", names)


if __name__ == "__main__":
    unittest.main()

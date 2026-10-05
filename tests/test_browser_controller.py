import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai.browser import (
    BrowserController,
    BrowserErrorCode,
    BrowserException,
    PlaywrightBackend,
)
from afnan_ai.browser.backend import BrowserBackend


class FakePage:
    """In-memory page with real history semantics."""

    def __init__(self):
        self.history = ["about:blank"]
        self.index = 0

    @property
    def url(self):
        return self.history[self.index]

    def goto(self, url):
        self.history = self.history[: self.index + 1] + [url]
        self.index += 1

    def title(self):
        if "example" in self.url:
            return "Example Domain"
        return f"Title of {self.url}"


class FakeBrowserBackend(BrowserBackend):
    """Deterministic in-memory backend (no real browser)."""

    name = "fake"

    def __init__(self, *, fail_start=False, fail_connect=False):
        self.fail_start = fail_start
        self.fail_connect = fail_connect
        self.started_with = None
        self.connected_to = None
        self.pages = []
        self.stopped = False

    def start(self, browser, headless):
        if self.fail_start:
            raise BrowserException(
                "fake browser binary missing",
                code=BrowserErrorCode.BROWSER_UNAVAILABLE,
            )
        self.started_with = (browser, headless)

    def connect(self, endpoint):
        if self.fail_connect:
            raise BrowserException(
                "connection refused",
                code=BrowserErrorCode.CONNECTION_FAILED,
            )
        self.connected_to = endpoint

    def new_page(self):
        page = FakePage()
        self.pages.append(page)
        return page

    def close_page(self, handle):
        self.pages.remove(handle)

    def goto(self, handle, url):
        if "fail.example" in url:
            raise RuntimeError("net::ERR_NAME_NOT_RESOLVED")
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
        return handle.title()

    def stop(self):
        self.stopped = True
        self.pages.clear()


def make_controller(**backend_kwargs):
    backend = FakeBrowserBackend(**backend_kwargs)
    return BrowserController(backend=backend), backend


class TestLifecycle(unittest.TestCase):
    def test_launch_reports_session(self):
        controller, backend = make_controller()
        info = controller.launch()
        self.assertTrue(info["running"])
        self.assertTrue(info["launched"])
        self.assertEqual(info["browser"], "chromium")
        self.assertEqual(backend.started_with, ("chromium", False))
        self.assertTrue(controller.is_running)

    def test_launch_is_idempotent(self):
        controller, _ = make_controller()
        controller.launch()
        info = controller.launch()
        self.assertTrue(info["running"])
        self.assertFalse(info["launched"])

    def test_unsupported_browser_is_structured_error(self):
        controller, _ = make_controller()
        with self.assertRaises(BrowserException) as ctx:
            controller.launch(browser="netscape")
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.BROWSER_UNAVAILABLE
        )
        self.assertFalse(controller.is_running)

    def test_backend_unavailable_is_structured_error(self):
        controller, _ = make_controller(fail_start=True)
        with self.assertRaises(BrowserException) as ctx:
            controller.launch()
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.BROWSER_UNAVAILABLE
        )

    def test_playwright_missing_is_structured_error(self):
        controller = BrowserController(backend=PlaywrightBackend())
        with mock.patch.dict(sys.modules, {"playwright": None}):
            with self.assertRaises(BrowserException) as ctx:
                controller.launch()
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.BROWSER_UNAVAILABLE
        )

    def test_connect_and_empty_endpoint(self):
        controller, backend = make_controller()
        info = controller.connect("http://localhost:9222")
        self.assertTrue(info["running"])
        self.assertEqual(backend.connected_to, "http://localhost:9222")

        controller2, _ = make_controller()
        with self.assertRaises(BrowserException) as ctx:
            controller2.connect("  ")
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.CONNECTION_FAILED
        )

    def test_connect_failure_is_structured_error(self):
        controller, _ = make_controller(fail_connect=True)
        with self.assertRaises(BrowserException) as ctx:
            controller.connect("http://localhost:9222")
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.CONNECTION_FAILED
        )

    def test_shutdown_is_idempotent_and_final(self):
        controller, backend = make_controller()
        controller.launch()
        controller.new_tab()
        first = controller.shutdown()
        second = controller.shutdown()
        self.assertTrue(first["closed"])
        self.assertTrue(first["was_running"])
        self.assertFalse(second["was_running"])
        self.assertTrue(backend.stopped)
        self.assertFalse(controller.is_running)
        with self.assertRaises(BrowserException) as ctx:
            controller.new_tab()
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.BROWSER_NOT_STARTED
        )


class TestTabsAndNavigation(unittest.TestCase):
    def setUp(self):
        self.controller, self.backend = make_controller()
        self.controller.launch()

    def test_operations_before_launch_are_structured_errors(self):
        fresh, _ = make_controller()
        for op in (
            lambda: fresh.new_tab(),
            lambda: fresh.navigate("https://example.com"),
            lambda: fresh.current_page(),
            lambda: fresh.close_tab(),
        ):
            with self.assertRaises(BrowserException) as ctx:
                op()
            self.assertEqual(
                ctx.exception.code, BrowserErrorCode.BROWSER_NOT_STARTED
            )

    def test_new_tab_and_page_state(self):
        tab = self.controller.new_tab("https://example.com")
        self.assertEqual(tab["tab_id"], "tab_1")
        self.assertTrue(tab["active"])
        page = self.controller.current_page()
        self.assertEqual(page["url"], "https://example.com")
        self.assertEqual(page["title"], "Example Domain")
        self.assertEqual(self.controller.title(), "Example Domain")
        self.assertEqual(self.controller.current_url(), "https://example.com")

    def test_navigate_adds_scheme_and_creates_tab(self):
        page = self.controller.navigate("example.com")
        self.assertEqual(page["url"], "https://example.com")
        self.assertEqual(len(self.controller.list_tabs()), 1)

    def test_tabs_select_and_close(self):
        first = self.controller.new_tab("https://example.com")
        second = self.controller.new_tab("https://python.org")
        self.assertEqual(self.controller.active_tab_id, second["tab_id"])

        selected = self.controller.select_tab(first["tab_id"])
        self.assertTrue(selected["active"])
        self.assertEqual(
            self.controller.current_page()["url"], "https://example.com"
        )

        closed = self.controller.close_tab(first["tab_id"])
        self.assertEqual(closed["closed_tab"], first["tab_id"])
        self.assertEqual(closed["active_tab"], second["tab_id"])
        tabs = self.controller.list_tabs()
        self.assertEqual([t["tab_id"] for t in tabs], [second["tab_id"]])

    def test_unknown_tab_is_structured_error(self):
        self.controller.new_tab()
        for op in (
            lambda: self.controller.select_tab("tab_99"),
            lambda: self.controller.close_tab("tab_99"),
            lambda: self.controller.current_page("tab_99"),
            lambda: self.controller.navigate("https://x.com", tab_id="tab_99"),
        ):
            with self.assertRaises(BrowserException) as ctx:
                op()
            self.assertEqual(
                ctx.exception.code, BrowserErrorCode.INVALID_TAB
            )

    def test_no_tab_open_is_structured_error(self):
        with self.assertRaises(BrowserException) as ctx:
            self.controller.current_page()
        self.assertEqual(ctx.exception.code, BrowserErrorCode.INVALID_TAB)

    def test_back_forward_reload(self):
        self.controller.new_tab("https://example.com")
        self.controller.navigate("https://python.org")
        page = self.controller.back()
        self.assertEqual(page["url"], "https://example.com")
        page = self.controller.forward()
        self.assertEqual(page["url"], "https://python.org")
        page = self.controller.reload()
        self.assertEqual(page["url"], "https://python.org")

    def test_empty_url_is_structured_error(self):
        self.controller.new_tab()
        with self.assertRaises(BrowserException) as ctx:
            self.controller.navigate("   ")
        self.assertEqual(ctx.exception.code, BrowserErrorCode.INVALID_URL)

    def test_navigation_failure_is_structured_error(self):
        self.controller.new_tab()
        with self.assertRaises(BrowserException) as ctx:
            self.controller.navigate("https://fail.example")
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.NAVIGATION_FAILED
        )
        self.assertIn("ERR_NAME_NOT_RESOLVED", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

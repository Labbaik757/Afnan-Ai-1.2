"""Comprehensive Browser Agent stress + acceptance tests.

Long multi-step tasks, dynamic pages, failed clicks, stale
elements, unexpected popups, navigation/session changes,
timeouts, failed downloads/uploads, auth failure + recovery,
iframe/shadow-DOM interaction, sensitive-action approval, secret
hygiene and crash safety — all against an in-memory fake site.
The final acceptance class maps one-to-one onto the required
guarantees:

1. normal browser tasks complete;
2. failed tasks recover or terminate safely;
3. sensitive actions never execute without approval;
4. secrets are never exposed in records;
5. browser failures never crash the Agent;
6. Windows/Linux/macOS architecture compatibility holds.
"""
import io
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai.agent import AfnanAgent
from afnan_ai.browser import BrowserController
from afnan_ai.browser.backend import BrowserBackend
from afnan_ai.browser.base import BrowserErrorCode, BrowserException
from afnan_ai.llm.base import LLMProvider
from afnan_ai.orchestrator import OrchestrationStatus
from afnan_ai.platform.factory import get_adapter

LOGIN = "https://shop.example/login"
HOME = "https://shop.example/home"
ACCOUNT = "https://shop.example/account"
CORRECT_PASSWORD = "correct-horse"


# ----------------------------------------------------------------------
# Fake site
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
        self.uploaded = None

    @property
    def editable(self):
        return self.tag in ("input", "textarea", "select")


class StressPage:
    def __init__(self):
        self.history = ["about:blank"]
        self.index = 0
        self.elements = []
        self.shadow = []
        self.frames = {}
        self.text = ""
        self.page_title = ""
        self.pending = []

    @property
    def url(self):
        return self.history[self.index]

    def load(self, url):
        for el in self.elements + self.shadow:
            el.attached = False
        self.pending = []
        self.frames = {}
        self.shadow = []
        if "login" in url:
            self.page_title = "Login"
            self.text = "Welcome, please sign in"
            self.elements = [
                FakeElement("input", {"id": "username", "type": "text",
                                      "placeholder": "Username"}),
                FakeElement("input", {"id": "password", "type": "password"}),
                FakeElement("button", {"id": "signin"}, text="Sign in"),
                FakeElement("a", {"id": "help",
                                  "data-popup-url": "https://shop.example/help"},
                            text="Help"),
                FakeElement("a", {"id": "download"}, text="Download brochure"),
                FakeElement("input", {"id": "avatar", "type": "file"}),
            ]
        elif "home" in url:
            self.page_title = "Dashboard"
            self.text = "Your dashboard"
            self.elements = [
                FakeElement("button", {"id": "order"}, text="Place Order"),
                FakeElement("button", {"id": "delete"}, text="Delete account"),
                FakeElement("button", {"id": "send"}, text="Send message"),
                FakeElement("button", {"id": "explode"}, text="Explode"),
                FakeElement("button", {"id": "logout"}, text="Logout"),
            ]
            self.frames = {
                "#payment-frame": [
                    FakeElement("input", {"id": "note", "type": "text",
                                          "placeholder": "Note"}),
                    FakeElement("input", {"id": "cardnum", "type": "text",
                                          "placeholder": "Card number"}),
                ]
            }
            self.shadow = [
                FakeElement("button", {"id": "shadow-btn"},
                            text="Shadow Action")
            ]
            self.pending.append(
                (FakeElement("button", {"id": "continue"}, text="Continue"), 2)
            )
        elif "account" in url:
            self.page_title = "Account"
            self.text = "Your account settings"
            self.elements = [
                FakeElement("input", {"id": "nickname", "type": "text"})
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


class FakeStressBackend(BrowserBackend):
    name = "fake-stress"

    def __init__(self):
        self.pages = []
        self.started = False
        self.authed = False
        self.orders_placed = 0
        self.messages_sent = 0
        self.account_deleted = False
        self.shadow_clicked = False
        self.download_attempts = 0

    def start(self, browser, headless):
        self.started = True

    def connect(self, endpoint):
        self.started = True

    def stop(self):
        self.started = False
        self.pages.clear()

    def new_page(self):
        page = StressPage()
        self.pages.append(page)
        return page

    def close_page(self, handle):
        self.pages.remove(handle)

    def goto(self, handle, url):
        # protected page bounces anonymous users to login
        if "/account" in url and not self.authed:
            url = LOGIN
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
        if locator.get("frame"):
            pool = handle.frames.get(locator["frame"], [])
        else:
            pool = handle.elements + handle.shadow
        return [el for el in pool
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
        eid = element.attrs.get("id")
        if eid == "download":
            self.download_attempts += 1
            raise RuntimeError("net::ERR_ABORTED download failed")
        if eid == "explode":
            raise RuntimeError("driver exploded")
        element.clicked += 1
        popup = element.attrs.get("data-popup-url")
        if popup:
            self.new_page().goto(popup)
            return
        if eid == "signin":
            password = next(
                (el for el in handle.elements
                 if el.attrs.get("id") == "password"), None
            )
            if password is not None and password.value == CORRECT_PASSWORD:
                self.authed = True
                handle.goto(HOME)
            else:
                handle.text = (
                    "Welcome, please sign in. Invalid credentials."
                )
        elif eid == "order":
            self.orders_placed += 1
            handle.text = "Order placed. Thank you!"
        elif eid == "delete":
            self.account_deleted = True
        elif eid == "send":
            self.messages_sent += 1
        elif eid == "shadow-btn":
            self.shadow_clicked = True

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

    def set_input_files(self, handle, element, path, timeout_ms):
        element.uploaded = path
        element.value = path

    # -- observation --------------------------------------------------
    def page_text(self, handle):
        return handle.text

    def interactive_elements(self, handle, limit):
        tags = ("a", "button", "input", "select", "textarea")
        return [el for el in handle.elements if el.tag in tags][:limit]

    def wait_for(self, handle, spec, timeout_ms):
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


# ----------------------------------------------------------------------
# Helpers
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
        "step_id": step_id,
        "description": f"{tool} step",
        "tool_name": tool,
        "arguments": arguments,
        "expected_result": expected,
    }


def plan_json(goal, steps):
    return json.dumps({"goal": goal, "steps": steps})


LAUNCH = step("s1", "browser_launch", {}, "Browser is running")
NAV_LOGIN = step("s2", "browser_navigate", {"url": LOGIN},
                 "Login page with username field")
TYPE_USER = step("s3", "browser_type",
                 {"selector": "#username", "text": "afnan"},
                 "Username input value afnan")
TYPE_PASS = step("s4", "browser_type",
                 {"selector": "#password", "text": CORRECT_PASSWORD},
                 "Password input value entered")
CLICK_SIGNIN = step("s5", "browser_click", {"selector": "#signin"},
                    "Dashboard home page is open")


def make_agent(replies, approver=None, **agent_kwargs):
    backend = FakeStressBackend()
    controller = BrowserController(backend=backend)
    agent = AfnanAgent(
        llm_provider=QueueLLM(replies),
        browser_controller=controller,
        browser_approver=approver,
        **agent_kwargs,
    )
    return agent, backend


def login_via_controller(controller):
    controller.launch()
    controller.new_tab(LOGIN)
    controller.type_text({"selector": "#username"}, "afnan")
    controller.type_text({"selector": "#password"}, CORRECT_PASSWORD)
    controller.click({"selector": "#signin"})


# ----------------------------------------------------------------------
# Stress scenarios
# ----------------------------------------------------------------------

class TestStressScenarios(unittest.TestCase):
    def test_long_multi_step_task_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_json("Full session", [
                LAUNCH, NAV_LOGIN,
                step("s3x", "browser_observe_page", {},
                     "Login page observed with elements"),
                TYPE_USER, TYPE_PASS, CLICK_SIGNIN,
                step("s6", "browser_wait_for",
                     {"condition": "text_present", "text": "Your dashboard"},
                     "Dashboard text visible on page"),
                step("s7", "browser_wait_for",
                     {"condition": "element_present",
                      "selector": "#continue"},
                     "Continue button visible on page"),
                step("s8", "browser_click", {"selector": "#continue"},
                     "Continue button on page"),
                step("s9", "browser_screenshot", {"output_dir": tmp},
                     "Screenshot dashboard"),
                step("s10", "browser_observe_page", {},
                     "Dashboard page observed with elements"),
            ])
            agent, backend = make_agent([plan], max_iterations=20)
            result = agent.orchestrator.run("Full session")
            self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
            self.assertEqual(result.recovery_attempts, [])
            self.assertGreaterEqual(len(result.state.tool_results), 10)
            # state stays fully serializable after the long run
            json.loads(result.state.to_json())

    def test_failed_click_recovers_via_relocation(self):
        v1 = plan_json("Sign in", [
            LAUNCH, NAV_LOGIN,
            step("s3", "browser_click", {"selector": "#login-button"},
                 "Dashboard home page is open"),
        ])
        fixed = plan_json("Sign in", [
            step("r1", "browser_type",
                 {"selector": "#username", "text": "afnan"},
                 "Username input value afnan"),
            step("r2", "browser_type",
                 {"selector": "#password", "text": CORRECT_PASSWORD},
                 "Password input value entered"),
            step("r3", "browser_click", {"text": "Sign in"},
                 "Dashboard home page is open"),
        ])
        agent, backend = make_agent([v1, fixed])
        result = agent.orchestrator.run("Sign in")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(len(result.recovery_attempts), 1)
        failed = [r for r in result.state.tool_results if not r.success]
        self.assertTrue(failed)  # failure honestly recorded

    def test_stale_ref_and_reload_mid_task(self):
        backend = FakeStressBackend()
        controller = BrowserController(backend=backend)
        controller.launch()
        controller.new_tab(LOGIN)
        ref = controller.find_elements({"selector": "#signin"})[0]["ref"]
        controller.reload()
        with self.assertRaises(BrowserException) as ctx:
            controller.click({"ref": ref})
        self.assertEqual(ctx.exception.code, BrowserErrorCode.STALE_ELEMENT)
        # locator-based actions survive the reload fine
        controller.click({"selector": "#signin"})

    def test_unexpected_popup_recovers(self):
        v1 = plan_json("Open help", [
            LAUNCH, NAV_LOGIN,
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
        self.assertEqual(agent.browser.current_page()["url"],
                         "https://shop.example/help")

    def test_session_redirect_then_login_then_account(self):
        plan = plan_json("Open account", [
            LAUNCH,
            step("s2", "browser_navigate", {"url": ACCOUNT},
                 "Login page with username field"),
            TYPE_USER, TYPE_PASS, CLICK_SIGNIN,
            step("s6", "browser_navigate", {"url": ACCOUNT},
                 "Account page with settings"),
        ])
        agent, backend = make_agent([plan])
        result = agent.orchestrator.run("Open account")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(agent.browser.current_page()["url"], ACCOUNT)

    def test_timeout_terminates_safely(self):
        plan = plan_json("Impossible wait", [
            LAUNCH, NAV_LOGIN,
            step("s3", "browser_wait_for",
                 {"condition": "element_present", "selector": "#never",
                  "timeout_ms": 50},
                 "Impossible element appears"),
        ])
        agent, backend = make_agent([plan, "this is not json"])
        result = agent.orchestrator.run("Impossible wait")
        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        failed = [r for r in result.state.tool_results if not r.success]
        self.assertTrue(any(r.tool == "browser_wait_for" for r in failed))
        # browser still fully usable after the failure
        self.assertEqual(agent.browser.current_page()["url"], LOGIN)

    def test_failed_download_recovers(self):
        v1 = plan_json("Get the brochure", [
            LAUNCH, NAV_LOGIN,
            step("s3", "browser_click", {"selector": "#download"},
                 "Brochure file downloaded"),
        ])
        fixed = plan_json("Get the brochure", [
            step("r1", "browser_observe_page", {},
                 "Login page observed with elements"),
        ])
        agent, backend = make_agent([v1, fixed])
        result = agent.orchestrator.run("Get the brochure")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(backend.download_attempts, 1)  # no retry

    def test_failed_upload_missing_file(self):
        v1 = plan_json("Upload avatar", [
            LAUNCH, NAV_LOGIN,
            step("s3", "browser_upload_file",
                 {"selector": "#avatar", "path": "/nope/missing.png"},
                 "Avatar file uploaded"),
        ])
        fixed = plan_json("Upload avatar", [
            step("r1", "browser_observe_page", {},
                 "Login page observed with elements"),
        ])
        agent, backend = make_agent([v1, fixed])
        result = agent.orchestrator.run("Upload avatar")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        failed = [r for r in result.state.tool_results if not r.success]
        self.assertTrue(any(r.tool == "browser_upload_file" for r in failed))

    def test_auth_failure_recovers_with_correct_password(self):
        v1 = plan_json("Sign in", [
            LAUNCH, NAV_LOGIN, TYPE_USER,
            step("s4", "browser_type",
                 {"selector": "#password", "text": "wrong-pass"},
                 "Password input value entered"),
            CLICK_SIGNIN,
        ])
        fixed = plan_json("Sign in", [
            step("r1", "browser_clear", {"selector": "#password"},
                 "Password input value"),
            step("r2", "browser_type",
                 {"selector": "#password", "text": CORRECT_PASSWORD},
                 "Password input value entered"),
            step("r3", "browser_click", {"text": "Sign in"},
                 "Dashboard home page is open"),
        ])
        agent, backend = make_agent([v1, fixed])
        result = agent.orchestrator.run("Sign in")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(agent.browser.current_page()["url"], HOME)
        self.assertTrue(result.recovery_attempts)

    def test_iframe_and_shadow_dom(self):
        backend = FakeStressBackend()
        controller = BrowserController(backend=backend)
        login_via_controller(controller)
        # iframe content is reachable through the frame locator
        controller.type_text(
            {"frame": "#payment-frame", "selector": "#note"}, "hello"
        )
        note = backend.pages[0].frames["#payment-frame"][0]
        self.assertEqual(note.value, "hello")
        # payment field inside the iframe still needs approval
        with self.assertRaises(BrowserException) as ctx:
            controller.type_text(
                {"frame": "#payment-frame", "selector": "#cardnum"},
                "4111111111111111",
            )
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.APPROVAL_REQUIRED
        )
        # shadow-DOM content is found and clicked like normal DOM
        controller.click({"selector": "#shadow-btn"})
        self.assertTrue(backend.shadow_clicked)

    def test_sensitive_purchase_blocked_then_approved(self):
        plan = plan_json("Buy", [
            LAUNCH, NAV_LOGIN, TYPE_USER, TYPE_PASS, CLICK_SIGNIN,
            step("s6", "browser_click", {"selector": "#order"},
                 "Order placed on page"),
        ])
        fallback = plan_json("Buy", [
            step("r1", "browser_observe_page", {},
                 "Dashboard page observed with elements"),
        ])
        # no approver: the purchase must not happen
        agent, backend = make_agent([plan, fallback])
        result = agent.orchestrator.run("Buy")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(backend.orders_placed, 0)
        denied = [
            d for d in agent.browser.approval_gate.decisions
            if not d["allowed"]
        ]
        self.assertTrue(any(d["category"] == "purchase" for d in denied))

        # with a human yes, the same plan goes through
        agent2, backend2 = make_agent([plan], approver=lambda req: True)
        result2 = agent2.orchestrator.run("Buy")
        self.assertEqual(result2.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(backend2.orders_placed, 1)
        outputs = [
            r.output for r in result2.state.tool_results
            if r.tool == "browser_click" and r.success
        ]
        self.assertTrue(
            any((o or {}).get("sensitivity", {}).get("category") == "purchase"
                for o in outputs if isinstance(o, dict))
        )

    def test_backend_explosion_does_not_crash_agent(self):
        plan = plan_json("Boom", [
            LAUNCH, NAV_LOGIN, TYPE_USER, TYPE_PASS, CLICK_SIGNIN,
            step("s6", "browser_click", {"selector": "#explode"},
                 "Explosion happened"),
        ])
        agent, backend = make_agent([plan, "not json either"])
        result = agent.orchestrator.run("Boom")  # must not raise
        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        # controller still works afterwards
        page = agent.browser.current_page()
        self.assertEqual(page["url"], HOME)


# ----------------------------------------------------------------------
# Final acceptance — the six required guarantees
# ----------------------------------------------------------------------

class TestFinalAcceptance(unittest.TestCase):
    def test_1_normal_tasks_complete(self):
        plan = plan_json("Sign in", [LAUNCH, NAV_LOGIN, TYPE_USER,
                                     TYPE_PASS, CLICK_SIGNIN])
        agent, backend = make_agent([plan])
        result = agent.orchestrator.run("Sign in")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)

    def test_2_failed_tasks_recover_or_terminate(self):
        plan = plan_json("Wait forever", [
            LAUNCH, NAV_LOGIN,
            step("s3", "browser_wait_for",
                 {"condition": "element_present", "selector": "#ghost",
                  "timeout_ms": 50},
                 "Ghost appears"),
        ])
        agent, backend = make_agent([plan, "garbage"])
        result = agent.orchestrator.run("Wait forever")
        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        self.assertTrue(
            any(not r.success for r in result.state.tool_results)
        )

    def test_3_sensitive_actions_need_approval(self):
        backend = FakeStressBackend()
        controller = BrowserController(backend=backend)
        login_via_controller(controller)
        with self.assertRaises(BrowserException) as ctx:
            controller.click({"selector": "#order"})
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.APPROVAL_REQUIRED
        )
        self.assertEqual(backend.orders_placed, 0)
        with self.assertRaises(BrowserException):
            controller.click({"selector": "#delete"})
        self.assertFalse(backend.account_deleted)
        with self.assertRaises(BrowserException):
            controller.click({"selector": "#send"})
        self.assertEqual(backend.messages_sent, 0)

    def test_4_secrets_never_exposed(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        old_level = root.level
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        try:
            plan = plan_json("Sign in", [LAUNCH, NAV_LOGIN, TYPE_USER,
                                         TYPE_PASS, CLICK_SIGNIN])
            agent, backend = make_agent([plan])
            result = agent.orchestrator.run("Sign in")
        finally:
            root.removeHandler(handler)
            root.setLevel(old_level)
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)

        state_blob = result.state.to_json()
        self.assertNotIn(CORRECT_PASSWORD, state_blob)
        prompt = agent.planner.build_prompt(
            "Sign in", state=result.state
        )
        self.assertNotIn(CORRECT_PASSWORD, prompt)
        self.assertNotIn(CORRECT_PASSWORD, stream.getvalue())
        decisions = json.dumps(agent.browser.approval_gate.decisions)
        self.assertNotIn(CORRECT_PASSWORD, decisions)
        obs = agent.browser.observe()
        self.assertNotIn(CORRECT_PASSWORD, json.dumps(obs))

    def test_5_browser_failures_do_not_crash_the_agent(self):
        backend = FakeStressBackend()
        controller = BrowserController(backend=backend)
        login_via_controller(controller)
        with self.assertRaises(BrowserException) as ctx:
            controller.click({"selector": "#explode"})
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.OPERATION_FAILED
        )
        # still alive and usable
        self.assertEqual(controller.current_page()["url"], HOME)
        controller.click({"selector": "#logout"})

    def test_6_cross_platform_architecture(self):
        # the correct adapter is selected per platform, and the
        # browser/security layers carry no OS or driver coupling
        self.assertEqual(
            type(get_adapter("Windows")).__name__, "WindowsAdapter"
        )
        self.assertEqual(
            type(get_adapter("Darwin")).__name__, "MacOSAdapter"
        )
        self.assertEqual(
            type(get_adapter("Linux")).__name__, "LinuxAdapter"
        )
        root = Path(__file__).resolve().parents[1]
        for rel in ("afnan_ai/browser/security.py", "afnan_ai/redaction.py"):
            source = (root / rel).read_text(encoding="utf-8")
            self.assertNotIn("playwright", source)
            self.assertNotIn("sys.platform", source)


if __name__ == "__main__":
    unittest.main()

"""Operations layer: network awareness, task checkpointing,
rate-limit/anti-bot awareness, approval records and browser
profiles.

All against fake backends — no real browser, network or model.
Security invariants checked throughout: secrets never land in
checkpoints, profiles never share session data, rate limiting
pauses instead of retrying aggressively, and approvals are
recorded with their outcome.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from afnan_ai.agent import AfnanAgent
from afnan_ai.browser.base import BrowserErrorCode, BrowserException
from afnan_ai.browser.controller import BrowserController
from afnan_ai.browser.ratelimit import (
    RateLimitDetector,
    RateLimitPolicy,
)
from afnan_ai.browser.reliability import BrowserReliability
from afnan_ai.browser.security import (
    ApprovalGate,
    SecurityPolicy,
    classify_action,
)
from afnan_ai.browser.tools import create_browser_tools
from afnan_ai.checkpointing import CheckpointError, CheckpointManager
from afnan_ai.tools.registry import ToolRegistry
from test_browser_advanced import (
    DynamicBackend,
    QueueLLM,
    el,
    make_controller,
    plan_json,
    step,
)


class OpsBackend(DynamicBackend):
    """DynamicBackend + network events, profile contexts, cookies."""

    def __init__(self):
        super().__init__()
        self.network_by_url: dict[str, list] = {}
        self.visited: list[str] = []
        self.contexts: dict[str, dict] = {"default": {"cookies": {}}}
        self.active_context = "default"

    def goto(self, handle, url):
        super().goto(handle, url)
        self.visited.append(url)

    def network_events(self, handle):
        page = self.handles[handle]
        return [
            dict(e)
            for e in self.network_by_url.get(page["url"], [])
        ]

    def create_profile_context(self, name, options):
        self.contexts.setdefault(name, {"cookies": {}})
        return name

    def set_active_profile(self, name):
        if name not in self.contexts:
            raise RuntimeError(f"unknown profile {name}")
        self.active_context = name

    def set_cookie(self, key, value):
        self.contexts[self.active_context]["cookies"][key] = value

    def get_cookie(self, key):
        return self.contexts[self.active_context]["cookies"].get(key)


def registry_for(controller):
    registry = ToolRegistry()
    for tool in create_browser_tools(controller):
        registry.register(tool)
    return registry


# ----------------------------------------------------------------------
# 1. Network awareness
# ----------------------------------------------------------------------


class TestNetworkAwareness(unittest.TestCase):
    def test_network_status_summarizes_failures(self):
        backend = OpsBackend()
        backend.add_page(
            "https://shop.example/", title="Shop", text="Shop page",
            elements=[],
        )
        backend.network_by_url["https://shop.example/"] = [
            {"url": "https://cdn.example/logo.png", "method": "GET",
             "status": 404, "failure": None,
             "resource_type": "image"},
            {"url": "https://api.example/data?token=sekrit42",
             "method": "GET", "status": None,
             "failure": "net::ERR_TIMED_OUT",
             "resource_type": "fetch"},
            {"url": "https://api.example/pay", "method": "POST",
             "status": 429, "failure": None,
             "resource_type": "fetch"},
            {"url": "https://shop.example/app.js", "method": "GET",
             "status": 200, "failure": None,
             "resource_type": "script"},
        ]
        controller, _ = make_controller(backend)
        controller.navigate("https://shop.example/")
        status = controller.network_status()
        self.assertFalse(status["healthy"])
        self.assertEqual(status["failed_requests"], 3)
        self.assertEqual(status["timeouts"], 1)
        self.assertEqual(status["rate_limited_responses"], 1)
        self.assertEqual(status["http_errors"], 2)
        self.assertNotIn("sekrit42", json.dumps(status))

    def test_network_tool_and_reliability_advice(self):
        backend = OpsBackend()
        backend.add_page(
            "https://shop.example/", title="Shop", text="Shop page",
            elements=[],
        )
        controller, _ = make_controller(backend)
        controller.navigate("https://shop.example/")
        result = registry_for(controller).execute(
            "browser_network_status", {}
        )
        self.assertTrue(result.success)
        self.assertTrue(result.output["healthy"])

        reliability = BrowserReliability(controller)
        advice = reliability.advise(
            SimpleNamespace(tool_name="browser_click"),
            SimpleNamespace(
                reason="Page resources failed to load: "
                       "net::ERR_BLOCKED_BY_CLIENT",
                evidence={"error": "failed to load"},
            ),
            {
                "url": "https://shop.example/", "title": "Shop",
                "network": {
                    "failed_requests": 1,
                    "recent_failures": [
                        {"url": "https://api.example/x",
                         "status": None,
                         "failure": "net::ERR_BLOCKED_BY_CLIENT"}
                    ],
                },
            },
        )
        self.assertEqual(advice["kind"], "network_failure")
        self.assertEqual(len(advice["network_failures"]), 1)


# ----------------------------------------------------------------------
# 2. Task checkpointing
# ----------------------------------------------------------------------

CP_A = "https://cp.example/a"
CP_B = "https://cp.example/b"
CP_C = "https://cp.example/c"


def _cp_backend(with_c: bool) -> OpsBackend:
    backend = OpsBackend()
    backend.add_page(CP_A, title="Alpha", text="Alpha page content",
                     elements=[])
    backend.add_page(CP_B, title="Beta", text="Beta page content",
                     elements=[])
    if with_c:
        backend.add_page(CP_C, title="Gamma",
                         text="Gamma page content", elements=[])
    return backend


CP_PLAN = plan_json("Visit three pages", [
    step("s1", "browser_navigate", {"url": CP_A}, "Alpha page"),
    step("s2", "browser_navigate", {"url": CP_B}, "Beta page"),
    step("s3", "browser_navigate", {"url": CP_C}, "Gamma page"),
])


class TestCheckpointing(unittest.TestCase):
    def test_save_load_and_latest(self):
        backend = _cp_backend(with_c=True)
        controller, _ = make_controller(backend)
        with tempfile.TemporaryDirectory() as tmp:
            agent = AfnanAgent(
                llm_provider=QueueLLM([CP_PLAN]),
                browser_controller=controller,
                checkpoint_dir=tmp,
            )
            result = agent.run_task("Visit three pages")
            self.assertEqual(result.status, "completed")
            manager = CheckpointManager(tmp)
            saved = manager.latest()
            self.assertIsNotNone(saved)
            self.assertEqual(saved["goal"], "Visit three pages")
            self.assertEqual(
                saved["extra"]["browser"]["tabs"][0]["url"], CP_C
            )
            reloaded = manager.load(saved["task_id"])
            self.assertEqual(reloaded["task_id"], saved["task_id"])

    def test_corrupt_checkpoint_rejected(self):
        backend = _cp_backend(with_c=True)
        controller, _ = make_controller(backend)
        with tempfile.TemporaryDirectory() as tmp:
            agent = AfnanAgent(
                llm_provider=QueueLLM([CP_PLAN]),
                browser_controller=controller,
                checkpoint_dir=tmp,
            )
            agent.run_task("Visit three pages")
            path = next(Path(tmp).glob("*.json"))
            data = json.loads(path.read_text())
            data["goal"] = "tampered goal"
            path.write_text(json.dumps(data))
            with self.assertRaises(CheckpointError):
                CheckpointManager(tmp).load(path)
            path.write_text("{not json")
            with self.assertRaises(CheckpointError):
                CheckpointManager(tmp).load(path)

    def test_resume_after_crash_skips_completed_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend1 = _cp_backend(with_c=False)
            controller1, _ = make_controller(backend1)
            agent1 = AfnanAgent(
                llm_provider=QueueLLM([CP_PLAN, CP_PLAN]),
                browser_controller=controller1,
                checkpoint_dir=tmp,
            )
            failed = agent1.run_task("Visit three pages")
            self.assertNotEqual(failed.status, "completed")

            path = next(Path(tmp).glob("*.json"))
            saved = CheckpointManager(tmp).load(path)
            completed_ids = {
                r["name"] for r in saved["state"]["completed_steps"]
            }
            self.assertEqual(completed_ids, {"s1", "s2"})

            backend2 = _cp_backend(with_c=True)
            controller2, _ = make_controller(backend2)
            agent2 = AfnanAgent(
                llm_provider=QueueLLM([]),
                browser_controller=controller2,
                checkpoint_dir=tmp,
            )
            resumed = agent2.resume_task(path)
            self.assertEqual(resumed.status, "completed")
            self.assertNotIn(CP_A, backend2.visited)
            self.assertEqual(backend2.visited.count(CP_C), 1)

    def test_secrets_never_written_to_checkpoint(self):
        backend = OpsBackend()
        backend.add_page(
            "https://vault.example/", title="Vault",
            text="Vault login page",
            elements=[
                el("input", "", {"id": "pw", "type": "password"}),
            ],
        )
        controller, _ = make_controller(backend)
        plan = plan_json("Sign in", [
            step("s1", "browser_navigate",
                 {"url": "https://vault.example/"}, "Vault login page"),
            step("s2", "browser_type",
                 {"selector": "#pw", "text": "typed-value",
                  "password": "hunter2-secret"},
                 "Vault login page"),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            agent = AfnanAgent(
                llm_provider=QueueLLM([plan, plan]),
                browser_controller=controller,
                checkpoint_dir=tmp,
            )
            agent.run_task("Sign in")
            raw = "".join(
                p.read_text() for p in Path(tmp).glob("*.json")
            )
            self.assertNotIn("hunter2-secret", raw)


# ----------------------------------------------------------------------
# 3. Rate-limit / anti-bot awareness
# ----------------------------------------------------------------------


class TestRateLimitAwareness(unittest.TestCase):
    def test_detector_reads_page_and_retry_after(self):
        detection = RateLimitDetector().detect({
            "url": "https://x.example/list", "title": "Slow down",
            "text": "Too many requests. Please slow down and "
                    "retry after 5 seconds.",
            "elements": [],
        })
        self.assertTrue(detection["detected"])
        self.assertEqual(detection["kind"], "rate_limited")
        self.assertEqual(detection["retry_after_s"], 5)

    def test_action_stops_on_rate_limit_page(self):
        backend = OpsBackend()
        backend.add_page(
            "https://x.example/list", title="Slow down",
            text="Too many requests. Please slow down and "
                 "retry after 5 seconds.",
            elements=[el("button", "Refresh", {"id": "r"})],
        )
        controller, _ = make_controller(backend)
        controller.navigate("https://x.example/list")
        found = controller.find_elements({"selector": "button"})
        with self.assertRaises(BrowserException) as ctx:
            controller.click({"ref": found[0]["ref"]})
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.RATE_LIMITED
        )

    def test_network_429_counts_as_rate_limited(self):
        backend = OpsBackend()
        backend.add_page(
            "https://x.example/feed", title="Feed", text="Feed",
            elements=[],
        )
        backend.network_by_url["https://x.example/feed"] = [
            {"url": "https://x.example/api", "method": "GET",
             "status": 429, "failure": None, "resource_type": "fetch"},
        ]
        controller, _ = make_controller(backend)
        controller.navigate("https://x.example/feed")
        status = controller.rate_limit_status()
        self.assertTrue(status["limited"])
        self.assertEqual(
            status["detection"]["kind"], "rate_limited"
        )

    def test_pacing_pauses_instead_of_hammering(self):
        backend = OpsBackend()
        backend.add_page(
            "https://x.example/home", title="Home", text="Home page",
            elements=[el("button", "OK", {"id": "ok"})],
        )
        controller, _ = make_controller(
            backend,
            rate_policy=RateLimitPolicy(max_actions_per_minute=2),
        )
        controller.navigate("https://x.example/home")
        found = controller.find_elements({"selector": "button"})
        ref = {"ref": found[0]["ref"]}
        controller.click(ref)
        controller.click(ref)
        with self.assertRaises(BrowserException) as ctx:
            controller.click(ref)
        self.assertEqual(
            ctx.exception.code, BrowserErrorCode.RATE_LIMITED
        )
        self.assertIn(
            "retry_after_s", ctx.exception.error.details
        )

    def test_backoff_waits_once_and_reports(self):
        backend = OpsBackend()
        backend.add_page(
            "https://x.example/list", title="Slow down",
            text="Too many requests. retry after 5 seconds.",
            elements=[],
        )
        controller, _ = make_controller(backend)
        controller.navigate("https://x.example/list")
        status = controller.rate_limit_backoff(max_wait_ms=40)
        self.assertTrue(status["limited"])
        self.assertEqual(status["waited_ms"], 40)


# ----------------------------------------------------------------------
# 4. Human-in-the-loop approvals
# ----------------------------------------------------------------------


def _purchase_risk():
    return classify_action(
        "browser_click",
        element={
            "tag": "button", "text": "Buy now",
            "attributes": {}, "visible": True, "enabled": True,
        },
        arguments={},
    )


class TestApprovalRecords(unittest.TestCase):
    def test_approval_timeout_denies_action(self):
        def slow_yes(request):
            time.sleep(0.2)
            return True

        gate = ApprovalGate(
            policy=SecurityPolicy(approval_timeout_s=0.05),
            approver=slow_yes,
        )
        decision = gate.check(
            _purchase_risk(), tool_name="browser_click"
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(gate.decisions[-1]["outcome"], "timeout")

    def test_decisions_recorded_with_outcomes(self):
        gate = ApprovalGate(approver=lambda request: False)
        decision = gate.check(
            _purchase_risk(), tool_name="browser_click"
        )
        self.assertFalse(decision.allowed)
        entry = gate.decisions[-1]
        self.assertEqual(entry["outcome"], "denied")
        self.assertEqual(entry["category"], "purchase")
        self.assertIn("decided_at", entry)

    def test_approvals_tool_and_state_recording(self):
        backend = OpsBackend()
        backend.add_page(
            "https://shop.example/", title="Shop",
            text="Shop page with Buy now button",
            elements=[el("button", "Buy now", {"id": "buy"})],
        )
        controller, _ = make_controller(backend)
        controller.launch()
        registry = registry_for(controller)
        controller.navigate("https://shop.example/")
        found = controller.find_elements({"selector": "button"})
        controller.approval_gate.approver = lambda request: True
        controller.click({"ref": found[0]["ref"]})
        result = registry.execute("browser_approvals", {})
        self.assertTrue(result.success)
        self.assertEqual(result.output["count"], 1)
        self.assertEqual(
            result.output["decisions"][0]["outcome"], "approved"
        )

    def test_approved_purchase_recorded_in_agent_state(self):
        backend = OpsBackend()
        backend.add_page(
            "https://shop.example/", title="Shop",
            text="Shop page with Buy now button",
            elements=[el("button", "Buy now", {"id": "buy"})],
        )
        controller, _ = make_controller(backend)
        plan = plan_json("Buy the thing", [
            step("s1", "browser_navigate",
                 {"url": "https://shop.example/"},
                 "Shop page Buy now button"),
            step("s2", "browser_click", {"selector": "button"},
                 "Shop page with Buy now button"),
        ])
        agent = AfnanAgent(
            llm_provider=QueueLLM([plan]),
            browser_controller=controller,
            browser_approver=lambda request: True,
        )
        result = agent.run_task("Buy the thing")
        self.assertEqual(result.status, "completed")
        self.assertIn(
            "purchase", json.dumps(result.state.to_dict())
        )


# ----------------------------------------------------------------------
# 5. Browser profiles
# ----------------------------------------------------------------------


class TestBrowserProfiles(unittest.TestCase):
    def _site(self, backend):
        backend.add_page(
            "https://a.example/", title="A", text="Page A",
            elements=[],
        )
        backend.add_page(
            "https://b.example/", title="B", text="Page B",
            elements=[],
        )

    def test_profiles_isolate_tabs_and_cookies(self):
        backend = OpsBackend()
        self._site(backend)
        controller, _ = make_controller(backend)
        controller.navigate("https://a.example/")
        backend.set_cookie("session", "default-cookie")

        controller.create_profile("work")
        switched = controller.select_profile("work")
        self.assertEqual(switched["name"], "work")
        self.assertEqual(controller.list_tabs(), [])
        self.assertIsNone(backend.get_cookie("session"))

        controller.new_tab("https://b.example/")
        backend.set_cookie("session", "work-cookie")
        self.assertEqual(backend.get_cookie("session"), "work-cookie")

        back = controller.select_profile("default")
        self.assertEqual(back["name"], "default")
        tabs = controller.list_tabs()
        self.assertEqual(len(tabs), 1)
        self.assertEqual(tabs[0]["url"], "https://a.example/")
        self.assertEqual(
            backend.get_cookie("session"), "default-cookie"
        )

    def test_preferences_sanitized_and_validation(self):
        backend = OpsBackend()
        controller, _ = make_controller(backend)
        created = controller.create_profile(
            "clean",
            {"user_agent": "UA-1", "password": "nope",
             "api_token": "nope", "locale": "en"},
        )
        self.assertEqual(
            created["preferences"],
            {"user_agent": "UA-1", "locale": "en"},
        )
        with self.assertRaises(BrowserException):
            controller.create_profile("clean")
        with self.assertRaises(BrowserException):
            controller.select_profile("missing")

    def test_profiles_tool_flow(self):
        backend = OpsBackend()
        self._site(backend)
        controller, _ = make_controller(backend)
        registry = registry_for(controller)
        created = registry.execute(
            "browser_profiles",
            {"action": "create", "name": "personal"},
        )
        self.assertTrue(created.success)
        selected = registry.execute(
            "browser_profiles",
            {"action": "select", "name": "personal"},
        )
        self.assertTrue(selected.success)
        listed = registry.execute(
            "browser_profiles", {"action": "list"}
        )
        names = {p["name"] for p in listed.output["profiles"]}
        self.assertEqual(names, {"default", "personal"})


if __name__ == "__main__":
    unittest.main()

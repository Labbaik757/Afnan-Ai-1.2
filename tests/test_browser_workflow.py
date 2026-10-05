"""Unified autonomous browser workflow: end-to-end scenarios.

One natural-language goal goes in; the agent plans, executes,
re-observes, verifies, recovers and answers from evidence —
search → open → extract, multi-tab comparison, form workflows,
pagination, downloads, dynamic pages, profile/session
continuity, and step/time/recovery limits.  All against fake
backends: no real browser, network or model, and no
website-specific logic anywhere in the agent.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from afnan_ai.agent import AfnanAgent
from afnan_ai.orchestrator import Agent as OrchestratorAgent
from afnan_ai.tools.base import Tool
from afnan_ai.tools.registry import ToolRegistry
from test_browser_advanced import (
    QueueLLM,
    el,
    item,
    make_controller,
    plan_json,
    probe_state,
    step,
)
from test_browser_operations import OpsBackend

SERP = "https://duckduckgo.com/html/?q=python+guide"
ARTICLE = "https://example.com/python-guide"
ARTICLE_TEXT = (
    "Python is a programming language known for readable syntax "
    "and a large ecosystem of libraries."
)


def _serp_page(backend):
    backend.add_page(
        SERP, title="DuckDuckGo",
        text="Search results for python guide",
        elements=[
            el("a", "Python Guide - Example",
               {"class": "result__a",
                "href": "//duckduckgo.com/l/?uddg=https%3A%2F%2F"
                        "example.com%2Fpython-guide&rut=x"}),
            el("div", "Python is a programming language known for "
                      "readable syntax.",
               {"class": "result__snippet"}),
        ],
    )
    backend.add_page(
        ARTICLE, title="Python Guide", text=ARTICLE_TEXT,
        elements=[el("a", "Official site",
                     {"href": "https://www.python.org/"})],
    )


def make_agent(replies, backend=None, **agent_kwargs):
    backend = backend or OpsBackend()
    controller, _ = make_controller(backend)
    agent = AfnanAgent(
        llm_provider=QueueLLM(replies),
        browser_controller=controller,
        **agent_kwargs,
    )
    return agent, backend


class TestSearchToExtraction(unittest.TestCase):
    def test_search_open_extract_end_to_end(self):
        backend = OpsBackend()
        _serp_page(backend)
        plan = plan_json("Research the Python guide", [
            step("s1", "browser_search",
                 {"query": "python guide"},
                 "Search results Python Guide"),
            step("s2", "browser_open_result", {"index": 0},
                 "Python programming language readable syntax"),
        ])
        agent, _ = make_agent([plan], backend=backend)
        outcome = agent.run_browser_goal("Research the Python guide")
        self.assertEqual(outcome.status, "completed")
        self.assertIn("readable syntax", outcome.answer)
        types = {e["type"] for e in outcome.evidence}
        self.assertIn("search_results", types)
        self.assertGreaterEqual(outcome.verified_steps, 1)
        state_dump = json.dumps(outcome.state.to_dict())
        self.assertIn("Browser briefing", state_dump)


class TestMultiTabComparison(unittest.TestCase):
    def test_two_tabs_compared_with_purposes(self):
        backend = OpsBackend()
        backend.add_page(
            "https://shop.example/a", title="Product A",
            text="Product A costs $10 with free shipping",
            elements=[],
        )
        backend.add_page(
            "https://shop.example/b", title="Product B",
            text="Product B costs $12 with fast delivery",
            elements=[],
        )
        plan = plan_json("Compare two products", [
            step("s1", "browser_new_tab",
                 {"url": "https://shop.example/a",
                  "purpose": "product-a"},
                 "Product A costs $10"),
            step("s2", "browser_extract_content", {},
                 "Product A $10 free shipping"),
            step("s3", "browser_new_tab",
                 {"url": "https://shop.example/b",
                  "purpose": "product-b"},
                 "Product B costs $12"),
            step("s4", "browser_extract_content", {},
                 "Product B $12 fast delivery"),
        ])
        agent, _ = make_agent([plan], backend=backend)
        outcome = agent.run_browser_goal("Compare two products")
        self.assertEqual(outcome.status, "completed")
        purposes = {t["purpose"] for t in outcome.tabs}
        self.assertIn("product-a", purposes)
        self.assertIn("product-b", purposes)
        state_dump = json.dumps(outcome.state.to_dict())
        self.assertIn("$10", state_dump)
        self.assertIn("$12", state_dump)


class TestFormWorkflow(unittest.TestCase):
    def _login_site(self, backend):
        backend.add_page(
            "https://bank.example/login", title="Sign in",
            text="Login page with username and password fields",
            elements=[
                el("input", "", {"id": "user", "type": "text",
                                 "class": "user-field"}),
                el("input", "", {"id": "pass", "type": "password",
                                 "class": "pass-field"}),
                el("button", "Sign in",
                   {"href": "https://bank.example/home"}),
            ],
        )
        backend.add_page(
            "https://bank.example/home", title="Home",
            text="Welcome to your dashboard",
            elements=[el("button", "Delete account", {"id": "del"})],
        )

    def _login_plan(self):
        return plan_json("Sign in and open the dashboard", [
            step("s1", "browser_navigate",
                 {"url": "https://bank.example/login"},
                 "Login page username password"),
            step("s2", "browser_type",
                 {"selector": ".user-field", "text": "afnan"},
                 "Login page username password"),
            step("s3", "browser_type",
                 {"selector": ".pass-field", "text": "s3cret-value"},
                 "Login page username password"),
            step("s4", "browser_click", {"selector": "button"},
                 "Welcome to your dashboard"),
        ])

    def test_login_form_workflow(self):
        backend = OpsBackend()
        self._login_site(backend)
        agent, _ = make_agent([self._login_plan()], backend=backend)
        outcome = agent.run_browser_goal(
            "Sign in and open the dashboard"
        )
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(
            agent.browser.current_page()["url"],
            "https://bank.example/home",
        )
        # the typed password never reaches the recorded state
        self.assertNotIn(
            "s3cret-value", json.dumps(outcome.state.to_dict())
        )

    def test_sensitive_action_blocked_without_approval(self):
        backend = OpsBackend()
        self._login_site(backend)
        backend.add_page(
            "https://bank.example/home", title="Home",
            text="Welcome to your dashboard",
            elements=[el("button", "Delete account", {"id": "del"})],
        )
        plan = plan_json("Delete the account", [
            step("s1", "browser_navigate",
                 {"url": "https://bank.example/home"},
                 "Welcome to your dashboard"),
            step("s2", "browser_click", {"selector": "button"},
                 "Welcome to your dashboard"),
        ])
        agent, _ = make_agent([plan, plan], backend=backend)
        outcome = agent.run_browser_goal("Delete the account")
        self.assertNotEqual(outcome.status, "completed")
        self.assertIn(
            "approval", json.dumps(outcome.state.to_dict())
        )

    def test_sensitive_action_runs_with_approval(self):
        backend = OpsBackend()
        self._login_site(backend)
        plan = plan_json("Delete the account", [
            step("s1", "browser_navigate",
                 {"url": "https://bank.example/home"},
                 "Welcome to your dashboard"),
            step("s2", "browser_click", {"selector": "button"},
                 "Welcome to your dashboard"),
        ])
        agent, _ = make_agent(
            [plan], backend=backend,
            browser_approver=lambda request: True,
        )
        outcome = agent.run_browser_goal("Delete the account")
        self.assertEqual(outcome.status, "completed")


class TestPaginationAndDownloads(unittest.TestCase):
    def test_pagination_collects_items(self):
        backend = OpsBackend()
        backend.add_page(
            "https://shop.example/feed", title="Feed", text="Feed",
            elements=[item(n) for n in range(1, 4)],
            batches=[
                [item(n) for n in (4, 5, 6)],
                [item(n) for n in (7,)],
            ],
        )
        plan = plan_json("Collect the feed items", [
            step("s1", "browser_navigate",
                 {"url": "https://shop.example/feed"}, "Feed"),
            step("s2", "browser_collect_items",
                 {"mode": "scroll", "max_items": 6},
                 "Collected 6 items"),
        ])
        agent, _ = make_agent([plan], backend=backend)
        outcome = agent.run_browser_goal("Collect the feed items")
        self.assertEqual(outcome.status, "completed")
        types = {e["type"] for e in outcome.evidence}
        self.assertIn("pagination", types)

    def test_download_tracked_and_verified(self):
        backend = OpsBackend()
        backend.add_page(
            "https://files.example/", title="Files",
            text="Files page. Download report (report.pdf) here",
            elements=[el("button", "Download report", {"id": "dl"})],
        )
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "report.pdf"
            pdf.write_bytes(b"%PDF-1.4 fake report")
            backend.download_list = [{
                "id": "dl_1", "url": "https://files.example/r.pdf",
                "filename": "report.pdf", "state": "completed",
                "path": str(pdf), "failure": None,
            }]
            plan = plan_json("Download the report", [
                step("s1", "browser_navigate",
                     {"url": "https://files.example/"},
                     "Files page Download report"),
                step("s2", "browser_click", {"selector": "button"},
                     "Files page Download report report.pdf"),
                step("s3", "browser_downloads",
                     {"action": "list"}, "report.pdf"),
            ])
            agent, _ = make_agent([plan], backend=backend)
            outcome = agent.run_browser_goal("Download the report")
            self.assertEqual(outcome.status, "completed")
            self.assertEqual(len(outcome.downloads), 1)
            self.assertEqual(
                outcome.downloads[0]["filename"], "report.pdf"
            )
            self.assertEqual(
                outcome.downloads[0]["integrity"], "ok"
            )


class TestDynamicSite(unittest.TestCase):
    def test_spa_waits_for_stable_then_extracts(self):
        backend = OpsBackend()
        backend.add_page(
            "https://spa.example/", title="App",
            text="Dashboard widgets ready: your reports, "
                 "messages and tasks have finished loading.",
            elements=[],
            probe_plan=[
                probe_state(text_len=10, hash_="loading"),
                probe_state(text_len=60, hash_="rendering"),
                probe_state(text_len=120, hash_="final"),
            ],
        )
        plan = plan_json("Read the dashboard", [
            step("s1", "browser_navigate",
                 {"url": "https://spa.example/"}, "Dashboard"),
            step("s2", "browser_wait_for_stable", {}, "stable"),
            step("s3", "browser_extract_content", {},
                 "Dashboard widgets ready"),
        ])
        agent, _ = make_agent([plan], backend=backend)
        outcome = agent.run_browser_goal("Read the dashboard")
        self.assertEqual(outcome.status, "completed")
        self.assertIn("Dashboard widgets", outcome.answer)


class TestRecoveryAndLimits(unittest.TestCase):
    def test_wrong_route_recovers_via_replan(self):
        backend = OpsBackend()
        backend.add_page(
            "https://site.example/", title="Home",
            text="Home page with an Open guide button",
            elements=[el("button", "Open guide", {"id": "open"})],
        )
        backend.add_page(
            ARTICLE, title="Python Guide", text=ARTICLE_TEXT,
            elements=[],
        )
        v1 = plan_json("Open the guide", [
            step("s1", "browser_navigate",
                 {"url": "https://site.example/"}, "Home page"),
            step("s2", "browser_click", {"selector": "#open"},
                 "Python programming language readable syntax"),
        ])
        v2 = plan_json("Open the guide", [
            step("r1", "browser_navigate", {"url": ARTICLE},
                 "Python programming language readable syntax"),
        ])
        agent, _ = make_agent([v1, v2], backend=backend)
        outcome = agent.run_browser_goal("Open the guide")
        self.assertEqual(outcome.status, "completed")
        self.assertGreaterEqual(outcome.recovery_attempts, 1)
        self.assertEqual(
            agent.browser.current_page()["url"], ARTICLE
        )

    def test_iteration_limit_enforced(self):
        backend = OpsBackend()
        _serp_page(backend)
        plan = plan_json("Long task", [
            step(f"s{i}", "browser_navigate", {"url": ARTICLE},
                 "Python programming language")
            for i in range(1, 5)
        ])
        agent, _ = make_agent([plan], backend=backend)
        outcome = agent.run_browser_goal(
            "Long task", max_iterations=2
        )
        self.assertEqual(outcome.status, "max_iterations_exceeded")
        self.assertEqual(outcome.iterations, 2)

    def test_time_budget_enforced(self):
        class SlowTool(Tool):
            name = "slow_step"
            description = "A deliberately slow step"
            input_schema = {
                "type": "object", "properties": {},
                "additionalProperties": False,
            }

            def run(self, arguments):
                time.sleep(0.25)
                return {"ok": True}

        registry = ToolRegistry()
        registry.register(SlowTool())
        plan = plan_json("Slow task", [
            step("s1", "slow_step", {}, "ok"),
            step("s2", "slow_step", {}, "ok"),
        ])
        orchestrator = OrchestratorAgent.from_components(
            QueueLLM([plan]), registry
        )
        result = orchestrator.run("Slow task", max_duration_s=0.1)
        self.assertEqual(result.status, "max_iterations_exceeded")
        self.assertEqual(result.error["code"], "time_limit_exceeded")
        self.assertEqual(result.iterations, 1)


class TestSessionContinuity(unittest.TestCase):
    def test_profile_keeps_session_between_goals(self):
        backend = OpsBackend()
        backend.add_page(
            "https://bank.example/home", title="Home",
            text="Welcome to your dashboard", elements=[],
        )
        open_plan = plan_json("Open the dashboard", [
            step("s1", "browser_new_tab",
                 {"url": "https://bank.example/home",
                  "purpose": "banking"},
                 "Welcome to your dashboard"),
        ])
        read_plan = plan_json("Read the dashboard", [
            step("s1", "browser_extract_content", {},
                 "Welcome to your dashboard"),
        ])
        agent, _ = make_agent([open_plan, read_plan], backend=backend)
        agent.browser.create_profile("work")

        first = agent.run_browser_goal(
            "Open the dashboard", profile="work"
        )
        self.assertEqual(first.status, "completed")
        self.assertEqual(agent.browser.current_profile, "work")

        # second goal, same profile: the briefing sees the tab
        # the first goal left open — no re-navigation needed
        second = agent.run_browser_goal(
            "Read the dashboard", profile="work"
        )
        self.assertEqual(second.status, "completed")
        urls = {t["url"] for t in second.tabs}
        self.assertIn("https://bank.example/home", urls)
        briefing = json.dumps(second.state.to_dict())
        self.assertIn("purpose=banking", briefing)

        # the default profile never saw this tab
        agent.browser.select_profile("default")
        default_urls = {t["url"] for t in agent.browser.list_tabs()}
        self.assertNotIn("https://bank.example/home", default_urls)


if __name__ == "__main__":
    unittest.main()

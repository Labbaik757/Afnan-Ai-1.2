import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai import Agent
from afnan_ai.agent import AfnanAgent
from afnan_ai.browser import (
    BrowserController,
    create_browser_tools,
    register_browser_tools,
)
from afnan_ai.executor import Executor
from afnan_ai.llm.base import LLMProvider
from afnan_ai.orchestrator import OrchestrationStatus
from afnan_ai.platform.base import PlatformAdapter
from afnan_ai.planner import Planner
from afnan_ai.tools import ToolRegistry
from afnan_ai.verifier import Verifier
from tests.test_browser_controller import FakeBrowserBackend


EXPECTED_TOOLS = [
    "browser_back",
    "browser_clear",
    "browser_click",
    "browser_close_tab",
    "browser_connect",
    "browser_current_page",
    "browser_find_elements",
    "browser_forward",
    "browser_inspect_element",
    "browser_launch",
    "browser_list_tabs",
    "browser_navigate",
    "browser_new_tab",
    "browser_press_key",
    "browser_reload",
    "browser_scroll",
    "browser_select_option",
    "browser_select_tab",
    "browser_shutdown",
    "browser_type",
]


def make_registry():
    backend = FakeBrowserBackend()
    controller = BrowserController(backend=backend)
    registry = ToolRegistry()
    register_browser_tools(registry, controller)
    return registry, controller, backend


class TestBrowserToolsViaRegistry(unittest.TestCase):
    def test_all_tools_registered(self):
        registry, _, _ = make_registry()
        self.assertEqual(sorted(registry.names()), EXPECTED_TOOLS)
        tools = create_browser_tools(BrowserController(backend=FakeBrowserBackend()))
        self.assertEqual(sorted(t.name for t in tools), EXPECTED_TOOLS)

    def test_full_browsing_flow(self):
        registry, controller, _ = make_registry()

        result = registry.execute("browser_launch", {})
        self.assertTrue(result.success)
        self.assertTrue(result.output["running"])

        result = registry.execute(
            "browser_new_tab", {"url": "https://example.com"}
        )
        self.assertTrue(result.success)
        self.assertEqual(result.output["tab_id"], "tab_1")

        result = registry.execute(
            "browser_navigate", {"url": "https://python.org"}
        )
        self.assertTrue(result.success)
        self.assertEqual(result.output["url"], "https://python.org")

        result = registry.execute("browser_back", {})
        self.assertEqual(result.output["url"], "https://example.com")

        result = registry.execute("browser_current_page", {})
        self.assertTrue(result.success)
        self.assertEqual(result.output["title"], "Example Domain")

        result = registry.execute("browser_list_tabs", {})
        self.assertEqual(len(result.output["tabs"]), 1)

        result = registry.execute("browser_reload", {})
        self.assertTrue(result.success)

        result = registry.execute("browser_shutdown", {})
        self.assertTrue(result.success)
        self.assertTrue(result.output["closed"])
        self.assertFalse(controller.is_running)

    def test_not_started_failure_is_structured(self):
        registry, _, _ = make_registry()
        result = registry.execute(
            "browser_navigate", {"url": "https://example.com"}
        )
        self.assertFalse(result.success)
        self.assertEqual(
            result.error.details["browser_error"]["code"],
            "browser_not_started",
        )

    def test_invalid_tab_failure_is_structured(self):
        registry, _, _ = make_registry()
        registry.execute("browser_launch", {})
        result = registry.execute(
            "browser_select_tab", {"tab_id": "tab_99"}
        )
        self.assertFalse(result.success)
        self.assertEqual(
            result.error.details["browser_error"]["code"], "invalid_tab"
        )

    def test_missing_argument_is_structured(self):
        registry, _, _ = make_registry()
        result = registry.execute("browser_navigate", {})
        self.assertFalse(result.success)
        self.assertEqual(result.error.code.value, "missing_arguments")

    def test_unavailable_browser_is_structured(self):
        backend = FakeBrowserBackend(fail_start=True)
        controller = BrowserController(backend=backend)
        registry = ToolRegistry()
        register_browser_tools(registry, controller)
        result = registry.execute("browser_launch", {})
        self.assertFalse(result.success)
        self.assertEqual(
            result.error.details["browser_error"]["code"],
            "browser_unavailable",
        )


class StubLLM(LLMProvider):
    name = "stub"
    display_name = "Stub"
    model = "stub-1"

    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages):
        return self.reply


class FakeAdapter(PlatformAdapter):
    name = "linux"

    def __init__(self):
        self.launched = []

    def speak_system(self, text):
        pass

    def open_path(self, path):
        pass

    def launch_app(self, app_key):
        self.launched.append(app_key)
        return True

    def find_folder(self, foldername):
        return None


class TestBrowserToolsInAgentPipeline(unittest.TestCase):
    def test_planner_can_select_browser_tools(self):
        backend = FakeBrowserBackend()
        controller = BrowserController(backend=backend)
        registry = ToolRegistry()
        register_browser_tools(registry, controller)
        plan = json.dumps(
            {
                "goal": "Open example.com and read its title",
                "steps": [
                    {
                        "step_id": "step_1",
                        "description": "Launch the browser",
                        "tool_name": "browser_launch",
                        "arguments": {},
                        "expected_result": "The browser is running",
                    },
                    {
                        "step_id": "step_2",
                        "description": "Navigate to example.com",
                        "tool_name": "browser_navigate",
                        "arguments": {"url": "https://example.com"},
                        "expected_result": "Example Domain page is open",
                    },
                ],
            }
        )
        agent = Agent(
            planner=Planner(StubLLM(plan), registry),
            executor=Executor(registry),
            verifier=Verifier(),
            recover_on_uncertain=False,
        )
        result = agent.run("Open example.com and read its title")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertTrue(controller.is_running)
        self.assertEqual(
            controller.current_url(), "https://example.com"
        )
        self.assertEqual(controller.title(), "Example Domain")

    def test_afnan_agent_has_browser_tools_by_default(self):
        adapter = FakeAdapter()
        backend = FakeBrowserBackend()
        controller = BrowserController(backend=backend)
        assistant = AfnanAgent(
            adapter=adapter, browser_controller=controller
        )
        for name in EXPECTED_TOOLS:
            self.assertIn(name, assistant.tools.names())
        result = assistant.execute_tool("browser_launch", {})
        self.assertTrue(result.success)
        self.assertTrue(controller.is_running)

    def test_browser_tools_can_be_disabled(self):
        assistant = AfnanAgent(
            adapter=FakeAdapter(), enable_browser_tools=False
        )
        self.assertNotIn("browser_launch", assistant.tools.names())
        # Phase 1 tools are unaffected
        self.assertIn("open_url", assistant.tools.names())


if __name__ == "__main__":
    unittest.main()

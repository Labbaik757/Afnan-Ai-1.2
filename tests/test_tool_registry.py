import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai.agent import AfnanAgent
from afnan_ai.platform.base import PlatformAdapter
from afnan_ai.tools import (
    FunctionTool,
    OpenApplicationTool,
    OpenUrlTool,
    SearchGoogleTool,
    TakeScreenshotTool,
    Tool,
    ToolErrorCode,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolRegistry,
    ToolValidationError,
    create_default_registry,
)


class FakeAdapter(PlatformAdapter):
    name = "linux"

    def __init__(self, launch_result=True):
        self.spoken = []
        self.opened = []
        self.launched = []
        self.launch_result = launch_result

    def speak_system(self, text):
        self.spoken.append(text)

    def open_path(self, path):
        self.opened.append(path)

    def launch_app(self, app_key):
        self.launched.append(app_key)
        return self.launch_result

    def find_folder(self, foldername):
        return None


class FakeImage:
    def __init__(self):
        self.saved_to = None

    def save(self, path):
        self.saved_to = path


def echo_tool():
    return FunctionTool(
        "echo",
        "Echo back the text.",
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        lambda text: f"echo: {text}",
    )


class TestRegistration(unittest.TestCase):
    def test_default_registry_has_required_tools(self):
        registry = create_default_registry(FakeAdapter())
        for name in (
            "open_url",
            "open_application",
            "search_google",
            "take_screenshot",
        ):
            self.assertIn(name, registry.names())
            self.assertTrue(registry.has(name))
            self.assertIn(name, registry)

    def test_tool_definition_has_name_description_schema(self):
        registry = create_default_registry(FakeAdapter())
        definitions = {d["name"]: d for d in registry.definitions()}
        open_url = definitions["open_url"]
        self.assertTrue(open_url["description"])
        self.assertEqual(open_url["input_schema"]["required"], ["url"])
        self.assertIn("url", open_url["input_schema"]["properties"])
        # definitions are serialisable (usable for LLM function calling)
        json.dumps(definitions)

    def test_register_and_get_custom_tool(self):
        registry = ToolRegistry()
        tool = echo_tool()
        registry.register(tool)
        self.assertIs(registry.get("echo"), tool)
        self.assertEqual(registry.names(), ["echo"])
        self.assertEqual(len(registry), 1)

    def test_register_duplicate_rejected(self):
        registry = ToolRegistry([echo_tool()])
        with self.assertRaises(ToolRegistrationError) as ctx:
            registry.register(echo_tool())
        self.assertEqual(
            ctx.exception.error.code, ToolErrorCode.TOOL_ALREADY_REGISTERED
        )

    def test_register_can_replace_when_asked(self):
        registry = ToolRegistry([echo_tool()])
        replacement = echo_tool()
        registry.register(replacement, replace=True)
        self.assertIs(registry.get("echo"), replacement)

    def test_register_non_tool_rejected(self):
        registry = ToolRegistry()
        with self.assertRaises(ToolRegistrationError) as ctx:
            registry.register("not a tool")
        self.assertEqual(ctx.exception.error.code, ToolErrorCode.INVALID_TOOL)

    def test_unregister(self):
        registry = ToolRegistry([echo_tool()])
        removed = registry.unregister("echo")
        self.assertEqual(removed.name, "echo")
        self.assertFalse(registry.has("echo"))

    def test_direct_tool_interface(self):
        tool = OpenUrlTool(opener=lambda url: True)
        self.assertEqual(tool.name, "open_url")
        self.assertTrue(tool.description)
        self.assertEqual(tool.execute({"url": "https://example.com"})["url"], "https://example.com")
        self.assertIsInstance(tool, Tool)


class TestExecution(unittest.TestCase):
    def test_execute_custom_tool_success(self):
        registry = ToolRegistry([echo_tool()])
        result = registry.execute("echo", {"text": "hello"})
        self.assertTrue(result.success)
        self.assertEqual(result.output, "echo: hello")
        self.assertIsNone(result.error)
        json.dumps(result.to_dict())

    def test_execute_open_url(self):
        opened = []
        registry = create_default_registry(
            FakeAdapter(), opener=lambda url: opened.append(url)
        )
        result = registry.execute("open_url", {"url": "https://youtube.com"})
        self.assertTrue(result.success)
        self.assertEqual(opened, ["https://youtube.com"])
        self.assertEqual(result.output["url"], "https://youtube.com")

    def test_execute_open_application(self):
        adapter = FakeAdapter()
        registry = create_default_registry(adapter)
        result = registry.execute("open_application", {"application": "chrome"})
        self.assertTrue(result.success)
        self.assertEqual(adapter.launched, ["chrome"])

    def test_open_application_aliases(self):
        adapter = FakeAdapter()
        tool = OpenApplicationTool(adapter)
        tool.execute({"application": "Google Chrome"})
        self.assertEqual(adapter.launched, ["chrome"])

    def test_execute_search_google_builds_url(self):
        opened = []
        registry = create_default_registry(
            FakeAdapter(), opener=lambda url: opened.append(url)
        )
        result = registry.execute("search_google", {"query": "python tutorial"})
        self.assertTrue(result.success)
        self.assertIn("google.com/search?q=python%20tutorial", opened[0])
        self.assertEqual(result.output["query"], "python tutorial")

    def test_execute_search_google_tool_class(self):
        self.assertEqual(
            SearchGoogleTool.build_url("a b"),
            "https://www.google.com/search?q=a%20b",
        )

    def test_execute_take_screenshot(self):
        adapter = FakeAdapter()
        image = FakeImage()
        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(
                [TakeScreenshotTool(adapter, capture=lambda: image)]
            )
            result = registry.execute("take_screenshot", {"output_dir": tmp})
            self.assertTrue(result.success)
            self.assertTrue(result.output.endswith(".png"))
            self.assertEqual(image.saved_to, result.output)
            self.assertEqual(adapter.opened, [result.output])

    def test_execute_or_raise_returns_output(self):
        registry = ToolRegistry([echo_tool()])
        self.assertEqual(registry.execute_or_raise("echo", {"text": "x"}), "echo: x")


class TestStructuredErrors(unittest.TestCase):
    def test_invalid_tool_name_returns_structured_error(self):
        registry = create_default_registry(FakeAdapter())
        result = registry.execute("does_not_exist", {})
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.code, ToolErrorCode.TOOL_NOT_FOUND)
        self.assertEqual(result.error.tool, "does_not_exist")
        self.assertIn("open_url", result.error.details["available"])
        json.dumps(result.to_dict())  # must be serialisable

    def test_get_invalid_tool_raises_structured(self):
        registry = ToolRegistry()
        with self.assertRaises(ToolNotFoundError) as ctx:
            registry.get("nope")
        self.assertEqual(ctx.exception.error.code, ToolErrorCode.TOOL_NOT_FOUND)
        self.assertIsNone(registry.get_or_none("nope"))

    def test_missing_arguments(self):
        registry = create_default_registry(FakeAdapter(), opener=lambda url: None)
        for name in ("open_url", "open_application", "search_google"):
            result = registry.execute(name, {})
            self.assertFalse(result.success, name)
            self.assertEqual(
                result.error.code, ToolErrorCode.MISSING_ARGUMENTS, name
            )
            self.assertTrue(result.error.details["missing"], name)

    def test_missing_arguments_via_kwargs_and_direct_tool(self):
        registry = ToolRegistry([echo_tool()])
        result = registry.execute("echo")
        self.assertEqual(result.error.code, ToolErrorCode.MISSING_ARGUMENTS)
        with self.assertRaises(ToolValidationError):
            echo_tool().execute({})

    def test_invalid_argument_type(self):
        registry = create_default_registry(FakeAdapter(), opener=lambda url: None)
        result = registry.execute("open_url", {"url": 123})
        self.assertFalse(result.success)
        self.assertEqual(result.error.code, ToolErrorCode.INVALID_ARGUMENTS)

    def test_unknown_argument_rejected(self):
        registry = ToolRegistry([echo_tool()])
        result = registry.execute("echo", {"text": "x", "surprise": 1})
        self.assertFalse(result.success)
        self.assertEqual(result.error.code, ToolErrorCode.INVALID_ARGUMENTS)

    def test_blank_required_argument_is_missing(self):
        registry = create_default_registry(FakeAdapter(), opener=lambda url: None)
        result = registry.execute("search_google", {"query": "   "})
        self.assertEqual(result.error.code, ToolErrorCode.MISSING_ARGUMENTS)

    def test_execution_failure_from_crashing_tool(self):
        def boom(text):
            raise RuntimeError("kaboom")

        registry = ToolRegistry(
            [
                FunctionTool(
                    "boom",
                    "Always fails.",
                    {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                    boom,
                )
            ]
        )
        result = registry.execute("boom", {"text": "x"})  # must not raise
        self.assertFalse(result.success)
        self.assertEqual(result.error.code, ToolErrorCode.EXECUTION_FAILED)
        self.assertIn("kaboom", result.error.message)

    def test_execution_failure_when_app_unavailable(self):
        adapter = FakeAdapter(launch_result=False)
        registry = create_default_registry(adapter)
        result = registry.execute("open_application", {"application": "chrome"})
        self.assertFalse(result.success)
        self.assertEqual(result.error.code, ToolErrorCode.EXECUTION_FAILED)

    def test_execution_failure_when_opener_raises(self):
        def bad_opener(url):
            raise OSError("no browser")

        registry = ToolRegistry([OpenUrlTool(opener=bad_opener)])
        result = registry.execute("open_url", {"url": "https://example.com"})
        self.assertFalse(result.success)
        self.assertEqual(result.error.code, ToolErrorCode.EXECUTION_FAILED)

    def test_screenshot_unavailable_is_execution_failure(self):
        registry = ToolRegistry(
            [TakeScreenshotTool(FakeAdapter(), capture=lambda: None)]
        )
        result = registry.execute("take_screenshot", {})
        self.assertFalse(result.success)
        self.assertEqual(result.error.code, ToolErrorCode.EXECUTION_FAILED)

    def test_tool_execution_error_preserved(self):
        tool = TakeScreenshotTool(FakeAdapter(), capture=lambda: None)
        with self.assertRaises(ToolExecutionError):
            tool.execute({})

    def test_execute_or_raise_raises_structured(self):
        registry = ToolRegistry()
        with self.assertRaises(ToolNotFoundError):
            registry.execute_or_raise("ghost", {})


class TestAgentToolIntegration(unittest.TestCase):
    def make_agent(self, adapter=None):
        agent = AfnanAgent(adapter=adapter or FakeAdapter())
        agent.speak = lambda text: agent.adapter.spoken.append(text)
        return agent

    def test_agent_has_default_tools(self):
        agent = self.make_agent()
        names = [d["name"] for d in agent.list_tools()]
        for name in ("open_url", "open_application", "search_google", "take_screenshot"):
            self.assertIn(name, names)

    def test_agent_open_chrome_goes_through_tool(self):
        agent = self.make_agent()
        agent.process_command("open chrome")
        self.assertIn("chrome", agent.adapter.launched)
        tool_names = [r.tool for r in agent.state.tool_results]
        self.assertIn("open_application", tool_names)

    def test_agent_search_google_goes_through_tool(self):
        agent = self.make_agent()
        with mock.patch("webbrowser.open") as wb:
            agent.process_command("search google for python")
            self.assertIn("google.com/search", str(wb.call_args))
        tool_names = [r.tool for r in agent.state.tool_results]
        self.assertIn("search_google", tool_names)

    def test_agent_screenshot_goes_through_tool(self):
        agent = self.make_agent()
        fake_shot = mock.Mock()
        with mock.patch("afnan_ai.agent.pyautogui") as pg:
            pg.screenshot.return_value = fake_shot
            result = agent.take_screenshot()
            self.assertTrue(result.startswith("screenshots"))
            self.assertTrue(agent.adapter.opened)
        tool_names = [r.tool for r in agent.state.tool_results]
        self.assertIn("take_screenshot", tool_names)

    def test_agent_execute_tool_invalid_returns_structured(self):
        agent = self.make_agent()
        result = agent.execute_tool("no_such_tool", {})
        self.assertFalse(result.success)
        self.assertEqual(result.error.code, ToolErrorCode.TOOL_NOT_FOUND)
        # ...and the failure is visible in AgentState
        self.assertFalse(agent.state.tool_results[-1].success)

    def test_agent_execute_tool_missing_args(self):
        agent = self.make_agent()
        result = agent.execute_tool("open_url", {})
        self.assertFalse(result.success)
        self.assertEqual(result.error.code, ToolErrorCode.MISSING_ARGUMENTS)

    def test_agent_chrome_fallback_when_app_fails(self):
        agent = self.make_agent(FakeAdapter(launch_result=False))
        with mock.patch("webbrowser.open") as wb:
            agent.process_command("open chrome")
            wb.assert_called_once_with("https://www.google.com")

    def test_agent_accepts_a_custom_registry(self):
        registry = ToolRegistry([echo_tool()])
        agent = AfnanAgent(adapter=FakeAdapter(), tool_registry=registry)
        result = agent.execute_tool("echo", {"text": "hi"})
        self.assertEqual(result.output, "echo: hi")

    def test_agent_can_register_a_new_tool_dynamically(self):
        agent = self.make_agent()
        agent.tools.register(echo_tool())
        self.assertEqual(agent.execute_tool("echo", {"text": "yo"}).output, "echo: yo")


if __name__ == "__main__":
    unittest.main()

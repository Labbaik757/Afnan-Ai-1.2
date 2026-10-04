import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai.agent import AfnanAgent
from afnan_ai.llm import (
    LLMConnectionError,
    LLMInvalidResponseError,
    LLMProvider,
    LLMUnavailableError,
    OllamaProvider,
    available_providers,
    create_provider,
    get_default_provider,
)
from afnan_ai.platform.base import PlatformAdapter


class FakeClient:
    """Stand-in for the ollama client, recording calls."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def chat(self, model, messages):
        self.calls.append({"model": model, "messages": messages})
        if self.error is not None:
            raise self.error
        return self.response


class FakeAdapter(PlatformAdapter):
    name = "linux"

    def __init__(self):
        self.spoken = []
        self.opened = []
        self.launched = []

    def speak_system(self, text):
        self.spoken.append(text)

    def open_path(self, path):
        self.opened.append(path)

    def launch_app(self, app_key):
        self.launched.append(app_key)
        return True

    def find_folder(self, foldername):
        return None


def make_agent(provider):
    agent = AfnanAgent(adapter=FakeAdapter(), llm_provider=provider)
    agent.speak = lambda text: agent.adapter.spoken.append(text)
    return agent


class TestSuccessfulResponse(unittest.TestCase):
    def test_ollama_provider_returns_message_content(self):
        client = FakeClient(response={"message": {"content": "Hello boss"}})
        provider = OllamaProvider(client=client)
        reply = provider.generate("hello")
        self.assertEqual(reply, "Hello boss")
        # existing behaviour preserved: llama3 + single user message
        self.assertEqual(client.calls[0]["model"], "llama3")
        self.assertEqual(
            client.calls[0]["messages"], [{"role": "user", "content": "hello"}]
        )

    def test_ollama_provider_chat_with_messages(self):
        client = FakeClient(response={"message": {"content": "ok"}})
        provider = OllamaProvider(model="llama3", client=client)
        reply = provider.chat(
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ]
        )
        self.assertEqual(reply, "ok")
        self.assertEqual(len(client.calls[0]["messages"]), 2)

    def test_custom_model_is_used(self):
        client = FakeClient(response={"message": {"content": "ok"}})
        provider = OllamaProvider(model="mistral", client=client)
        provider.generate("hi")
        self.assertEqual(client.calls[0]["model"], "mistral")

    def test_agent_returns_provider_reply_for_unknown_command(self):
        client = FakeClient(response={"message": {"content": "AI answer"}})
        agent = make_agent(OllamaProvider(client=client))
        agent.process_command("what is python")
        self.assertIn("AI answer", agent.adapter.spoken)

    def test_agent_ask_local_ai_preserves_existing_api(self):
        client = FakeClient(response={"message": {"content": "local reply"}})
        agent = make_agent(OllamaProvider(client=client))
        self.assertEqual(agent.ask_local_ai("hi"), "local reply")
        self.assertEqual(agent.ask_ai("hi"), "local reply")


class TestConnectionFailure(unittest.TestCase):
    def test_connection_error_is_wrapped(self):
        client = FakeClient(error=ConnectionError("Connection refused"))
        provider = OllamaProvider(client=client)
        with self.assertRaises(LLMConnectionError):
            provider.generate("hi")

    def test_any_client_failure_becomes_connection_error(self):
        client = FakeClient(error=RuntimeError("server exploded"))
        provider = OllamaProvider(client=client)
        with self.assertRaises(LLMConnectionError):
            provider.chat([{"role": "user", "content": "hi"}])

    def test_agent_speaks_not_responding_on_connection_failure(self):
        # exact message Afnan has always spoken when Ollama is down
        client = FakeClient(error=ConnectionError("refused"))
        agent = make_agent(OllamaProvider(client=client))
        reply = agent.ask_local_ai("hi")
        self.assertEqual(
            reply,
            "Sorry boss, AI is not responding. Make sure Ollama is running.",
        )

    def test_agent_unknown_command_survives_connection_failure(self):
        client = FakeClient(error=OSError("no route"))
        agent = make_agent(OllamaProvider(client=client))
        agent.process_command("tell me a joke")  # must not raise
        self.assertIn(
            "Sorry boss, AI is not responding. Make sure Ollama is running.",
            agent.adapter.spoken,
        )

    def test_unavailable_when_client_missing(self):
        provider = OllamaProvider(client=None)
        # Force the lazy-import path to report no client, without
        # depending on whether the ollama package is installed here
        provider._client_resolved = True
        provider._client = None
        self.assertFalse(provider.is_available)
        with self.assertRaises(LLMUnavailableError):
            provider.generate("hi")

    def test_agent_speaks_not_available_when_unavailable(self):
        provider = OllamaProvider(client=None)
        provider._client_resolved = True
        provider._client = None
        agent = make_agent(provider)
        self.assertEqual(
            agent.ask_local_ai("hi"),
            "Sorry boss, AI is not available. Ollama is not installed.",
        )


class TestInvalidResponse(unittest.TestCase):
    def test_missing_message_key_is_invalid(self):
        provider = OllamaProvider(client=FakeClient(response={"response": "x"}))
        with self.assertRaises(LLMInvalidResponseError):
            provider.generate("hi")

    def test_missing_content_is_invalid(self):
        provider = OllamaProvider(client=FakeClient(response={"message": {}}))
        with self.assertRaises(LLMInvalidResponseError):
            provider.generate("hi")

    def test_empty_content_is_invalid(self):
        provider = OllamaProvider(
            client=FakeClient(response={"message": {"content": "   "}})
        )
        with self.assertRaises(LLMInvalidResponseError):
            provider.generate("hi")

    def test_none_response_is_invalid(self):
        provider = OllamaProvider(client=FakeClient(response=None))
        with self.assertRaises(LLMInvalidResponseError):
            provider.generate("hi")

    def test_non_string_content_is_invalid(self):
        provider = OllamaProvider(
            client=FakeClient(response={"message": {"content": 42}})
        )
        with self.assertRaises(LLMInvalidResponseError):
            provider.generate("hi")

    def test_agent_treats_invalid_response_as_not_responding(self):
        agent = make_agent(
            OllamaProvider(client=FakeClient(response={"unexpected": True}))
        )
        self.assertEqual(
            agent.ask_ai("hi"),
            "Sorry boss, AI is not responding. Make sure Ollama is running.",
        )


class EchoProvider(LLMProvider):
    """A second provider proving the agent needs no code change."""

    name = "echo"
    display_name = "Echo"
    model = "echo-1"

    def chat(self, messages):
        return "echo: " + messages[-1]["content"]


class TestProviderInterface(unittest.TestCase):
    def test_agent_works_with_a_completely_different_provider(self):
        agent = make_agent(EchoProvider())
        self.assertEqual(agent.ask_ai("hello"), "echo: hello")
        agent.process_command("some unknown thing")
        self.assertIn("echo: some unknown thing", agent.adapter.spoken)

    def test_default_provider_is_ollama(self):
        provider = get_default_provider()
        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(provider.model, "llama3")

    def test_create_provider_by_name(self):
        self.assertIsInstance(create_provider("ollama"), OllamaProvider)
        self.assertIn("ollama", available_providers())

    def test_unknown_provider_rejected(self):
        with self.assertRaises(LLMUnavailableError):
            create_provider("does-not-exist")

    def test_agent_and_main_have_no_direct_ollama_call(self):
        root = Path(__file__).resolve().parents[1]
        for rel in ("afnan_ai/agent.py", "main.py"):
            src = (root / rel).read_text(encoding="utf-8")
            self.assertNotIn("ollama.chat", src, rel)
            self.assertNotIn("import ollama", src, rel)
            self.assertNotIn("ollama.Client", src, rel)

    def test_ollama_import_is_only_inside_provider(self):
        root = Path(__file__).resolve().parents[1]
        src = (root / "afnan_ai" / "llm" / "ollama.py").read_text(encoding="utf-8")
        self.assertIn("import ollama", src)


if __name__ == "__main__":
    unittest.main()

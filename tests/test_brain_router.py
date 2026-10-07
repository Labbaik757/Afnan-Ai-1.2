"""Tests for the hybrid brain router (afnan_ai/llm/router.py)."""

import time
import unittest
from collections.abc import Iterator, Sequence

from afnan_ai.llm.base import (
    ChatMessage,
    LLMConnectionError,
    LLMProvider,
    LLMUnavailableError,
)
from afnan_ai.llm.router import (
    OFFLINE_PREFIX,
    BrainRouter,
    classify_intent,
)


class FakeProvider(LLMProvider):
    """Controllable stand-in for a real provider."""

    def __init__(self, name="fake", model="fake-1b", reply="fake reply",
                 delay_s=0.0, fail=False):
        self.name = name
        self.display_name = name.title()
        self.model = model
        self._reply = reply
        self._delay = delay_s
        self._fail = fail
        self.calls: list[list[ChatMessage]] = []

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        self.calls.append(list(messages))
        if self._fail:
            raise LLMConnectionError("fake cloud down")
        if self._delay:
            time.sleep(self._delay)
        return self._reply

    def chat_stream(
        self, messages: Sequence[ChatMessage]
    ) -> Iterator[str]:
        yield self.chat(messages)


def _router(**kw) -> BrainRouter:
    local = FakeProvider(name="ollama", model="llama3.2:1b",
                         reply="local reply")
    cloud = FakeProvider(name="groq", model="openai/gpt-oss-120b",
                         reply="cloud reply")
    return BrainRouter(local, cloud, **kw), local, cloud


def _msg(text: str) -> list[ChatMessage]:
    return [{"role": "user", "content": text}]


class TestClassifyIntent(unittest.TestCase):
    def test_english_greeting_instant(self):
        self.assertEqual(classify_intent("hello"), "instant")
        self.assertEqual(classify_intent("Hello, how are you?"), "instant")

    def test_roman_urdu_greeting_instant(self):
        self.assertEqual(classify_intent("assalam o alaikum"), "instant")
        self.assertEqual(classify_intent("kya haal hai"), "instant")
        self.assertEqual(classify_intent("shukriya"), "instant")

    def test_urdu_script_greeting_instant(self):
        self.assertEqual(classify_intent("سلام"), "instant")
        self.assertEqual(classify_intent("کیا حال ہے؟"), "instant")

    def test_thank_you_instant(self):
        self.assertEqual(classify_intent("thank you so much"), "instant")

    def test_simple_question_instant(self):
        self.assertEqual(classify_intent("hi"), "instant")
        self.assertEqual(classify_intent("ok"), "instant")

    def test_english_task_genius(self):
        self.assertEqual(
            classify_intent("open browser and search for cats"), "genius")
        self.assertEqual(
            classify_intent("create a signup form"), "genius")

    def test_roman_urdu_task_genius(self):
        self.assertEqual(classify_intent("browser kholo"), "genius")
        self.assertEqual(classify_intent("youtube kholo"), "genius")
        self.assertEqual(classify_intent("form bharo"), "genius")

    def test_urdu_script_task_genius(self):
        self.assertEqual(classify_intent("براؤزر کھولو"), "genius")
        self.assertEqual(classify_intent("یوٹیوب کھولو"), "genius")

    def test_deep_question_genius(self):
        self.assertEqual(
            classify_intent("explain quantum physics in detail"), "genius")
        self.assertEqual(
            classify_intent("compare python and javascript for me"),
            "genius")
        self.assertEqual(
            classify_intent("quantum physics samjhao tafseel se"), "genius")

    def test_private_password(self):
        self.assertEqual(
            classify_intent("my password is hunter2"), "private")

    def test_private_bank(self):
        self.assertEqual(
            classify_intent("my bank account number is 1234"), "private")

    def test_private_cnic(self):
        self.assertEqual(
            classify_intent("mera cnic number note kar lo"), "private")

    def test_private_medical(self):
        self.assertEqual(
            classify_intent("I have a medical issue, doctor ko dikhana hai"),
            "private")

    def test_private_beats_task(self):
        # "open my bank account" is a task AND private → private wins.
        self.assertEqual(
            classify_intent("open my bank account page"), "private")

    def test_empty_text_instant(self):
        self.assertEqual(classify_intent(""), "instant")
        self.assertEqual(classify_intent("   "), "instant")


class TestRoute(unittest.TestCase):
    def test_instant_routes_to_local(self):
        router, local, cloud = _router()
        self.assertIs(router.route("hello"), local)
        self.assertEqual(router.last_lane, "instant")

    def test_genius_routes_to_cloud(self):
        router, local, cloud = _router()
        self.assertIs(router.route("browser kholo"), cloud)
        self.assertEqual(router.last_lane, "genius")

    def test_private_routes_to_local(self):
        router, local, cloud = _router()
        self.assertIs(router.route("my password is x"), local)
        self.assertEqual(router.last_lane, "private")

    def test_context_hint_forces_lane(self):
        router, local, cloud = _router()
        self.assertIs(router.route("hello", context_hint="genius"), cloud)
        self.assertIs(router.route("open browser", context_hint="instant"),
                      local)

    def test_private_without_local_raises(self):
        cloud = FakeProvider(name="groq", reply="cloud")
        router = BrainRouter(None, cloud)  # type: ignore[arg-type]
        with self.assertRaises(LLMUnavailableError):
            router.route("my password is x")


class TestChat(unittest.TestCase):
    def test_chat_greeting_uses_local(self):
        router, local, cloud = _router()
        reply = router.chat(_msg("hello"))
        self.assertEqual(reply, "local reply")
        self.assertEqual(len(local.calls), 1)
        self.assertEqual(len(cloud.calls), 0)

    def test_chat_task_uses_cloud(self):
        router, local, cloud = _router()
        reply = router.chat(_msg("open browser"))
        self.assertEqual(reply, "cloud reply")
        self.assertEqual(len(cloud.calls), 1)
        self.assertEqual(len(local.calls), 0)

    def test_chat_private_never_touches_cloud(self):
        router, local, cloud = _router()
        reply = router.chat(_msg("my bank pin is 1234"))
        self.assertEqual(reply, "local reply")
        self.assertEqual(len(cloud.calls), 0)
        self.assertEqual(len(local.calls), 1)

    def test_cloud_timeout_falls_back_with_prefix(self):
        slow_cloud = FakeProvider(name="groq", reply="cloud reply",
                                  delay_s=5.0)
        local = FakeProvider(name="ollama", reply="local reply")
        router = BrainRouter(local, slow_cloud, max_cloud_latency_s=0.3)
        reply = router.chat(_msg("open browser"))
        self.assertTrue(reply.startswith(OFFLINE_PREFIX),
                        f"missing offline prefix: {reply!r}")
        self.assertIn("local reply", reply)
        self.assertEqual(router.last_lane, "instant")

    def test_cloud_failure_falls_back_with_prefix(self):
        dead_cloud = FakeProvider(name="groq", fail=True)
        local = FakeProvider(name="ollama", reply="local reply")
        router = BrainRouter(local, dead_cloud)
        reply = router.chat(_msg("explain this"))
        self.assertTrue(reply.startswith(OFFLINE_PREFIX))
        self.assertIn("local reply", reply)

    def test_chat_uses_last_user_message(self):
        router, local, cloud = _router()
        messages = [
            {"role": "user", "content": "open browser"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "thanks"},
        ]
        reply = router.chat(messages)
        self.assertEqual(reply, "local reply")  # "thanks" → instant

    def test_generate_routes(self):
        router, local, cloud = _router()
        self.assertEqual(router.generate("hello"), "local reply")
        self.assertEqual(router.generate("open browser"), "cloud reply")

    def test_chat_stream_delegates_to_local(self):
        router, local, cloud = _router()
        chunks = list(router.chat_stream(_msg("hello")))
        self.assertEqual("".join(chunks), "local reply")

    def test_chat_stream_cloud_failure_falls_back(self):
        dead_cloud = FakeProvider(name="groq", fail=True)
        local = FakeProvider(name="ollama", reply="local reply")
        router = BrainRouter(local, dead_cloud)
        chunks = list(router.chat_stream(_msg("open browser")))
        text = "".join(chunks)
        self.assertTrue(text.startswith(OFFLINE_PREFIX))
        self.assertIn("local reply", text)

    def test_status_reports_lanes(self):
        router, local, cloud = _router()
        router.chat(_msg("browser kholo"))
        st = router.status()
        self.assertEqual(st["last_lane"], "genius")
        self.assertEqual(st["cloud"]["model"], "openai/gpt-oss-120b")
        self.assertEqual(st["local"]["model"], "llama3.2:1b")

    def test_is_available(self):
        router, local, cloud = _router()
        self.assertTrue(router.is_available)


if __name__ == "__main__":
    unittest.main()

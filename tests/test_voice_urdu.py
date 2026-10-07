"""Tests for Urdu voice: on-device wake word, Urdu STT
language config, and the Urdu TTS fallback chain."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from afnan_ai import speech as speech_module
from afnan_ai.config import AgentConfig
from afnan_ai.wakeword import (
    FakeWakeWordDetector,
    OpenWakeWordDetector,
    WakeWordDetector,
    create_detector,
    detector_status,
)


class DetectorInterfaceTests(unittest.TestCase):
    def test_fake_scores_and_threshold(self):
        det = FakeWakeWordDetector([0.1, 0.9, 0.4])
        self.assertFalse(det.is_wake(b"\x00" * 2560, 0.5))
        self.assertTrue(det.is_wake(b"\x00" * 2560, 0.5))
        self.assertFalse(det.is_wake(b"\x00" * 2560, 0.5))
        self.assertEqual(det.calls, 3)

    def test_constant_score(self):
        det = FakeWakeWordDetector([0.8])
        self.assertTrue(det.is_wake(b"\x00" * 10, 0.5))
        self.assertTrue(det.is_wake(b"\x00" * 10, 0.5))

    def test_is_wake_never_raises(self):
        class Boom(WakeWordDetector):
            def score(self, frame: bytes) -> float:
                raise RuntimeError("boom")

        self.assertFalse(Boom().is_wake(b"\x00", 0.5))

    def test_frame_samples_default(self):
        det = FakeWakeWordDetector()
        self.assertEqual(det.frame_samples, 1280)


class CreateDetectorTests(unittest.TestCase):
    def test_no_model_returns_none(self):
        self.assertIsNone(create_detector(None))
        self.assertIsNone(create_detector(""))

    def test_missing_model_returns_none(self):
        self.assertIsNone(
            create_detector("/no/such/model.onnx")
        )

    def test_fake_scores_returns_fake(self):
        det = create_detector(None, fake_scores=[0.9])
        self.assertIsInstance(det, FakeWakeWordDetector)
        self.assertTrue(det.is_wake(b"\x00", 0.5))

    def test_missing_library_returns_none(self):
        with mock.patch.dict("sys.modules", {"openwakeword": None}):
            # openwakeword.model import fails → None
            det = create_detector(
                "/tmp/fake.onnx",
            )
            self.assertIsNone(det)

    def test_status_report(self):
        status = detector_status(None)
        self.assertFalse(status["model_configured"])
        self.assertFalse(status["on_device"])
        self.assertIn("library_installed", status)


class UrduDetectionTests(unittest.TestCase):
    def test_urdu_detected(self):
        self.assertTrue(
            speech_module.contains_urdu("افنان کیسے ہو")
        )

    def test_english_not_urdu(self):
        self.assertFalse(
            speech_module.contains_urdu("how are you Afnan")
        )

    def test_mixed_detected(self):
        self.assertTrue(
            speech_module.contains_urdu("hello افنان")
        )

    def test_empty(self):
        self.assertFalse(speech_module.contains_urdu(""))


class TtsChainTests(unittest.TestCase):
    def _fake_adapter(self):
        class FakeAdapter:
            def __init__(self):
                self.spoken = []

            def speak_system(self, text):
                self.spoken.append(text)

        return FakeAdapter()

    def test_urdu_prefers_edge_tts(self):
        adapter = self._fake_adapter()
        calls = []
        with mock.patch.object(
            speech_module,
            "_speak_with_edge_tts",
            side_effect=lambda t, voice=None: calls.append(
                ("edge", t, voice)
            )
            or True,
        ), mock.patch.object(
            speech_module,
            "_speak_with_pyttsx3",
            side_effect=lambda t: calls.append(("pyttsx3", t)),
        ):
            speech_module.speak("افنان", adapter)
        self.assertEqual(calls[0][0], "edge")
        self.assertNotIn("pyttsx3", [c[0] for c in calls])

    def test_urdu_falls_back_through_chain(self):
        adapter = self._fake_adapter()
        with mock.patch.object(
            speech_module, "_speak_with_edge_tts",
            return_value=False,
        ), mock.patch.object(
            speech_module, "_speak_with_gtts", return_value=False
        ), mock.patch.object(
            speech_module, "_speak_with_pyttsx3",
            return_value=None,
        ) as pyttsx3_mock:
            speech_module.speak("افنان", adapter)
        pyttsx3_mock.assert_called_once_with("افنان")

    def test_english_skips_urdu_chain(self):
        adapter = self._fake_adapter()
        with mock.patch.object(
            speech_module, "_speak_with_edge_tts"
        ) as edge_mock, mock.patch.object(
            speech_module,
            "_speak_with_pyttsx3",
            return_value=None,
        ):
            speech_module.speak("hello", adapter)
        edge_mock.assert_not_called()

    def test_edge_tts_missing_library(self):
        with mock.patch.dict("sys.modules", {"edge_tts": None}):
            self.assertFalse(
                speech_module._speak_with_edge_tts("x", voice="v")
            )

    def test_gtts_missing_library(self):
        with mock.patch.dict("sys.modules", {"gtts": None}):
            self.assertFalse(
                speech_module._speak_with_gtts("x")
            )

    def test_speak_never_raises(self):
        adapter = self._fake_adapter()
        with mock.patch.object(
            speech_module, "_speak_with_pyttsx3",
            side_effect=RuntimeError("no engine"),
        ):
            # adapter.speak_system appends; must not raise
            speech_module.speak("hello", adapter)
        self.assertEqual(adapter.spoken, ["hello"])


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        config = AgentConfig()
        self.assertEqual(config.stt_language, "ur-PK")
        # Bundled model is the default when present in repo.
        import os

        if config.wakeword_model is not None:
            self.assertTrue(
                config.wakeword_model.endswith(
                    ("afnan.json", "afnan.onnx")
                )
            )
            self.assertTrue(
                os.path.exists(config.wakeword_model)
            )
        self.assertEqual(config.wakeword_threshold, 0.7)
        self.assertEqual(
            config.tts_urdu_voice, "ur-PK-GulNawazNeural"
        )

    def test_env_overrides(self):
        env = {
            "AFNAN_STT_LANGUAGE": "en-IN",
            "AFNAN_WAKEWORD_MODEL": "/tmp/afnan.onnx",
            "AFNAN_WAKEWORD_THRESHOLD": "0.7",
            "AFNAN_TTS_URDU_VOICE": "ur-PK-AsadNeural",
        }
        with mock.patch.dict(os.environ, env):
            config = AgentConfig.from_env()
        self.assertEqual(config.stt_language, "en-IN")
        self.assertEqual(
            config.wakeword_model, "/tmp/afnan.onnx"
        )
        self.assertEqual(config.wakeword_threshold, 0.7)
        self.assertEqual(
            config.tts_urdu_voice, "ur-PK-AsadNeural"
        )

    def test_bad_threshold_falls_back(self):
        with mock.patch.dict(
            os.environ, {"AFNAN_WAKEWORD_THRESHOLD": "nope"}
        ):
            config = AgentConfig.from_env()
        self.assertEqual(config.wakeword_threshold, 0.7)


class AgentVoiceWiringTests(unittest.TestCase):
    def _make_agent(self):
        from afnan_ai.agent import AfnanAgent
        from afnan_ai.platform.base import PlatformAdapter

        class FakeAdapter(PlatformAdapter):
            name = "linux"

            def speak_system(self, text):
                pass

            def open_path(self, path):
                pass

            def launch_app(self, app_key):
                return True

            def find_folder(self, foldername):
                return None

        agent = AfnanAgent(adapter=FakeAdapter())
        return agent

    def test_listen_uses_config_language(self):
        agent = self._make_agent()
        agent.config = AgentConfig(stt_language="ur-PK")
        captured = {}

        class FakeAudio:
            pass

        class FakeRecognizer:
            def adjust_for_ambient_noise(
                self, source, duration=0.5
            ):
                pass

            def listen(
                self, source, timeout=None,
                phrase_time_limit=None,
            ):
                return FakeAudio()

            def recognize_google(self, audio, language=None):
                captured["language"] = language
                return "افنان"

        agent.recognizer = FakeRecognizer()
        with mock.patch(
            "afnan_ai.agent.sr"
        ) as sr_mock:
            sr_mock.Microphone.return_value.__enter__.return_value = (
                object()
            )
            # sr.Microphone() context manager
            cm = mock.MagicMock()
            cm.__enter__.return_value = object()
            sr_mock.Microphone.return_value = cm
            result = agent.listen_command()
        self.assertEqual(result, "افنان")
        self.assertEqual(captured["language"], "ur-PK")

    def test_speak_passes_urdu_voice(self):
        agent = self._make_agent()
        agent.config = AgentConfig(
            tts_urdu_voice="ur-PK-AsadNeural"
        )
        with mock.patch(
            "afnan_ai.agent._speech"
        ) as speech_mock:
            agent.speak("افنان")
        speech_mock.speak.assert_called_once()
        _, kwargs = speech_mock.speak.call_args
        self.assertEqual(
            kwargs.get("urdu_voice"), "ur-PK-AsadNeural"
        )

    def test_no_detector_without_model(self):
        from afnan_ai.wakeword import create_detector

        # Explicit None → no detector (cloud fallback).
        self.assertIsNone(create_detector(None))

    def test_bundled_model_detector(self):
        """Real model: positive clip wakes, noise does not.

        Skipped when numpy is unavailable.
        """
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        import os

        from afnan_ai.wakeword import NumpyWakeWordDetector

        model = os.path.join(
            os.path.dirname(__file__), "..", "afnan_ai",
            "wakeword_models", "afnan.json",
        )
        model = os.path.abspath(model)
        if not os.path.exists(model):
            self.skipTest("bundled model not present")
        det = NumpyWakeWordDetector(model)
        self.assertEqual(det.window_samples, 16000)
        # silence → 0.0 via energy gate
        self.assertEqual(
            det.score(b"\x00" * 2560), 0.0
        )
        # bad format rejected
        import tempfile

        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as fh:
            fh.write('{"format": "nope"}')
            bad = fh.name
        with self.assertRaises(RuntimeError):
            NumpyWakeWordDetector(bad)
        os.unlink(bad)

    def test_detector_status_numpy(self):
        from afnan_ai.wakeword import detector_status

        status = detector_status("/tmp/afnan.json")
        self.assertTrue(status["model_configured"])
        self.assertFalse(status["model_exists"])
        self.assertIn("numpy_installed", status)

    def test_wait_for_wake_uses_fake_detector(self):
        agent = self._make_agent()
        agent.config = AgentConfig(wakeword_threshold=0.5)
        det = FakeWakeWordDetector([0.9])

        class FakeStream:
            def read(self, n):
                return b"\x00" * (n * 2)

        class FakeMic:
            def __init__(self, *a, **k):
                self.stream = FakeStream()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class FakeRecognizer:
            def adjust_for_ambient_noise(
                self, source, duration=0.5
            ):
                pass

        agent.recognizer = FakeRecognizer()
        with mock.patch(
            "afnan_ai.agent.sr"
        ) as sr_mock:
            sr_mock.Microphone.side_effect = FakeMic
            woke = agent._wait_for_wake_local(det, timeout_s=5)
        self.assertTrue(woke)
        self.assertGreater(det.calls, 0)


class ConversationModeTests(unittest.TestCase):
    def _make_agent(self):
        from afnan_ai.agent import AfnanAgent
        from afnan_ai.platform.base import PlatformAdapter

        class FakeAdapter(PlatformAdapter):
            name = "linux"

            def speak_system(self, text):
                pass

            def open_path(self, path):
                pass

            def launch_app(self, app_key):
                return True

            def find_folder(self, foldername):
                return None

        agent = AfnanAgent(adapter=FakeAdapter())
        agent.speak = lambda text: None
        return agent

    def test_conversation_defaults(self):
        config = AgentConfig()
        self.assertTrue(config.conversation_mode)
        self.assertEqual(config.conversation_timeout, 60.0)
        self.assertTrue(config.stream_responses)

    def test_conversation_env_overrides(self):
        env = {
            "AFNAN_CONVERSATION_MODE": "false",
            "AFNAN_CONVERSATION_TIMEOUT": "30",
            "AFNAN_STREAM_RESPONSES": "0",
        }
        with mock.patch.dict(os.environ, env):
            config = AgentConfig.from_env()
        self.assertFalse(config.conversation_mode)
        self.assertEqual(config.conversation_timeout, 30.0)
        self.assertFalse(config.stream_responses)

    def test_conversation_bad_values_fall_back(self):
        env = {
            "AFNAN_CONVERSATION_MODE": "maybe",
            "AFNAN_CONVERSATION_TIMEOUT": "nope",
        }
        with mock.patch.dict(os.environ, env):
            config = AgentConfig.from_env()
        # "maybe" is not a truthy token -> False is the parsed
        # value; invalid timeout falls back to the default.
        self.assertFalse(config.conversation_mode)
        self.assertEqual(config.conversation_timeout, 60.0)

    def test_goodbye_ends_conversation(self):
        agent = self._make_agent()
        handled = []
        agent.process_command = lambda cmd: handled.append(cmd)
        spoken = []
        agent.speak = spoken.append
        with mock.patch.object(
            agent, "listen_command", side_effect=["kya haal hai", "khuda hafiz"]
        ):
            agent._conversation_loop()
        self.assertEqual(handled, ["kya haal hai"])
        self.assertIn("Theek hai boss", spoken)

    def test_silence_timeout_ends_conversation(self):
        agent = self._make_agent()
        agent.config = AgentConfig(conversation_timeout=0)
        with mock.patch.object(
            agent, "listen_command", side_effect=AssertionError("no listen")
        ):
            agent._conversation_loop()  # returns immediately

    def test_empty_listens_keep_conversation_alive(self):
        agent = self._make_agent()
        agent.config = AgentConfig(conversation_timeout=60)
        handled = []
        agent.process_command = lambda cmd: handled.append(cmd)
        calls = {"n": 0}

        def fake_listen(timeout=7, phrase_time=8):
            calls["n"] += 1
            if calls["n"] < 3:
                return ""
            return "alvida"

        with mock.patch.object(agent, "listen_command", fake_listen):
            agent._conversation_loop()
        self.assertEqual(handled, [])
        self.assertEqual(calls["n"], 3)


class SentenceSplitTests(unittest.TestCase):
    def test_splits_complete_sentences(self):
        from afnan_ai.agent import _split_complete_sentences

        complete, rest = _split_complete_sentences(
            "Hello boss. How are you?"
        )
        self.assertEqual(complete, ["Hello boss."])
        self.assertEqual(rest, "How are you?")

    def test_multiple_sentences(self):
        from afnan_ai.agent import _split_complete_sentences

        complete, rest = _split_complete_sentences("One. Two. Three")
        self.assertEqual(complete, ["One.", "Two."])
        self.assertEqual(rest, "Three")

    def test_urdu_full_stop(self):
        from afnan_ai.agent import _split_complete_sentences

        complete, rest = _split_complete_sentences(
            "آپ کیسے ہیں۔ میں ٹھیک ہوں"
        )
        self.assertEqual(complete, ["آپ کیسے ہیں۔"])
        self.assertEqual(rest, "میں ٹھیک ہوں")

    def test_no_terminator_keeps_buffer(self):
        from afnan_ai.agent import _split_complete_sentences

        complete, rest = _split_complete_sentences("hello there")
        self.assertEqual(complete, [])
        self.assertEqual(rest, "hello there")

    def test_empty_buffer(self):
        from afnan_ai.agent import _split_complete_sentences

        complete, rest = _split_complete_sentences("   ")
        self.assertEqual(complete, [])
        self.assertEqual(rest, "   ")


class StreamingSpeakTests(unittest.TestCase):
    def _make_agent(self, provider):
        from afnan_ai.agent import AfnanAgent
        from afnan_ai.platform.base import PlatformAdapter

        class FakeAdapter(PlatformAdapter):
            name = "linux"

            def speak_system(self, text):
                pass

            def open_path(self, path):
                pass

            def launch_app(self, app_key):
                return True

            def find_folder(self, foldername):
                return None

        agent = AfnanAgent(adapter=FakeAdapter(), llm_provider=provider)
        spoken = []
        agent.speak = spoken.append
        return agent, spoken

    def test_speaks_sentence_by_sentence(self):
        from afnan_ai.llm.base import LLMProvider

        class StreamProvider(LLMProvider):
            name = "stream"

            def chat(self, messages):
                return "Hello boss. How are you?"

            def chat_stream(self, messages):
                yield "Hello boss. "
                yield "How are "
                yield "you?"

        agent, spoken = self._make_agent(StreamProvider())
        agent._speak_streaming("hi")
        self.assertEqual(spoken, ["Hello boss.", "How are you?"])

    def test_urdu_streaming(self):
        from afnan_ai.llm.base import LLMProvider

        class StreamProvider(LLMProvider):
            name = "stream"

            def chat(self, messages):
                return "x"

            def chat_stream(self, messages):
                yield "آپ کیسے ہیں۔ "
                yield "میں ٹھیک ہوں"

        agent, spoken = self._make_agent(StreamProvider())
        agent._speak_streaming("hi")
        self.assertEqual(spoken, ["آپ کیسے ہیں۔", "میں ٹھیک ہوں"])

    def test_stream_error_speaks_sorry(self):
        from afnan_ai.llm import LLMConnectionError
        from afnan_ai.llm.base import LLMProvider

        class FailProvider(LLMProvider):
            name = "fail"
            display_name = "Ollama"

            def chat(self, messages):
                raise LLMConnectionError("down")

            def chat_stream(self, messages):
                raise LLMConnectionError("down")
                yield  # pragma: no cover - generator

        agent, spoken = self._make_agent(FailProvider())
        agent._speak_streaming("hi")
        self.assertEqual(len(spoken), 1)
        self.assertIn("Sorry boss, AI is not responding", spoken[0])

    def test_fallback_uses_streaming_when_enabled(self):
        from afnan_ai.llm.base import LLMProvider

        class StreamProvider(LLMProvider):
            name = "stream"

            def chat(self, messages):
                return "nope"

            def chat_stream(self, messages):
                yield "streamed. ok"

        agent, spoken = self._make_agent(StreamProvider())
        agent.config = AgentConfig(stream_responses=True)
        agent._state_begin = lambda command: None
        agent._state_succeed = lambda result=None: None
        agent._legacy_chat_fallback("hello")
        # "Thinking boss" + streamed sentences
        self.assertEqual(spoken[0], "Thinking boss")
        self.assertIn("streamed.", spoken[1:])

    def test_fallback_blocking_when_streaming_disabled(self):
        from afnan_ai.llm.base import LLMProvider

        class StreamProvider(LLMProvider):
            name = "stream"

            def chat(self, messages):
                return "blocking reply"

            def chat_stream(self, messages):
                yield "should not be used"

        agent, spoken = self._make_agent(StreamProvider())
        agent.config = AgentConfig(stream_responses=False)
        agent._state_begin = lambda command: None
        agent._state_succeed = lambda result=None: None
        agent._legacy_chat_fallback("hello")
        self.assertEqual(spoken, ["Thinking boss", "blocking reply"])


if __name__ == "__main__":
    unittest.main()

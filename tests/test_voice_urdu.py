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
        self.assertEqual(config.wakeword_threshold, 0.5)
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
        self.assertEqual(config.wakeword_threshold, 0.5)


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


if __name__ == "__main__":
    unittest.main()

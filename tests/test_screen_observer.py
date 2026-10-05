"""ScreenObserver tests: screenshot capture, dimensions,
visual element detection with confidence, low-confidence
action gating, screen-change detection, browser DOM/visual
fusion + fallback, AgentState recording, and failure cases.
All synthetic PNGs and fake captures — no real screen needed.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai.agent import AfnanAgent
from afnan_ai.llm.base import LLMProvider
from afnan_ai.orchestrator import OrchestrationStatus
from afnan_ai.screen import (
    FunctionCaptureSource,
    ScreenErrorCode,
    ScreenException,
    ScreenObserver,
    create_screen_tools,
)
from afnan_ai.screen.detector import RawDetection, RegionDetector
from afnan_ai.screen.image import decode_png, encode_png
from afnan_ai.screen.base import VisualRegion
from afnan_ai.tools.registry import ToolRegistry


def make_png(rects, w=160, h=120, bg=(18, 18, 22)):
    rows = [[bg] * w for _ in range(h)]
    for (x, y, rw, rh, color) in rects:
        for yy in range(y, y + rh):
            for xx in range(x, x + rw):
                rows[yy][xx] = color
    return encode_png(w, h, rows)


BRIGHT = make_png([
    (20, 20, 60, 40, (235, 235, 235)),   # bright panel
    (100, 75, 44, 16, (210, 210, 210)),  # button-like
])
FAINT = make_png([
    (30, 30, 50, 30, (66, 66, 70)),      # faint, low contrast
    (100, 20, 44, 16, (235, 235, 235)),  # bright
])
BLANK = make_png([])


def png_source(data):
    return FunctionCaptureSource(lambda: data)


# ----------------------------------------------------------------------
# PNG codec
# ----------------------------------------------------------------------

class TestPngCodec(unittest.TestCase):
    def test_roundtrip(self):
        data = make_png([(5, 5, 10, 10, (200, 30, 40))], w=40, h=30)
        image = decode_png(data)
        self.assertEqual((image.width, image.height), (40, 30))
        self.assertEqual(image.pixel(7, 7), (200, 30, 40))
        self.assertEqual(image.pixel(0, 0), (18, 18, 22))

    def test_invalid_image(self):
        with self.assertRaises(ScreenException) as ctx:
            decode_png(b"this is not a png")
        self.assertEqual(ctx.exception.code, ScreenErrorCode.INVALID_IMAGE)


# ----------------------------------------------------------------------
# Capture + observation
# ----------------------------------------------------------------------

class TestCaptureAndObserve(unittest.TestCase):
    def test_dimensions_and_detection(self):
        observer = ScreenObserver(png_source(BRIGHT))
        obs = observer.observe()
        self.assertEqual((obs.width, obs.height), (160, 120))
        self.assertEqual(obs.source, "desktop")
        self.assertFalse(obs.screen_changed)
        self.assertGreaterEqual(len(obs.elements), 2)
        top = max(obs.elements, key=lambda e: e.confidence)
        self.assertGreaterEqual(top.confidence, 0.8)
        region = top.region
        self.assertIsNotNone(region)
        self.assertLessEqual(abs(region.x - 20), 2)
        self.assertLessEqual(abs(region.y - 20), 2)
        self.assertTrue(all(e.ref.startswith("vis_") for e in obs.elements))
        self.assertTrue(all(e.source == "visual" for e in obs.elements))

    def test_no_source_is_structured_error(self):
        observer = ScreenObserver()
        with self.assertRaises(ScreenException) as ctx:
            observer.observe()
        self.assertEqual(
            ctx.exception.code, ScreenErrorCode.SCREEN_UNAVAILABLE
        )

    def test_capture_failures_are_structured(self):
        def boom():
            raise RuntimeError("no display")
        observer = ScreenObserver(FunctionCaptureSource(boom))
        with self.assertRaises(ScreenException) as ctx:
            observer.observe()
        self.assertEqual(ctx.exception.code, ScreenErrorCode.CAPTURE_FAILED)

        observer = ScreenObserver(FunctionCaptureSource(lambda: None))
        with self.assertRaises(ScreenException) as ctx:
            observer.observe()
        self.assertEqual(
            ctx.exception.code, ScreenErrorCode.SCREEN_UNAVAILABLE
        )

        observer = ScreenObserver(png_source(b"junk-bytes"))
        with self.assertRaises(ScreenException) as ctx:
            observer.observe()
        self.assertEqual(ctx.exception.code, ScreenErrorCode.INVALID_IMAGE)

    def test_confidence_reflects_contrast(self):
        bright_obs = ScreenObserver(png_source(BRIGHT)).observe()
        faint_obs = ScreenObserver(png_source(FAINT)).observe()
        bright_best = max(e.confidence for e in bright_obs.elements)
        faint = [
            e for e in faint_obs.elements
            if e.region and abs(e.region.x - 30) <= 3
        ]
        self.assertTrue(faint)
        self.assertLess(faint[0].confidence, bright_best)

    def test_screenshot_saved_on_request(self):
        observer = ScreenObserver(png_source(BRIGHT))
        with tempfile.TemporaryDirectory() as tmp:
            obs = observer.observe(save_dir=tmp, filename="shot.png")
            path = Path(obs.screenshot_path)
            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), BRIGHT)

    def test_detector_failure_is_structured(self):
        class BrokenDetector:
            def detect(self, image):
                raise RuntimeError("vision exploded")
        observer = ScreenObserver(png_source(BRIGHT), detector=BrokenDetector())
        with self.assertRaises(ScreenException) as ctx:
            observer.observe()
        self.assertEqual(
            ctx.exception.code, ScreenErrorCode.DETECTION_FAILED
        )


# ----------------------------------------------------------------------
# Screen change + refs
# ----------------------------------------------------------------------

class TestScreenChange(unittest.TestCase):
    def test_change_detection_and_ref_lifecycle(self):
        frames = [BRIGHT]
        observer = ScreenObserver(
            FunctionCaptureSource(lambda: frames[0])
        )
        first = observer.observe()
        ref = first.elements[0].ref
        again = observer.observe()
        self.assertFalse(again.screen_changed)
        # refs survive an unchanged re-observe
        observer.get_element(ref)

        frames[0] = FAINT
        changed = observer.observe()
        self.assertTrue(changed.screen_changed)
        # refs from before the change are stale
        with self.assertRaises(ScreenException) as ctx:
            observer.get_element(ref)
        self.assertEqual(
            ctx.exception.code, ScreenErrorCode.ELEMENT_NOT_FOUND
        )

    def test_find_elements_filters(self):
        observer = ScreenObserver(png_source(BRIGHT))
        observer.observe()
        strong = observer.find_elements(min_confidence=0.8)
        self.assertTrue(strong)
        self.assertTrue(all(e.confidence >= 0.8 for e in strong))
        self.assertEqual(observer.find_elements(kind="nope"), [])


# ----------------------------------------------------------------------
# Confidence gating
# ----------------------------------------------------------------------

class _StubDetector:
    def __init__(self, confidences):
        self._confidences = confidences

    def detect(self, image):
        return [
            RawDetection(
                VisualRegion(5 + i * 10, 5, 8, 8), "region", c
            )
            for i, c in enumerate(self._confidences)
        ]


class TestConfidenceGating(unittest.TestCase):
    def test_tiers(self):
        observer = ScreenObserver(
            png_source(BLANK), detector=_StubDetector([0.95, 0.65, 0.30])
        )
        obs = observer.observe()
        by_conf = {round(e.confidence, 2): e for e in obs.elements}
        ok = observer.assess_action(by_conf[0.95].ref)
        self.assertEqual(ok.tier, "ok")
        self.assertTrue(ok.allowed_automatically)
        verify = observer.assess_action(by_conf[0.65].ref)
        self.assertEqual(verify.tier, "verify")
        self.assertFalse(verify.allowed_automatically)
        approval = observer.assess_action(by_conf[0.30].ref)
        self.assertEqual(approval.tier, "approval_required")
        self.assertFalse(approval.allowed_automatically)
        self.assertIn("approval", approval.reason)

    def test_assess_unknown_ref(self):
        observer = ScreenObserver(png_source(BLANK))
        observer.observe()
        with self.assertRaises(ScreenException):
            observer.assess_action("vis_999")


# ----------------------------------------------------------------------
# Browser fusion + visual fallback
# ----------------------------------------------------------------------

class _StubBrowserController:
    """Duck-typed BrowserController stand-in for fusion tests."""

    def __init__(self, png, dom_elements=None, dom_fails=False):
        self._png = png
        self._dom = dom_elements or []
        self._dom_fails = dom_fails

    def screenshot(self, output_dir=None, tab_id=None, **kwargs):
        path = Path(output_dir) / "page.png"
        path.write_bytes(self._png)
        return {"path": str(path)}

    def observe(self):
        if self._dom_fails:
            raise RuntimeError("DOM unavailable")
        return {"elements": self._dom}


class TestBrowserFusion(unittest.TestCase):
    def test_dom_elements_win_and_visuals_fill_in(self):
        dom = [
            {"ref": "el_1", "tag": "button", "text": "Buy"},
            {"ref": "el_2", "tag": "input", "text": ""},
        ]
        stub = _StubBrowserController(BRIGHT, dom_elements=dom)
        observer = ScreenObserver(browser_controller=stub)
        obs = observer.observe(origin="browser")
        self.assertTrue(obs.dom_available)
        self.assertEqual(len(obs.dom_elements), 2)
        self.assertTrue(all(e.confidence == 1.0 for e in obs.dom_elements))
        self.assertTrue(obs.visual_elements)  # pixels still detected
        buy = [e for e in obs.dom_elements if e.label == "Buy"][0]
        self.assertEqual(buy.extra["dom_ref"], "el_1")
        self.assertEqual(observer.assess_action(buy.ref).tier, "ok")

    def test_visual_fallback_when_dom_unavailable(self):
        stub = _StubBrowserController(BRIGHT, dom_fails=True)
        observer = ScreenObserver(browser_controller=stub)
        obs = observer.observe(origin="browser")
        self.assertFalse(obs.dom_available)
        self.assertEqual(obs.dom_elements, [])
        self.assertGreaterEqual(len(obs.visual_elements), 2)

    def test_browser_origin_without_controller(self):
        observer = ScreenObserver(png_source(BRIGHT))
        with self.assertRaises(ScreenException) as ctx:
            observer.observe(origin="browser")
        self.assertEqual(
            ctx.exception.code, ScreenErrorCode.SCREEN_UNAVAILABLE
        )


# ----------------------------------------------------------------------
# Tools + agent integration
# ----------------------------------------------------------------------

class _QueueLLM(LLMProvider):
    name = "queue"
    display_name = "Queue"
    model = "queue-1"

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages):
        if not self.replies:
            raise AssertionError("QueueLLM ran out of replies")
        return self.replies.pop(0)


class TestToolsAndAgent(unittest.TestCase):
    def test_tool_names_and_structured_output(self):
        observer = ScreenObserver(png_source(BRIGHT))
        names = sorted(t.name for t in create_screen_tools(observer))
        self.assertEqual(
            names,
            ["screen_assess_action", "screen_find_elements", "screen_observe"],
        )
        self.assertNotIn("screen_click", names)  # observer never clicks

        registry = ToolRegistry()
        for tool in create_screen_tools(observer):
            registry.register(tool)
        result = registry.execute("screen_observe", {})
        self.assertTrue(result.success, result.error)
        blob = json.dumps(result.output)
        self.assertNotIn("image_bytes", blob)
        self.assertIn("screen_observation", blob)
        self.assertEqual(result.output["width"], 160)
        self.assertIn("observation", result.output)

        found = registry.execute(
            "screen_find_elements", {"min_confidence": 0.8}
        )
        self.assertTrue(found.success)
        ref = found.output["elements"][0]["ref"]
        assessed = registry.execute("screen_assess_action", {"ref": ref})
        self.assertEqual(assessed.output["tier"], "ok")

        bad = registry.execute("screen_assess_action", {"ref": "vis_x"})
        self.assertFalse(bad.success)
        self.assertEqual(
            bad.error.details["screen_error"]["code"], "element_not_found"
        )

    def test_no_capture_tool_fails_structured(self):
        registry = ToolRegistry()
        for tool in create_screen_tools(ScreenObserver()):
            registry.register(tool)
        result = registry.execute("screen_observe", {})
        self.assertFalse(result.success)
        self.assertEqual(
            result.error.details["screen_error"]["code"],
            "screen_unavailable",
        )

    def test_agent_records_screen_observation_in_state(self):
        plan = json.dumps({
            "goal": "Look at the screen",
            "steps": [{
                "step_id": "s1",
                "description": "Observe the screen",
                "tool_name": "screen_observe",
                "arguments": {},
                "expected_result": "Screen observation with elements",
            }],
        })
        observer = ScreenObserver(png_source(BRIGHT))
        agent = AfnanAgent(
            llm_provider=_QueueLLM([plan]), screen_observer=observer
        )
        self.assertIn(
            "screen_observe", [t["name"] for t in agent.list_tools()]
        )
        result = agent.orchestrator.run("Look at the screen")
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        state_blob = result.state.to_json()
        self.assertIn("screen_observation", state_blob)
        self.assertIn("confidence", state_blob)
        # the planner-facing state carries structure, not pixels
        self.assertNotIn("image_bytes", state_blob)

    def test_screen_tools_can_be_disabled(self):
        agent = AfnanAgent(
            llm_provider=_QueueLLM(["{}"]), enable_screen_tools=False,
            enable_browser_tools=False,
        )
        self.assertIsNone(agent.screen_observer)
        self.assertNotIn(
            "screen_observe", [t["name"] for t in agent.list_tools()]
        )

    def test_verifier_hook_routes_screen_steps(self):
        observer = ScreenObserver(png_source(BRIGHT))
        agent = AfnanAgent(
            llm_provider=_QueueLLM(["{}"]), screen_observer=observer,
            enable_browser_tools=False,
        )

        class _Step:
            def __init__(self, name):
                self.tool_name = name

        provider = agent.verifier.observation_provider
        seen = provider(_Step("screen_observe"))
        self.assertIsNotNone(seen)
        self.assertEqual(seen["kind"], "screen_observation")
        self.assertIsNone(provider(_Step("open_url")))

    def test_screen_package_has_no_platform_coupling(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / "afnan_ai" / "screen").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("import pyautogui", source, path.name)
            self.assertNotIn("import playwright", source, path.name)
            self.assertNotIn("sys.platform", source, path.name)
            self.assertNotIn("startfile", source, path.name)


if __name__ == "__main__":
    unittest.main()

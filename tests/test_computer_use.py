"""Computer Use layer tests: platform backends, window/app
management, semantic location, validated actions,
wrong-target prevention, popup handling, crash recovery,
file workflows, approval protection, and an agent-level
browser+desktop combined workflow."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from afnan_ai.computer.backend import ComputerBackend
from afnan_ai.computer.command_backend import CommandComputerBackend
from afnan_ai.computer.controller import ComputerController
from afnan_ai.computer.errors import ComputerError
from afnan_ai.computer.files import FileService
from afnan_ai.computer.locator import rank_candidates
from afnan_ai.computer.models import ComputerElement, WindowInfo
from afnan_ai.computer.policy import ComputerApprovalGate

from test_browser_advanced import plan_json, step
from test_browser_workflow import make_agent


class FakeComputerBackend(ComputerBackend):
    name = "fake"

    def __init__(self):
        self.active_id = "w1"
        self.clicks: list[tuple] = []
        self.typed: list[str] = []
        self.keys: list[str] = []
        self.hotkeys: list[list[str]] = []
        self.scrolls: list[tuple] = []
        self.launched: list[str] = []
        self.closed_apps: list[str] = []
        self.focused: list[str] = []
        self.crashed: set[str] = set()
        self._window_apps = {"w1": "Editor", "w2": "Settings"}
        self._elements = {
            "w1": [
                {"role": "button", "name": "Save",
                 "bbox": {"x": 10, "y": 10, "width": 80, "height": 24}},
                {"role": "textbox", "name": "Document",
                 "editable": True,
                 "bbox": {"x": 10, "y": 60, "width": 300, "height": 100}},
                {"role": "textbox", "name": "Password",
                 "editable": True, "password_field": True,
                 "bbox": {"x": 10, "y": 180, "width": 200, "height": 24}},
                {"role": "button", "name": "Export...",
                 "bbox": {"x": 100, "y": 10, "width": 90, "height": 24}},
            ],
            "w2": [
                {"role": "button", "name": "Apply",
                 "bbox": {"x": 10, "y": 10, "width": 80, "height": 24}},
            ],
        }

    def screen_size(self):
        return (1280, 800)

    def cursor_position(self):
        return (100, 100)

    def capture_screenshot(self):
        return b"\x89PNG-fake"

    def _windows(self):
        result = []
        for window_id, app in self._window_apps.items():
            if app in self.crashed:
                continue
            title = (
                "Welcome - Editor" if window_id == "w1"
                else "Settings" if window_id == "w2"
                else app
            )
            result.append(WindowInfo(
                window_id=window_id, title=title,
                app=app, active=window_id == self.active_id,
            ))
        return result

    def list_windows(self):
        return self._windows()

    def accessibility_elements(self, window_id=None):
        window_id = window_id or self.active_id
        app = self._window_apps.get(window_id, "")
        if app in self.crashed:
            raise ComputerError(
                "application_crashed", f"{app} stopped responding"
            )
        return [
            dict(item, window_id=window_id, app=app)
            for item in self._elements.get(window_id, [])
        ]

    def mouse_click(self, x, y, *, button="left", click_count=1):
        self.clicks.append((x, y, button, click_count))
        # Clicking Export... opens a popup window.
        if (x, y) == (145, 22):
            self._window_apps["w3"] = "Editor"
            self._elements["w3"] = []
            self.active_id = "w3"

    def type_text(self, text):
        self.typed.append(text)

    def key_press(self, key):
        self.keys.append(key)

    def hotkey(self, keys):
        self.hotkeys.append(list(keys))

    def scroll(self, dx, dy):
        self.scrolls.append((dx, dy))

    def mouse_move(self, x, y):
        pass

    def mouse_drag(self, x1, y1, x2, y2):
        self.scrolls.append(("drag", x1, y1, x2, y2))

    def focus_window(self, window_id):
        self.focused.append(window_id)
        self.active_id = window_id

    def close_window(self, window_id):
        self._window_apps.pop(window_id, None)

    def launch_application(self, name_or_path):
        self.launched.append(str(name_or_path))
        window_id = f"w_{name_or_path}"
        self._window_apps[window_id] = str(name_or_path)
        self._elements.setdefault(window_id, [])
        self.active_id = window_id
        from afnan_ai.computer.models import AppInfo
        return AppInfo(name=str(name_or_path), running=True)

    def close_application(self, name):
        self.closed_apps.append(name)
        for window_id, app in list(self._window_apps.items()):
            if app == name:
                del self._window_apps[window_id]

    def list_applications(self):
        from afnan_ai.computer.models import AppInfo
        return [AppInfo(name="Editor"), AppInfo(name="Files")]


def make_controller(approver=None, observer=None):
    backend = FakeComputerBackend()
    controller = ComputerController(
        backend, screen_observer=observer,
        gate=ComputerApprovalGate(approver),
    )
    return controller, backend


class TestCommandBackends(unittest.TestCase):
    def backend(self, system, outputs=None):
        calls = []
        outputs = outputs or {}

        def runner(argv):
            calls.append(argv)
            for marker, out in outputs.items():
                if marker in " ".join(argv):
                    return out
            return ""

        spawns = []
        backend = CommandComputerBackend(
            system,
            runner=runner,
            spawner=lambda argv: spawns.append(argv),
        )
        return backend, calls, spawns

    def test_linux_click_command(self):
        backend, calls, _ = self.backend("linux")
        backend.mouse_click(10, 20)
        self.assertEqual(calls[0][0], "xdotool")
        self.assertIn("click", calls[0])

    def test_linux_windows_parsed(self):
        backend, _, _ = self.backend(
            "linux",
            {"wmctrl": "0x03e00003  0 host My Title\n",
             "getactivewindow": "65011715"},
        )
        windows = backend.list_windows()
        self.assertEqual(windows[0].title, "My Title")
        self.assertTrue(windows[0].active)

    def test_macos_type_uses_osascript(self):
        backend, calls, _ = self.backend("darwin")
        backend.type_text("hello")
        self.assertEqual(calls[0][0], "osascript")
        self.assertIn("keystroke", calls[0][-1])

    def test_windows_window_list_parsed(self):
        backend, calls, _ = self.backend(
            "windows", {"Get-Process": "123|notepad|Untitled\n"}
        )
        windows = backend.list_windows()
        self.assertEqual(calls[0][0], "powershell")
        self.assertEqual(windows[0].app, "notepad")

    def test_missing_tool_is_structured(self):
        def runner(argv):
            raise ComputerError(
                "backend_unavailable", f"missing {argv[0]}"
            )

        backend = CommandComputerBackend("linux", runner=runner)
        with self.assertRaises(ComputerError):
            backend.mouse_move(1, 2)


class TestControllerObservation(unittest.TestCase):
    def test_observe_reports_windows_active_and_elements(self):
        controller, _ = make_controller()
        payload = controller.observe(include_visual=False)
        self.assertEqual(payload["active_app"], "Editor")
        self.assertEqual(len(payload["windows"]), 2)
        names = [e["name"] for e in payload["elements"]]
        self.assertIn("Save", names)
        self.assertIn("Welcome - Editor", names)  # window target
        self.assertNotIn("PNG", str(payload))  # no raw image

    def test_locate_finds_save_button_high_confidence(self):
        controller, _ = make_controller()
        controller.observe(include_visual=False)
        found = controller.locate("Save button")
        top = found["candidates"][0]
        self.assertEqual(top["name"], "Save")
        self.assertEqual(top["tier"], "high")
        self.assertTrue(top["element_id"])

    def test_low_confidence_candidate_is_unactionable(self):
        element = ComputerElement(
            element_id="el_9", role="region", name="Mystery blob",
            source="visual", confidence=0.2,
        )
        ranked = rank_candidates(
            "mystery control panel", [element]
        )
        self.assertEqual(ranked[0]["tier"], "low")
        self.assertEqual(ranked[0]["element_id"], "")

    def test_application_launch_is_tracked(self):
        controller, backend = make_controller()
        result = controller.open_application("Files", wait_s=0.5)
        self.assertIn("Files", backend.launched)
        self.assertIsNotNone(result["window"])


class TestValidatedActions(unittest.TestCase):
    def test_click_types_and_verifies(self):
        controller, backend = make_controller(
            approver=lambda req: True
        )
        controller.observe(include_visual=False)
        save = controller.locate("Save button")["candidates"][0]
        result = controller.act(
            "click", element_id=save["element_id"],
            expect_text="Save",
        )
        self.assertTrue(result["success"])
        self.assertTrue(result["verified"])
        self.assertEqual(len(backend.clicks), 1)

    def test_popup_window_is_reported(self):
        controller, backend = make_controller(
            approver=lambda req: True
        )
        controller.observe(include_visual=False)
        export = controller.locate("Export")["candidates"][0]
        result = controller.act(
            "click", element_id=export["element_id"]
        )
        self.assertTrue(result["screen_changed"])
        self.assertIn(
            "new_windows", result["after_observation"]
        )

    def test_wrong_window_action_is_refused(self):
        controller, backend = make_controller(
            approver=lambda req: True
        )
        controller.observe(include_visual=False)
        # "Apply" lives in the inactive Settings window.
        apply_btn = next(
            e for e in controller._observation.elements
            if e.name == "Apply"
        )
        with self.assertRaises(ComputerError) as caught:
            controller.act("click", element_id=apply_btn.element_id)
        self.assertEqual(caught.exception.code, "window_mismatch")
        self.assertEqual(backend.clicks, [])
        controller.focus_window("w2")
        result = controller.act(
            "click", element_id=apply_btn.element_id
        )
        self.assertTrue(result["success"])

    def test_stale_element_after_crash(self):
        controller, backend = make_controller(
            approver=lambda req: True
        )
        controller.observe(include_visual=False)
        save = controller.locate("Save button")["candidates"][0]
        backend.crashed.add("Editor")
        payload = controller.observe(include_visual=False)
        names = [e["name"] for e in payload["elements"]]
        self.assertNotIn("Save", names)
        with self.assertRaises(ComputerError) as caught:
            controller.act("click", element_id=save["element_id"])
        self.assertEqual(caught.exception.code, "stale_element")
        backend.crashed.clear()
        controller.observe(include_visual=False)
        save = controller.locate("Save button")["candidates"][0]
        result = controller.act(
            "click", element_id=save["element_id"]
        )
        self.assertTrue(result["success"])

    def test_coordinate_only_click_needs_approval(self):
        controller, backend = make_controller()
        controller.observe(include_visual=False)
        with self.assertRaises(ComputerError) as caught:
            controller.act("click", x=5, y=5)
        self.assertEqual(caught.exception.code, "approval_required")
        self.assertEqual(backend.clicks, [])

    def test_key_and_hotkey_actions(self):
        controller, backend = make_controller(
            approver=lambda req: True
        )
        controller.observe(include_visual=False)
        controller.act("key_press", key="Return")
        controller.act("hotkey", keys=["ctrl", "s"])
        self.assertEqual(backend.keys, ["Return"])
        self.assertEqual(backend.hotkeys, [["ctrl", "s"]])

    def test_destructive_hotkey_blocked_without_approver(self):
        controller, backend = make_controller()
        controller.observe(include_visual=False)
        with self.assertRaises(ComputerError) as caught:
            controller.act("hotkey", keys=["alt", "f4"])
        self.assertEqual(caught.exception.code, "approval_required")
        self.assertEqual(backend.hotkeys, [])


class TestSensitiveProtection(unittest.TestCase):
    def test_password_typing_needs_approval_and_stays_secret(self):
        controller, backend = make_controller()
        controller.observe(include_visual=False)
        password = next(
            e for e in controller._observation.elements
            if e.password_field
        )
        with self.assertRaises(ComputerError) as caught:
            controller.act(
                "type", element_id=password.element_id,
                text="hunter2",
            )
        self.assertEqual(caught.exception.code, "approval_required")
        self.assertEqual(backend.typed, [])

    def test_password_typing_with_approver_never_echoes(self):
        controller, backend = make_controller(
            approver=lambda req: True
        )
        controller.observe(include_visual=False)
        password = next(
            e for e in controller._observation.elements
            if e.password_field
        )
        result = controller.act(
            "type", element_id=password.element_id,
            text="hunter2",
        )
        self.assertEqual(backend.typed, ["hunter2"])
        self.assertEqual(result["typed"], "***")
        self.assertNotIn("hunter2", str(result))

    def test_close_application_requires_approval(self):
        controller, backend = make_controller()
        with self.assertRaises(ComputerError) as caught:
            controller.close_application("Editor")
        self.assertEqual(caught.exception.code, "approval_required")
        self.assertEqual(backend.closed_apps, [])
        controller2, backend2 = make_controller(
            approver=lambda req: True
        )
        controller2.close_application("Editor")
        self.assertEqual(backend2.closed_apps, ["Editor"])

    def test_state_restart_recovers_observation(self):
        controller, backend = make_controller()
        controller.observe(include_visual=False)
        # A new controller over the same backend (process
        # restart) sees the same desktop.
        restarted = ComputerController(backend)
        payload = restarted.observe(include_visual=False)
        self.assertEqual(payload["active_app"], "Editor")
        self.assertEqual(len(payload["windows"]), 2)


class TestFileService(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_file_workflow(self):
        downloads = os.path.join(self.root, "downloads")
        os.makedirs(downloads)
        report = os.path.join(downloads, "report.txt")
        with open(report, "w", encoding="utf-8") as handle:
            handle.write("data")
        service = FileService(
            gate=ComputerApprovalGate(lambda req: True),
            open_path=lambda path: None,
            downloads_dirs=[downloads],
        )
        found = service.find_downloads(max_age_s=600)
        self.assertEqual(
            found["downloads"][0]["name"], "report.txt"
        )
        out_dir = os.path.join(self.root, "out")
        service.create_folder(out_dir)
        copied = service.copy_file(report, out_dir)
        self.assertTrue(os.path.exists(copied["copied_to"]))
        renamed = service.rename_file(
            copied["copied_to"], "renamed.txt"
        )
        saved = service.save_text(
            os.path.join(out_dir, "result.txt"), "done"
        )
        self.assertEqual(saved["bytes"], 4)
        listing = service.list_dir(out_dir)
        names = [e["name"] for e in listing["entries"]]
        self.assertIn("renamed.txt", names)

    def test_move_blocked_without_approver(self):
        source = os.path.join(self.root, "a.txt")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write("x")
        service = FileService(
            downloads_dirs=[self.root]
        )
        with self.assertRaises(ComputerError) as caught:
            service.move_file(
                source, os.path.join(self.root, "b.txt")
            )
        self.assertEqual(caught.exception.code, "approval_required")
        self.assertTrue(os.path.exists(source))


class TestAgentCombinedWorkflow(unittest.TestCase):
    def test_agent_chooses_computer_tools_alongside_browser(self):
        desktop = FakeComputerBackend()
        steps = [
            step("c1", "computer_open_application",
                 {"name": "Editor"}, "Editor"),
            step("c2", "computer_type", {"text": "hello"},
                 "Document"),
        ]
        replies = [plan_json("Use the desktop", steps), '{"goal": "done", "complete": true, "steps": []}']
        agent, _browser_backend = make_agent(
            replies,
            computer_backend=desktop,
            computer_approver=lambda req: True,
            enable_browser_tools=False,
            enable_screen_tools=False,
        )
        # Browser tools exist in the same registry family when
        # enabled; here the planner picked desktop tools.
        self.assertIsNotNone(agent.tools.get_or_none("computer_click"))
        self.assertIsNotNone(agent.tools.get_or_none("file_list"))
        result = agent.run_agent_loop("Use the desktop")
        self.assertEqual(desktop.launched, ["Editor"])
        self.assertEqual(desktop.typed, ["hello"])
        from afnan_ai.orchestrator import OrchestrationStatus
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()

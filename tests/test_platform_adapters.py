import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afnan_ai.platform import get_adapter, detect_system_name
from afnan_ai.platform.windows import WindowsAdapter
from afnan_ai.platform.macos import MacOSAdapter
from afnan_ai.platform.linux import LinuxAdapter


class TestAdapterSelection(unittest.TestCase):
    def test_windows_selected(self):
        self.assertIsInstance(get_adapter("Windows"), WindowsAdapter)

    def test_macos_selected_from_darwin(self):
        self.assertIsInstance(get_adapter("Darwin"), MacOSAdapter)

    def test_linux_selected(self):
        self.assertIsInstance(get_adapter("Linux"), LinuxAdapter)

    def test_detect_normalises_names(self):
        self.assertEqual(detect_system_name("Windows"), "windows")
        self.assertEqual(detect_system_name("Darwin"), "macos")
        self.assertEqual(detect_system_name("Linux"), "linux")

    def test_auto_detection_returns_a_known_adapter(self):
        adapter = get_adapter()
        self.assertIn(adapter.name, {"windows", "macos", "linux"})


class TestWindowsAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = WindowsAdapter()

    def test_open_path_uses_startfile(self):
        with mock.patch("os.startfile", create=True) as startfile:
            self.adapter.open_path("C:\\Users\\test\\file.txt")
            startfile.assert_called_once()

    def test_launch_chrome_uses_cmd_start(self):
        with mock.patch("subprocess.run") as run:
            self.assertTrue(self.adapter.launch_app("chrome"))
            run.assert_called()
            self.assertIn("chrome", str(run.call_args))

    def test_safari_not_supported_on_windows(self):
        self.assertFalse(self.adapter.supports_app("safari"))
        self.assertFalse(self.adapter.launch_app("safari"))

    def test_speak_uses_powershell(self):
        # Maya-style: single blocking os.system() call to the
        # OS-native TTS.
        with mock.patch("os.system") as system:
            self.adapter.speak_system("hello")
            args = str(system.call_args)
            self.assertIn("powershell", args)
            self.assertIn("System.Speech", args)
            self.assertIn("hello", args)


class TestMacOSAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = MacOSAdapter()

    def test_open_path_uses_open(self):
        with mock.patch("subprocess.run") as run:
            self.adapter.open_path("/tmp/file.txt")
            run.assert_called_once_with(["open", "/tmp/file.txt"], check=False)

    def test_launch_safari_uses_open_a(self):
        with mock.patch("subprocess.run") as run:
            self.assertTrue(self.adapter.launch_app("safari"))
            run.assert_called_once_with(["open", "-a", "Safari"], check=False)

    def test_safari_supported_on_macos(self):
        self.assertTrue(self.adapter.supports_app("safari"))

    def test_speak_uses_say(self):
        with mock.patch("subprocess.run") as run:
            self.adapter.speak_system("hello")
            run.assert_called_once_with(["say", "hello"], check=False)


class TestLinuxAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = LinuxAdapter()

    def test_open_path_uses_xdg_open(self):
        with mock.patch("subprocess.run") as run:
            self.adapter.open_path("/tmp/file.txt")
            run.assert_called_once_with(["xdg-open", "/tmp/file.txt"], check=False)

    def test_launch_chrome_finds_executable(self):
        with mock.patch("shutil.which", return_value="/usr/bin/google-chrome"), \
             mock.patch("subprocess.run") as run:
            self.assertTrue(self.adapter.launch_app("chrome"))
            run.assert_called()

    def test_launch_chrome_missing_executable(self):
        with mock.patch("shutil.which", return_value=None):
            self.assertFalse(self.adapter.launch_app("chrome"))

    def test_safari_not_supported_on_linux(self):
        self.assertFalse(self.adapter.supports_app("safari"))


class TestKnownFoldersCrossPlatform(unittest.TestCase):
    def test_known_folder_lookup_uses_home(self):
        for adapter in (WindowsAdapter(), MacOSAdapter(), LinuxAdapter()):
            with mock.patch.object(Path, "home", return_value=Path("/tmp")):
                # /tmp/Downloads may not exist -> None is acceptable,
                # the point is it never raises and never uses an OS command
                result = adapter.known_folder_path("Downloads")
                self.assertIn(result, (None, Path("/tmp/Downloads")))


if __name__ == "__main__":
    unittest.main()

"""Runtime platform detection and adapter factory."""

from __future__ import annotations

import platform as _platform_module

from afnan_ai.platform.base import PlatformAdapter
from afnan_ai.platform.linux import LinuxAdapter
from afnan_ai.platform.macos import MacOSAdapter
from afnan_ai.platform.windows import WindowsAdapter

_SYSTEM_MAP = {
    "windows": WindowsAdapter,
    "darwin": MacOSAdapter,
    "macos": MacOSAdapter,
    "linux": LinuxAdapter,
}


def detect_system_name(system: str | None = None) -> str:
    """Normalise a ``platform.system()`` value to windows/macos/linux."""
    raw = (system if system is not None else _platform_module.system()).lower()
    if raw == "darwin":
        return "macos"
    if raw.startswith("win"):
        return "windows"
    return "linux"


def get_adapter(system: str | None = None) -> PlatformAdapter:
    """Return the correct adapter for *system* (auto-detected by default).

    >>> type(get_adapter("Windows")).__name__
    'WindowsAdapter'
    >>> type(get_adapter("Darwin")).__name__
    'MacOSAdapter'
    >>> type(get_adapter("Linux")).__name__
    'LinuxAdapter'
    """
    key = detect_system_name(system)
    adapter_cls = _SYSTEM_MAP[key]
    return adapter_cls()

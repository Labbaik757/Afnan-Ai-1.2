"""Cross-platform security abstraction.

Windows, Linux and macOS share one security vocabulary;
OS-specific behavior stays inside this adapter:

* per-OS sensitive paths (never writable by sandboxes)
* per-OS executable rules (e.g. .exe/.bat on Windows)
* per-OS sandbox notes (seatbelt/bwrap/Job Objects differ,
  but the policy language is identical)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlatformSecurityProfile:
    os_name: str  # "linux" | "darwin" | "windows"
    sensitive_paths: tuple[str, ...]
    executable_suffixes: tuple[str, ...]
    sandbox_mechanism: str
    notes: str = ""


_PROFILES: dict[str, PlatformSecurityProfile] = {
    "linux": PlatformSecurityProfile(
        os_name="linux",
        sensitive_paths=(
            "/etc", "/root", "/sys", "/proc", "/boot",
            "/usr/sbin",
        ),
        executable_suffixes=("", ".sh"),
        sandbox_mechanism=(
            "bwrap / user-namespaces where available; "
            "fallback: rlimits + scrubbed env"
        ),
        notes="POSIX rlimits available for CPU/memory caps.",
    ),
    "darwin": PlatformSecurityProfile(
        os_name="darwin",
        sensitive_paths=(
            "/etc", "/private/etc", "/System", "/Library",
            "/usr/sbin",
        ),
        executable_suffixes=("", ".sh", ".command"),
        sandbox_mechanism=(
            "seatbelt profiles where available; fallback: "
            "rlimits + scrubbed env"
        ),
        notes="App Sandbox entitlements are per-app.",
    ),
    "windows": PlatformSecurityProfile(
        os_name="windows",
        sensitive_paths=(
            "C:\\Windows", "C:\\Program Files",
            "C:\\Program Files (x86)",
        ),
        executable_suffixes=(
            ".exe", ".bat", ".cmd", ".ps1", ".vbs",
        ),
        sandbox_mechanism=(
            "Job Objects + restricted tokens where "
            "available; fallback: scrubbed env"
        ),
        notes="Extension-based execution policy applies.",
    ),
}


def current_profile() -> PlatformSecurityProfile:
    if sys.platform.startswith("win"):
        return _PROFILES["windows"]
    if sys.platform == "darwin":
        return _PROFILES["darwin"]
    return _PROFILES["linux"]


def is_executable_name(
    name: str, profile: PlatformSecurityProfile | None = None
) -> bool:
    """OS-aware executable check for sandbox policy."""
    profile = profile or current_profile()
    lowered = str(name or "").lower()
    return any(
        lowered.endswith(suffix)
        for suffix in profile.executable_suffixes
        if suffix
    ) or (
        "/" in lowered or "\\" in lowered
    )

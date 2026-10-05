"""Owner-controlled policy profiles.

Restricted / Standard / Advanced / Fully Authorized.

"Fully Authorized" is NOT a bypass: dangerous and
destructive operations stay under policy, audit logging
stays on, credential protection stays on, sandbox
boundaries stay on, and the emergency stop stays armed.
A profile only changes which capabilities are granted by
default — never whether the policy engine runs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class PolicyProfile(str, Enum):
    RESTRICTED = "restricted"
    STANDARD = "standard"
    ADVANCED = "advanced"
    FULLY_AUTHORIZED = "fully_authorized"

    @classmethod
    def coerce(cls, value: Any) -> "PolicyProfile":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        for member in cls:
            if member.value == text:
                return member
        return cls.STANDARD


# capability_id → profiles that grant it by default.
_PROFILE_GRANTS: dict[str, set[str]] = {
    "browser.read": {"restricted", "standard", "advanced",
                     "fully_authorized"},
    "browser.navigate": {"standard", "advanced",
                         "fully_authorized"},
    "browser.click": {"standard", "advanced",
                      "fully_authorized"},
    "browser.download": {"standard", "advanced",
                         "fully_authorized"},
    "computer.observe": {"restricted", "standard",
                         "advanced", "fully_authorized"},
    "computer.input": {"advanced", "fully_authorized"},
    "filesystem.read": {"restricted", "standard",
                        "advanced", "fully_authorized"},
    "filesystem.write": {"standard", "advanced",
                         "fully_authorized"},
    "filesystem.delete": {"advanced", "fully_authorized"},
    "connector.email.read": {"standard", "advanced",
                             "fully_authorized"},
    "connector.email.send": {"advanced", "fully_authorized"},
    "connector.calendar.write": {"standard", "advanced",
                                 "fully_authorized"},
    "connector.bulk_delete": {"fully_authorized"},
    "skill.execute": {"advanced", "fully_authorized"},
    "code.sandbox.execute": {"standard", "advanced",
                              "fully_authorized"},
    "artifact.create": {"standard", "advanced",
                        "fully_authorized"},
    "artifact.delete": {"advanced", "fully_authorized"},
    "memory.write": {"standard", "advanced",
                     "fully_authorized"},
    "subagent.spawn": {"standard", "advanced",
                       "fully_authorized"},
}

# Invariants that hold in EVERY profile, including
# fully_authorized.  Documented here, enforced in code.
PROFILE_INVARIANTS: tuple[str, ...] = (
    "dangerous/destructive operations stay under policy",
    "audit logging always active",
    "credential protection always active",
    "sandbox boundaries always active",
    "emergency stop always armed",
    "no capability may bypass the policy engine",
)


def grants_for(
    profile: PolicyProfile | str
) -> tuple[str, ...]:
    """Default capability grants for a profile."""
    profile = PolicyProfile.coerce(profile)
    return tuple(
        sorted(
            cap_id
            for cap_id, profiles in _PROFILE_GRANTS.items()
            if profile.value in profiles
        )
    )


def describe_profile(
    profile: PolicyProfile | str
) -> dict[str, Any]:
    profile = PolicyProfile.coerce(profile)
    return {
        "profile": profile.value,
        "grants": grants_for(profile),
        "invariants": list(PROFILE_INVARIANTS),
    }

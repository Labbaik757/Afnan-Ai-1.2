"""Policy versioning.

Policies are versioned.  Every audit record carries the
policy version it was decided under, so the future can
tell exactly which authorization policy an action ran
under.

When the policy changes, running tasks do NOT gain new
permissions automatically: background workers snapshot
the version with their permission snapshot and revalidate
on restart — a version mismatch forces re-evaluation
under the new policy.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PolicyVersion:
    version_id: str
    created_at: float
    profile: str
    changelog: str = ""
    capabilities_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "created_at": self.created_at,
            "profile": self.profile,
            "changelog": self.changelog,
            "capabilities_hash": self.capabilities_hash,
        }


class PolicyVersionStore:
    def __init__(self) -> None:
        self._versions: list[PolicyVersion] = []
        self._lock = threading.RLock()

    def publish(
        self,
        profile: str,
        *,
        changelog: str = "",
        capabilities_hash: str = "",
    ) -> PolicyVersion:
        version = PolicyVersion(
            version_id=(
                f"pv-{time.strftime('%Y%m%d%H%M%S')}-"
                f"{uuid.uuid4().hex[:6]}"
            ),
            created_at=time.time(),
            profile=str(profile),
            changelog=str(changelog),
            capabilities_hash=str(capabilities_hash),
        )
        with self._lock:
            self._versions.append(version)
        return version

    def current(self) -> PolicyVersion | None:
        with self._lock:
            return (
                self._versions[-1]
                if self._versions
                else None
            )

    def get(self, version_id: str) -> PolicyVersion | None:
        with self._lock:
            for version in self._versions:
                if version.version_id == version_id:
                    return version
        return None

    def history(self) -> list[PolicyVersion]:
        with self._lock:
            return list(self._versions)

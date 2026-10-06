"""Artifact lineage: what each artifact was built from.

Every artifact tracks: task → subtask → source data →
relevant actions → verification.  The final output is
always traceable to the verified information behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afnan_ai.redaction import redact_text


@dataclass
class ArtifactLineage:
    artifact_id: str
    task_id: str = ""
    subtask_id: str = ""
    source_refs: list[str] = field(default_factory=list)
    action_refs: list[str] = field(default_factory=list)
    verification: str = ""
    verified: bool = False
    versions: list[dict[str, Any]] = field(
        default_factory=list
    )  # incremental updates: {version, note, at}

    def add_version(self, note: str, *, at: str = "") -> int:
        self.versions.append(
            {"version": len(self.versions) + 1, "note": note[:200], "at": at}
        )
        return len(self.versions)

    def mark_verified(self, note: str = "") -> None:
        self.verified = True
        self.verification = note[:300]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "subtask_id": self.subtask_id,
            "source_refs": list(self.source_refs),
            "action_refs": list(self.action_refs),
            "verification": redact_text(self.verification)[:300],
            "verified": self.verified,
            "versions": list(self.versions),
        }


class LineageRegistry:
    """Track lineage for every artifact a task creates."""

    def __init__(self) -> None:
        self._lineage: dict[str, ArtifactLineage] = {}

    def register(
        self,
        artifact_id: str,
        *,
        task_id: str = "",
        subtask_id: str = "",
        source_refs: list[str] | None = None,
    ) -> ArtifactLineage:
        lineage = self._lineage.get(artifact_id)
        if lineage is None:
            lineage = ArtifactLineage(
                artifact_id=artifact_id,
                task_id=task_id,
                subtask_id=subtask_id,
                source_refs=list(source_refs or []),
            )
            self._lineage[artifact_id] = lineage
        return lineage

    def record_action(
        self, artifact_id: str, action_ref: str
    ) -> None:
        lineage = self._lineage.get(artifact_id)
        if lineage is not None and action_ref not in lineage.action_refs:
            lineage.action_refs.append(action_ref)

    def get(self, artifact_id: str) -> ArtifactLineage | None:
        return self._lineage.get(artifact_id)

    def unverified(self) -> list[str]:
        return [
            aid
            for aid, lin in self._lineage.items()
            if not lin.verified
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            aid: lin.to_dict()
            for aid, lin in self._lineage.items()
        }

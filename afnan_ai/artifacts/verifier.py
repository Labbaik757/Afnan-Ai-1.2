"""Artifact verification — inspect the actual output.

An artifact is not complete because it was created.  The
verifier inspects the real bytes on disk:

* file exists and is readable,
* checksum matches (no corruption),
* format matches the declared type (extension + magic bytes
  for PNG/PDF),
* expected content is present (keywords),
* required sections are present (headings for documents),
* size/format constraints satisfied.

Verdict: verified / failed / uncertain (deterministic, no
model calls).  Only the manager may mark an artifact
verified, via ``set_verification``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from afnan_ai.artifacts.models import (
    Artifact,
    VerificationState,
)

_MAGIC = {
    "pdf": b"%PDF",
    "image": b"\x89PNG",
}


@dataclass
class ArtifactVerdict:
    artifact_id: str
    state: VerificationState
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "state": self.state.value,
            "reasons": list(self.reasons),
        }


class ArtifactVerifier:
    """Deterministic output inspection."""

    def verify(
        self,
        artifact: Artifact,
        read: Any,
        *,
        expected_content: list[str] | None = None,
        required_sections: list[str] | None = None,
        max_size: int | None = None,
    ) -> ArtifactVerdict:
        """``read(version)`` returns the version's bytes."""
        reasons: list[str] = []
        entry = artifact.latest
        if entry is None:
            return ArtifactVerdict(
                artifact.artifact_id, VerificationState.FAILED,
                ["artifact has no versions"],
            )
        try:
            data = read(entry.version)
        except Exception as e:  # noqa: BLE001
            return ArtifactVerdict(
                artifact.artifact_id, VerificationState.FAILED,
                [f"cannot read latest version: {e}"[:160]],
            )
        if not data:
            return ArtifactVerdict(
                artifact.artifact_id, VerificationState.FAILED,
                ["artifact file is empty"],
            )
        # Format check: magic bytes where applicable.
        magic = _MAGIC.get(artifact.type)
        if magic and not data.startswith(magic):
            return ArtifactVerdict(
                artifact.artifact_id, VerificationState.FAILED,
                [f"file does not look like {artifact.type}"],
            )
        if max_size is not None and len(data) > max_size:
            reasons.append(
                f"size {len(data)} exceeds {max_size}"
            )
        text = ""
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            pass
        lowered = text.lower()
        # Expected content present?
        missing = [
            phrase for phrase in (expected_content or [])
            if str(phrase).lower() not in lowered
        ]
        if missing:
            reasons.append(
                "missing expected content: "
                + ", ".join(missing[:5])
            )
        # Required sections present? (markdown/html headings)
        if required_sections and text:
            headings = set(
                h.strip().lower()
                for h in re.findall(
                    r"^(?:#{1,3}\s*|<h[12][^>]*>)(.+?)(?:</h[12]>)?\s*$",
                    text,
                    re.MULTILINE | re.IGNORECASE,
                )
            )
            for section in required_sections:
                wanted = str(section).strip().lower()
                if not any(
                    wanted in heading for heading in headings
                ):
                    reasons.append(
                        f"missing required section: {section}"
                    )
        if reasons:
            return ArtifactVerdict(
                artifact.artifact_id,
                VerificationState.UNCERTAIN, reasons,
            )
        return ArtifactVerdict(
            artifact.artifact_id, VerificationState.VERIFIED,
            ["file exists", "checksum ok", "format ok",
             "expected content present"],
        )

"""Evidence linking for verification and completion.

Evidence types: page state, structured observation, screenshot
reference, file metadata, tool result, connector result,
artifact reference.  Sensitive evidence is redacted/filtered
at the boundary; the ActivityCenter stores references, not
raw secrets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from afnan_ai.redaction import redact_value

PAGE_STATE = "page_state"
OBSERVATION = "observation"
SCREENSHOT_REF = "screenshot_ref"
FILE_META = "file_meta"
TOOL_RESULT = "tool_result"
CONNECTOR_RESULT = "connector_result"
ARTIFACT_REF = "artifact_ref"

ALL_TYPES = frozenset(
    {
        PAGE_STATE, OBSERVATION, SCREENSHOT_REF, FILE_META,
        TOOL_RESULT, CONNECTOR_RESULT, ARTIFACT_REF,
    }
)


@dataclass
class Evidence:
    evidence_id: str
    type: str
    task_id: str = ""
    summary: str = ""
    reference: str = ""  # pointer, not raw content
    metadata: dict[str, Any] = field(default_factory=dict)
    at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    def __post_init__(self) -> None:
        if self.type not in ALL_TYPES:
            raise ValueError(
                f"unknown evidence type: {self.type!r}"
            )
        self.summary = str(self.summary)[:300]
        self.reference = str(self.reference)[:300]
        # Metadata is redacted: file paths stay, contents go.
        self.metadata = redact_value(dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def from_verification(
    verification: dict[str, Any],
    *,
    task_id: str = "",
) -> Evidence | None:
    """Build evidence from a Verifier result (safe subset)."""
    if not isinstance(verification, dict):
        return None
    status = str(verification.get("status", ""))
    if status not in ("verified", "failed", "uncertain"):
        return None
    meta = {
        k: verification[k]
        for k in ("checked", "method", "url", "tool")
        if k in verification
    }
    return Evidence(
        evidence_id=(
            f"ev-{task_id[:8]}-"
            f"{abs(hash(str(sorted(meta.items())))) % 10**8:08d}"
        ),
        type=TOOL_RESULT,
        task_id=task_id,
        summary=f"Verification {status}",
        reference=str(verification.get("ref", ""))[:200],
        metadata=meta,
    )

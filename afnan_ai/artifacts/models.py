"""Artifact data models.

An Artifact is a durable, versioned deliverable the agent
produces: a document, report, spreadsheet, presentation,
image, web page, dataset or code output.  Types are
extensible — ``ArtifactType.coerce`` accepts any string and
maps the known ones to enum members.

Every content write passes through the shared redactor, so
secrets never land in an artifact by accident; source
references distinguish verified evidence from unverified
model-generated claims (the two are never conflated).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactType(str, Enum):
    DOCUMENT = "document"
    PDF = "pdf"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    IMAGE = "image"
    REPORT = "report"
    HTML = "html"
    STRUCTURED_DATA = "structured_data"
    CODE_OUTPUT = "code_output"

    @classmethod
    def coerce(cls, value: Any) -> str:
        """Known types map to enum values; anything else is
        kept as a custom type string (extensibility)."""
        text = str(value or "").strip().lower()
        for member in cls:
            if member.value == text:
                return member.value
        return text or cls.DOCUMENT.value


class ArtifactStatus(str, Enum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    READY = "ready"
    VERIFIED = "verified"
    FAILED = "failed"
    ARCHIVED = "archived"
    DELETED = "deleted"


class VerificationState(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


@dataclass
class SourceRef:
    """One evidence link for a generated claim.

    ``verified`` defaults to False and is only set by an
    explicit verification step — unverified model-generated
    facts are never marked as verified evidence.
    """

    claim: str
    source: str  # URL, file path, or tool-result reference
    excerpt: str = ""
    verified: bool = False
    added_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "source": self.source,
            "excerpt": self.excerpt[:500],
            "verified": self.verified,
            "added_at": self.added_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceRef":
        return cls(
            claim=str(data.get("claim", "")),
            source=str(data.get("source", "")),
            excerpt=str(data.get("excerpt", "")),
            verified=bool(data.get("verified", False)),
            added_at=str(data.get("added_at", _utcnow())),
        )


@dataclass
class ArtifactVersion:
    version: int
    file_name: str  # relative to the artifact directory
    checksum: str  # sha256 hex of the stored bytes
    size: int
    created_at: str = field(default_factory=_utcnow)
    change_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "file_name": self.file_name,
            "checksum": self.checksum,
            "size": self.size,
            "created_at": self.created_at,
            "change_note": self.change_note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactVersion":
        return cls(
            version=int(data.get("version", 1)),
            file_name=str(data.get("file_name", "")),
            checksum=str(data.get("checksum", "")),
            size=int(data.get("size", 0)),
            created_at=str(data.get("created_at", _utcnow())),
            change_note=str(data.get("change_note", "")),
        )


@dataclass
class Artifact:
    """A versioned deliverable."""

    artifact_id: str
    name: str
    type: str
    description: str = ""
    source_task_id: str = ""
    project_id: str = "default"
    goal_id: str = ""
    current_version: int = 1
    status: ArtifactStatus = ArtifactStatus.DRAFT
    verification: VerificationState = (
        VerificationState.UNVERIFIED
    )
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    versions: list[ArtifactVersion] = field(default_factory=list)
    sources: list[SourceRef] = field(default_factory=list)
    sensitive: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.artifact_id = str(self.artifact_id).strip()
        if not self.artifact_id:
            raise ValueError("artifact_id is required")
        self.type = ArtifactType.coerce(self.type)
        if isinstance(self.status, str):
            self.status = ArtifactStatus(self.status)
        if isinstance(self.verification, str):
            self.verification = VerificationState(
                self.verification
            )

    @property
    def latest(self) -> ArtifactVersion | None:
        return self.versions[-1] if self.versions else None

    def get_version(
        self, version: int
    ) -> ArtifactVersion | None:
        for entry in self.versions:
            if entry.version == version:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "source_task_id": self.source_task_id,
            "project_id": self.project_id,
            "goal_id": self.goal_id,
            "current_version": self.current_version,
            "status": self.status.value,
            "verification": self.verification.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "versions": [v.to_dict() for v in self.versions],
            "sources": [s.to_dict() for s in self.sources],
            "sensitive": self.sensitive,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        return cls(
            artifact_id=str(data.get("artifact_id", "")),
            name=str(data.get("name", "")),
            type=str(data.get("type", "document")),
            description=str(data.get("description", "")),
            source_task_id=str(data.get("source_task_id", "")),
            project_id=str(data.get("project_id", "default")),
            goal_id=str(data.get("goal_id", "")),
            current_version=int(data.get("current_version", 1)),
            status=str(data.get("status", "draft")),
            verification=str(
                data.get("verification", "unverified")
            ),
            created_at=str(data.get("created_at", _utcnow())),
            updated_at=str(data.get("updated_at", _utcnow())),
            versions=[
                ArtifactVersion.from_dict(v)
                for v in (data.get("versions") or [])
            ],
            sources=[
                SourceRef.from_dict(s)
                for s in (data.get("sources") or [])
            ],
            sensitive=bool(data.get("sensitive", False)),
            metadata=dict(data.get("metadata") or {}),
        )

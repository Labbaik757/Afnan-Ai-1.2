"""ArtifactManager — create/read/update/version/rename/
duplicate/export/archive/delete for deliverables.

Design notes:

* Writes are atomic (temp file + rename) and checksummed;
  every meaningful modification creates a new version, and
  a failed update restores the previous stable version.
* Concurrent edits are safe: per-artifact locks plus
  optimistic version checks (``expected_version``) — a
  stale writer gets a structured conflict, never silent
  corruption.  Parallel subagents share the manager safely.
* Destructive operations (delete, export of sensitive
  artifacts) go through the human-approval rules: with no
  approver they raise ``approval_required`` instead of
  running.
* Content is redacted on the way in; source references
  track evidence per claim without ever marking unverified
  model output as verified.
* Nothing here duplicates the AgentLoop — the manager is a
  controlled interface the agent (and its tools) call.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from afnan_ai.artifacts.builders import builder_for
from afnan_ai.artifacts.models import (
    Artifact,
    ArtifactStatus,
    ArtifactType,
    ArtifactVersion,
    SourceRef,
    VerificationState,
)
from afnan_ai.artifacts.workspace import ArtifactWorkspace
from afnan_ai.log_config import get_logger
from afnan_ai.persistence import JsonFileStore
from afnan_ai.redaction import redact_text, redact_value

logger = get_logger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactError(Exception):
    """Structured artifact failure."""

    def __init__(
        self, message: str, *, code: str = "artifact_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "message": str(self),
            "details": dict(self.details),
        }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ArtifactManager:
    """Versioned, audited artifact lifecycle management."""

    def __init__(
        self,
        workspace: ArtifactWorkspace,
        *,
        approver: Callable[[str], bool] | None = None,
        audit_path: str | Path | None = None,
    ) -> None:
        self.workspace = workspace
        self.approver = approver
        self._audit_path = (
            Path(str(audit_path)).expanduser()
            if audit_path else None
        )
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.RLock()

    # -- internal -------------------------------------------------------
    def _lock_for(self, artifact_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(
                artifact_id, threading.RLock()
            )

    def _index(self, project_id: str) -> JsonFileStore:
        return JsonFileStore(
            str(self.workspace.index_path(project_id))
        )

    def _load_all(
        self, project_id: str
    ) -> dict[str, dict[str, Any]]:
        return self._index(project_id).read(
            {"artifacts": {}}
        ).get("artifacts", {})

    def _save_all(
        self, project_id: str,
        records: dict[str, dict[str, Any]],
    ) -> None:
        self._index(project_id).write(
            {"artifacts": records}
        )

    def _get_record(
        self, artifact_id: str, project_id: str
    ) -> Artifact:
        records = self._load_all(project_id)
        data = records.get(artifact_id)
        if data is None:
            raise ArtifactError(
                f"unknown artifact {artifact_id!r}",
                code="unknown_artifact",
            )
        return Artifact.from_dict(data)

    def _put_record(self, artifact: Artifact) -> None:
        records = self._load_all(artifact.project_id)
        records[artifact.artifact_id] = artifact.to_dict()
        self._save_all(artifact.project_id, records)

    def _version_path(
        self, artifact: Artifact, version: ArtifactVersion,
        category: str,
    ) -> Path:
        return (
            self.workspace.artifact_dir(
                artifact.project_id, category,
                artifact.artifact_id,
            )
            / version.file_name
        )

    def _write_version_file(
        self, path: Path, content: bytes
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(content)
        os.replace(tmp, path)

    # -- lifecycle ------------------------------------------------------
    def create(
        self,
        *,
        name: str,
        artifact_type: str = "document",
        data: dict[str, Any] | None = None,
        content: bytes | None = None,
        description: str = "",
        source_task_id: str = "",
        project_id: str = "default",
        goal_id: str = "",
        category: str = "drafts",
        sensitive: bool = False,
        change_note: str = "created",
    ) -> Artifact:
        """Create an artifact from builder *data* (preferred)
        or raw *content* bytes."""
        artifact_id = uuid.uuid4().hex[:12]
        kind = ArtifactType.coerce(artifact_type)
        builder = builder_for(kind)
        if content is None:
            content = builder.build(data or {})
        elif isinstance(content, str):
            content = content.encode("utf-8")
        # Secrets never land in artifacts unredacted.
        if isinstance(content, bytes):
            try:
                text = content.decode("utf-8")
                redacted = redact_text(text)
                content = redacted.encode("utf-8")
            except UnicodeDecodeError:
                pass  # binary (png/pdf): checksum still guards
        artifact = Artifact(
            artifact_id=artifact_id,
            name=str(name or "untitled"),
            type=kind,
            description=redact_text(description)[:500],
            source_task_id=str(source_task_id),
            project_id=project_id,
            goal_id=str(goal_id),
            status=ArtifactStatus.DRAFT,
            sensitive=bool(sensitive),
        )
        version = ArtifactVersion(
            version=1,
            file_name=f"v1{builder.extension}",
            checksum=_sha256(content),
            size=len(content),
            change_note=change_note,
        )
        with self._lock_for(artifact_id):
            self._write_version_file(
                self._version_path(artifact, version, category),
                content,
            )
            artifact.versions.append(version)
            artifact.metadata["category"] = category
            self._put_record(artifact)
        self._audit("created", artifact)
        logger.info(
            "artifact created: %s (%s)", artifact_id, kind
        )
        return artifact

    def read(
        self, artifact_id: str, *,
        project_id: str = "default",
        version: int | None = None,
    ) -> bytes:
        artifact = self._get_record(artifact_id, project_id)
        if artifact.status is ArtifactStatus.DELETED:
            raise ArtifactError(
                f"artifact {artifact_id!r} is deleted",
                code="deleted",
            )
        entry = (
            artifact.get_version(version)
            if version is not None
            else artifact.latest
        )
        if entry is None:
            raise ArtifactError(
                f"no such version {version!r}",
                code="unknown_version",
            )
        path = self._version_path(
            artifact, entry,
            artifact.metadata.get("category", "drafts"),
        )
        data = path.read_bytes()
        if _sha256(data) != entry.checksum:
            raise ArtifactError(
                f"artifact {artifact_id!r} v{entry.version} "
                "failed checksum — file corrupted",
                code="corrupted",
            )
        return data

    def update(
        self,
        artifact_id: str,
        *,
        data: dict[str, Any] | None = None,
        content: bytes | None = None,
        change_note: str = "",
        expected_version: int | None = None,
        project_id: str = "default",
    ) -> Artifact:
        """Write a new version atomically.  A failed update
        restores the previous stable version; a stale
        ``expected_version`` raises a structured conflict."""
        with self._lock_for(artifact_id):
            artifact = self._get_record(
                artifact_id, project_id
            )
            if artifact.status in (
                ArtifactStatus.DELETED,
                ArtifactStatus.ARCHIVED,
            ):
                raise ArtifactError(
                    f"artifact {artifact_id!r} is "
                    f"{artifact.status.value}",
                    code="immutable",
                )
            if (
                expected_version is not None
                and expected_version != artifact.current_version
            ):
                raise ArtifactError(
                    f"version conflict: expected v"
                    f"{expected_version}, current v"
                    f"{artifact.current_version}",
                    code="version_conflict",
                    details={
                        "expected": expected_version,
                        "current": artifact.current_version,
                    },
                )
            builder = builder_for(artifact.type)
            new_content = (
                builder.build(data or {})
                if content is None else content
            )
            if isinstance(new_content, bytes):
                try:
                    new_content = redact_text(
                        new_content.decode("utf-8")
                    ).encode("utf-8")
                except UnicodeDecodeError:
                    pass
            new_version = artifact.current_version + 1
            entry = ArtifactVersion(
                version=new_version,
                file_name=f"v{new_version}{builder.extension}",
                checksum=_sha256(new_content),
                size=len(new_content),
                change_note=change_note or f"v{new_version}",
            )
            path = self._version_path(
                artifact, entry,
                artifact.metadata.get("category", "drafts"),
            )
            previous = artifact.latest
            try:
                self._write_version_file(path, new_content)
                # Verify what we wrote before committing.
                written = path.read_bytes()
                if _sha256(written) != entry.checksum:
                    raise ArtifactError(
                        "write verification failed",
                        code="corrupted",
                    )
            except Exception:
                # Restore previous stable version.
                if path.exists():
                    try:
                        path.unlink()
                    except OSError:
                        pass
                raise
            artifact.versions.append(entry)
            artifact.current_version = new_version
            artifact.updated_at = _utcnow()
            artifact.verification = (
                VerificationState.UNVERIFIED
            )
            if artifact.status is ArtifactStatus.VERIFIED:
                artifact.status = ArtifactStatus.READY
            self._put_record(artifact)
        self._audit("updated", artifact)
        return artifact

    def rename(
        self, artifact_id: str, new_name: str, *,
        project_id: str = "default",
    ) -> Artifact:
        with self._lock_for(artifact_id):
            artifact = self._get_record(
                artifact_id, project_id
            )
            artifact.name = str(new_name)[:200]
            artifact.updated_at = _utcnow()
            self._put_record(artifact)
        self._audit("renamed", artifact)
        return artifact

    def duplicate(
        self, artifact_id: str, *,
        new_name: str | None = None,
        project_id: str = "default",
    ) -> Artifact:
        with self._lock_for(artifact_id):
            source = self._get_record(artifact_id, project_id)
            content = self.read(
                artifact_id, project_id=project_id
            )
        return self.create(
            name=new_name or f"{source.name} (copy)",
            artifact_type=source.type,
            content=content,
            description=source.description,
            source_task_id=source.source_task_id,
            project_id=source.project_id,
            goal_id=source.goal_id,
            category=source.metadata.get(
                "category", "drafts"
            ),
            sensitive=source.sensitive,
            change_note=f"duplicated from {artifact_id}",
        )

    def export(
        self, artifact_id: str, dest: str | Path, *,
        project_id: str = "default",
    ) -> Path:
        """Copy the latest version out of the workspace.
        Sensitive artifacts (and all exports, when an
        approver is configured) need human approval."""
        with self._lock_for(artifact_id):
            artifact = self._get_record(
                artifact_id, project_id
            )
            if artifact.sensitive or self.approver is not None:
                self._require_approval(
                    artifact,
                    f"Export artifact '{artifact.name}' "
                    f"to {dest}?",
                )
            content = self.read(
                artifact_id, project_id=project_id
            )
            dest_path = Path(str(dest)).expanduser()
            dest_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            dest_path.write_bytes(content)
        self._audit("exported", artifact)
        return dest_path

    def archive(
        self, artifact_id: str, *,
        project_id: str = "default",
    ) -> Artifact:
        with self._lock_for(artifact_id):
            artifact = self._get_record(
                artifact_id, project_id
            )
            artifact.status = ArtifactStatus.ARCHIVED
            artifact.updated_at = _utcnow()
            self._put_record(artifact)
        self._audit("archived", artifact)
        return artifact

    def delete(
        self, artifact_id: str, *,
        project_id: str = "default",
    ) -> Artifact:
        """Destructive: always needs human approval."""
        with self._lock_for(artifact_id):
            artifact = self._get_record(
                artifact_id, project_id
            )
            self._require_approval(
                artifact,
                f"Delete artifact '{artifact.name}' "
                f"(v{artifact.current_version})?",
            )
            artifact.status = ArtifactStatus.DELETED
            artifact.updated_at = _utcnow()
            self._put_record(artifact)
        self._audit("deleted", artifact)
        return artifact

    def _require_approval(
        self, artifact: Artifact, request: str
    ) -> None:
        if self.approver is None:
            raise ArtifactError(
                "destructive artifact operation needs human "
                "approval; no approver configured",
                code="approval_required",
                details={
                    "artifact_id": artifact.artifact_id,
                    "resumable": True,
                },
            )
        try:
            allowed = self.approver(request)
        except Exception:
            allowed = False
        if not allowed:
            raise ArtifactError(
                "human denied the artifact operation",
                code="approval_denied",
                details={
                    "artifact_id": artifact.artifact_id
                },
            )

    def set_approver(
        self, approver: Callable[[str], bool] | None
    ) -> None:
        self.approver = approver

    # -- sources / evidence ---------------------------------------------------
    def add_source(
        self, artifact_id: str, claim: str, source: str, *,
        excerpt: str = "",
        verified: bool = False,
        project_id: str = "default",
    ) -> SourceRef:
        """Attach evidence for a claim.  ``verified`` stays
        False unless an explicit verification step sets it —
        model-generated facts are never auto-marked."""
        with self._lock_for(artifact_id):
            artifact = self._get_record(
                artifact_id, project_id
            )
            ref = SourceRef(
                claim=redact_text(claim)[:500],
                source=redact_text(source)[:500],
                excerpt=redact_text(excerpt),
                verified=bool(verified),
            )
            artifact.sources.append(ref)
            artifact.updated_at = _utcnow()
            self._put_record(artifact)
        self._audit("source_added", artifact)
        return ref

    def mark_source_verified(
        self, artifact_id: str, claim: str, *,
        project_id: str = "default",
    ) -> None:
        with self._lock_for(artifact_id):
            artifact = self._get_record(
                artifact_id, project_id
            )
            for ref in artifact.sources:
                if ref.claim == claim:
                    ref.verified = True
            artifact.updated_at = _utcnow()
            self._put_record(artifact)

    def get_sources(
        self, artifact_id: str, *,
        project_id: str = "default",
    ) -> list[SourceRef]:
        return self._get_record(
            artifact_id, project_id
        ).sources

    # -- lookup ---------------------------------------------------------------
    def get(
        self, artifact_id: str, *,
        project_id: str = "default",
    ) -> Artifact:
        return self._get_record(artifact_id, project_id)

    def list_artifacts(
        self, *,
        project_id: str = "default",
        status: ArtifactStatus | None = None,
        artifact_type: str | None = None,
    ) -> list[Artifact]:
        records = self._load_all(project_id)
        out = []
        for data in records.values():
            artifact = Artifact.from_dict(data)
            if status is not None and artifact.status is not status:
                continue
            if (
                artifact_type is not None
                and artifact.type != ArtifactType.coerce(artifact_type)
            ):
                continue
            out.append(artifact)
        return sorted(out, key=lambda a: a.created_at)

    def set_status(
        self, artifact_id: str, status: ArtifactStatus, *,
        project_id: str = "default",
    ) -> Artifact:
        with self._lock_for(artifact_id):
            artifact = self._get_record(
                artifact_id, project_id
            )
            if isinstance(status, str):
                status = ArtifactStatus(status)
            artifact.status = status
            artifact.updated_at = _utcnow()
            self._put_record(artifact)
        return artifact

    def set_verification(
        self, artifact_id: str,
        verification: VerificationState, *,
        project_id: str = "default",
    ) -> Artifact:
        with self._lock_for(artifact_id):
            artifact = self._get_record(
                artifact_id, project_id
            )
            if isinstance(verification, str):
                verification = VerificationState(verification)
            artifact.verification = verification
            if verification is VerificationState.VERIFIED:
                artifact.status = ArtifactStatus.VERIFIED
            elif verification is VerificationState.FAILED:
                artifact.status = ArtifactStatus.FAILED
            artifact.updated_at = _utcnow()
            self._put_record(artifact)
        return artifact

    def checkpoint_ref(
        self, artifact_id: str, *,
        project_id: str = "default",
    ) -> dict[str, Any]:
        """Resumable pointer for background workflows."""
        artifact = self._get_record(artifact_id, project_id)
        return {
            "artifact_id": artifact_id,
            "project_id": project_id,
            "version": artifact.current_version,
            "status": artifact.status.value,
        }

    # -- audit ------------------------------------------------------------------
    def _audit(
        self, event: str, artifact: Artifact,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self._audit_path is None:
            return
        try:
            store = JsonFileStore(str(self._audit_path))
            data = store.read({"events": []})
            events = data.get("events", [])
            entry: dict[str, Any] = {
                "event": event,
                "artifact_id": artifact.artifact_id,
                "name": artifact.name,
                "type": artifact.type,
                "version": artifact.current_version,
                "status": artifact.status.value,
                "at": _utcnow(),
            }
            if extra:
                entry["extra"] = redact_value(extra)
            events.append(entry)
            store.write({"events": events[-500:]})
        except Exception as e:  # noqa: BLE001 - never breaks
            logger.warning("artifact audit failed: %s", e)

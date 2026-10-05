"""Agent-facing artifact tools.

Thin wrappers over the ArtifactManager — the tools do no
orchestration themselves; they just let the Planner drive
artifact work through the normal AgentLoop path.
Destructive operations flow through the manager's
human-approval rules (approval_required pauses the loop
resumably, like every other sensitive tool).
"""

from __future__ import annotations

from typing import Any

from afnan_ai.artifacts.manager import (
    ArtifactError,
    ArtifactManager,
)
from afnan_ai.artifacts.models import ArtifactStatus
from afnan_ai.artifacts.verifier import ArtifactVerifier
from afnan_ai.tools.base import (
    FunctionTool,
    Tool,
    ToolErrorCode,
    ToolExecutionError,
)


def _wrap(manager: ArtifactManager, func, tool_name: str):
    def _run(**kwargs: Any) -> Any:
        try:
            return func(**kwargs)
        except ArtifactError as e:
            code = (
                ToolErrorCode.EXECUTION_FAILED
            )
            if e.code in ("approval_required", "approval_denied"):
                # Reuse the loop's existing approval signal.
                raise ToolExecutionError(
                    str(e), code=e.code, tool=tool_name,
                    details=e.to_dict(),
                ) from e
            raise ToolExecutionError(
                str(e), code=code, tool=tool_name,
                details=e.to_dict(),
            ) from e

    return _run


def _obj_schema(properties, required=None):
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
    }


def _str(desc: str) -> dict[str, Any]:
    return {"type": "string", "description": desc}


def create_artifact_tools(
    manager: ArtifactManager,
) -> list[Tool]:
    """Build the artifact_* tools bound to *manager*."""
    verifier = ArtifactVerifier()
    tools: list[Tool] = []

    def _create(
        name: str, artifact_type: str = "document",
        title: str = "", introduction: str = "",
        sections: Any = None, description: str = "",
        project_id: str = "default",
        sensitive: bool = False,
    ) -> dict[str, Any]:
        artifact = manager.create(
            name=name, artifact_type=artifact_type,
            data={
                "title": title or name,
                "introduction": introduction,
                "sections": sections or [],
            },
            description=description,
            project_id=project_id,
            sensitive=sensitive,
        )
        return {
            "artifact_id": artifact.artifact_id,
            "version": artifact.current_version,
            "status": artifact.status.value,
        }

    tools.append(FunctionTool(
        "artifact_create",
        "Create a deliverable artifact (document/report/html/"
        "pdf/spreadsheet/presentation/image/structured_data). "
        "Sections is a list of {heading, body}.",
        _obj_schema({
            "name": _str("Artifact name"),
            "artifact_type": _str(
                "document|report|html|pdf|spreadsheet|"
                "presentation|image|structured_data"
            ),
            "title": _str("Title shown in the artifact"),
            "introduction": _str("Opening paragraph"),
            "sections": {
                "type": "array",
                "description": "Sections as {heading, body}",
            },
            "description": _str("Short description"),
            "project_id": _str("Project id"),
            "sensitive": {
                "type": "boolean",
                "description": "Mark sensitive",
            },
        }, ["name"]),
        _wrap(manager, _create, "artifact_create"),
    ))

    def _read(
        artifact_id: str, project_id: str = "default",
        version: Any = None,
    ) -> dict[str, Any]:
        data = manager.read(
            artifact_id, project_id=project_id,
            version=int(version) if version else None,
        )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = f"<binary {len(data)} bytes>"
        return {
            "artifact_id": artifact_id,
            "size": len(data),
            "content": text[:8000],
        }

    tools.append(FunctionTool(
        "artifact_read",
        "Read an artifact's latest (or given) version.",
        _obj_schema({
            "artifact_id": _str("Artifact id"),
            "project_id": _str("Project id"),
            "version": _str("Version number (optional)"),
        }, ["artifact_id"]),
        _wrap(manager, _read, "artifact_read"),
    ))

    def _update(
        artifact_id: str, title: str = "",
        introduction: str = "", sections: Any = None,
        change_note: str = "",
        expected_version: Any = None,
        project_id: str = "default",
    ) -> dict[str, Any]:
        artifact = manager.update(
            artifact_id,
            data={
                "title": title,
                "introduction": introduction,
                "sections": sections or [],
            },
            change_note=change_note,
            expected_version=(
                int(expected_version)
                if expected_version else None
            ),
            project_id=project_id,
        )
        return {
            "artifact_id": artifact.artifact_id,
            "version": artifact.current_version,
        }

    tools.append(FunctionTool(
        "artifact_update",
        "Write a new version of an artifact (atomic; "
        "expected_version guards against conflicts).",
        _obj_schema({
            "artifact_id": _str("Artifact id"),
            "title": _str("Title"),
            "introduction": _str("Opening paragraph"),
            "sections": {
                "type": "array",
                "description": "Sections as {heading, body}",
            },
            "change_note": _str("What changed"),
            "expected_version": _str(
                "Fail if current != this (optional)"
            ),
            "project_id": _str("Project id"),
        }, ["artifact_id"]),
        _wrap(manager, _update, "artifact_update"),
    ))

    def _list(
        project_id: str = "default",
        status: str = "",
    ) -> dict[str, Any]:
        artifacts = manager.list_artifacts(
            project_id=project_id,
            status=(
                ArtifactStatus(status) if status else None
            ),
        )
        return {
            "artifacts": [
                {
                    "artifact_id": a.artifact_id,
                    "name": a.name,
                    "type": a.type,
                    "version": a.current_version,
                    "status": a.status.value,
                    "verification": a.verification.value,
                }
                for a in artifacts
            ]
        }

    tools.append(FunctionTool(
        "artifact_list",
        "List artifacts in a project.",
        _obj_schema({
            "project_id": _str("Project id"),
            "status": _str("Filter by status (optional)"),
        }),
        _wrap(manager, _list, "artifact_list"),
    ))

    def _verify(
        artifact_id: str,
        expected_content: Any = None,
        required_sections: Any = None,
        project_id: str = "default",
    ) -> dict[str, Any]:
        artifact = manager.get(
            artifact_id, project_id=project_id
        )
        verdict = verifier.verify(
            artifact,
            lambda version: manager.read(
                artifact_id, project_id=project_id,
                version=version,
            ),
            expected_content=list(expected_content or []),
            required_sections=list(required_sections or []),
        )
        manager.set_verification(
            artifact_id, verdict.state,
            project_id=project_id,
        )
        return verdict.to_dict()

    tools.append(FunctionTool(
        "artifact_verify",
        "Verify an artifact's actual output (exists, format, "
        "content, sections, corruption check).",
        _obj_schema({
            "artifact_id": _str("Artifact id"),
            "expected_content": {
                "type": "array",
                "description": "Phrases that must appear",
            },
            "required_sections": {
                "type": "array",
                "description": "Headings that must exist",
            },
            "project_id": _str("Project id"),
        }, ["artifact_id"]),
        _wrap(manager, _verify, "artifact_verify"),
    ))

    def _add_source(
        artifact_id: str, claim: str, source: str,
        excerpt: str = "",
        project_id: str = "default",
    ) -> dict[str, Any]:
        ref = manager.add_source(
            artifact_id, claim, source, excerpt=excerpt,
            project_id=project_id,
        )
        return ref.to_dict()

    tools.append(FunctionTool(
        "artifact_add_source",
        "Attach a source/evidence reference for a claim in "
        "an artifact (stays unverified until checked).",
        _obj_schema({
            "artifact_id": _str("Artifact id"),
            "claim": _str("The claim"),
            "source": _str("URL, file or tool reference"),
            "excerpt": _str("Evidence excerpt"),
            "project_id": _str("Project id"),
        }, ["artifact_id", "claim", "source"]),
        _wrap(manager, _add_source, "artifact_add_source"),
    ))

    def _export(
        artifact_id: str, dest: str,
        project_id: str = "default",
    ) -> dict[str, Any]:
        path = manager.export(
            artifact_id, dest, project_id=project_id
        )
        return {"exported_to": str(path)}

    tools.append(FunctionTool(
        "artifact_export",
        "Export an artifact out of the workspace "
        "(sensitive artifacts need human approval).",
        _obj_schema({
            "artifact_id": _str("Artifact id"),
            "dest": _str("Destination path"),
            "project_id": _str("Project id"),
        }, ["artifact_id", "dest"]),
        _wrap(manager, _export, "artifact_export"),
    ))

    def _delete(
        artifact_id: str, project_id: str = "default",
    ) -> dict[str, Any]:
        artifact = manager.delete(
            artifact_id, project_id=project_id
        )
        return {
            "artifact_id": artifact.artifact_id,
            "status": artifact.status.value,
        }

    tools.append(FunctionTool(
        "artifact_delete",
        "Delete an artifact (always needs human approval).",
        _obj_schema({
            "artifact_id": _str("Artifact id"),
            "project_id": _str("Project id"),
        }, ["artifact_id"]),
        _wrap(manager, _delete, "artifact_delete"),
    ))

    return tools


__all__ = ["create_artifact_tools"]

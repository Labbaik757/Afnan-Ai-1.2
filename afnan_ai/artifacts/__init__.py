"""Artifact System for Afnan AI.

Research and task results become real, usable
deliverables: versioned, verified, auditable.

* :class:`Artifact` — id, name, type, description, source
  task, version, status, file reference, timestamps,
  verification state.  Types are extensible
  (document/pdf/spreadsheet/presentation/image/report/
  html/structured_data/code_output + custom strings).
* :class:`ArtifactManager` — create/read/update/version/
  rename/duplicate/export/archive/delete with atomic
  writes, checksums, optimistic concurrency, per-artifact
  locks (parallel subagents safe), source/evidence
  tracking, and human approval on destructive operations.
* Builders (``builders.py``) — controlled content
  generation: data in, bytes out.  The agent never gets
  direct filesystem/code execution for artifacts.
* :class:`ArtifactVerifier` — inspects the actual output
  (exists, format, content, sections, corruption) and
  returns verified/failed/uncertain.
* :class:`ArtifactWorkspace` — per-project homes with
  research/drafts/final_reports/supporting categories.
* ``create_artifact_tools`` — thin agent tools so the
  normal AgentLoop drives artifact work (no duplicate
  orchestration).
"""

from afnan_ai.artifacts.builders import (
    ContentBuilder,
    CsvBuilder,
    HtmlBuilder,
    ImageBuilder,
    JsonBuilder,
    MarkdownBuilder,
    PdfBuilder,
    SlidesBuilder,
    TextBuilder,
    builder_for,
)
from afnan_ai.artifacts.manager import (
    ArtifactError,
    ArtifactManager,
)
from afnan_ai.artifacts.models import (
    Artifact,
    ArtifactStatus,
    ArtifactType,
    ArtifactVersion,
    SourceRef,
    VerificationState,
)
from afnan_ai.artifacts.tools import create_artifact_tools
from afnan_ai.artifacts.verifier import (
    ArtifactVerdict,
    ArtifactVerifier,
)
from afnan_ai.artifacts.workspace import ArtifactWorkspace

__all__ = [
    "Artifact",
    "ArtifactError",
    "ArtifactManager",
    "ArtifactStatus",
    "ArtifactType",
    "ArtifactVerdict",
    "ArtifactVerifier",
    "ArtifactVersion",
    "ContentBuilder",
    "CsvBuilder",
    "HtmlBuilder",
    "ImageBuilder",
    "JsonBuilder",
    "MarkdownBuilder",
    "PdfBuilder",
    "SlidesBuilder",
    "SourceRef",
    "TextBuilder",
    "ArtifactWorkspace",
    "VerificationState",
    "builder_for",
    "create_artifact_tools",
]

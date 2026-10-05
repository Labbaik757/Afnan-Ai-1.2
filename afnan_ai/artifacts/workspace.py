"""Artifact workspace — per-project artifact homes.

    <workspace_root>/
      <project_id>/
        research/
        drafts/
        final_reports/
        supporting/
        artifacts.json      (metadata index)

Artifacts associate with task, project and goal ids; the
standard categories keep research, drafts, final reports
and supporting files separated.
"""

from __future__ import annotations

from pathlib import Path

CATEGORIES = (
    "research",
    "drafts",
    "final_reports",
    "supporting",
)


class ArtifactWorkspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(str(root)).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_id: str) -> Path:
        project_id = str(project_id or "default").strip() or "default"
        # Keep project ids filesystem-safe.
        safe = "".join(
            c if (c.isalnum() or c in "-_") else "_"
            for c in project_id
        )[:64] or "default"
        path = self.root / safe
        path.mkdir(parents=True, exist_ok=True)
        for category in CATEGORIES:
            (path / category).mkdir(parents=True, exist_ok=True)
        return path

    def category_dir(
        self, project_id: str, category: str
    ) -> Path:
        category = (
            str(category or "drafts").strip().lower()
            .replace(" ", "_")
        )
        if category not in CATEGORIES:
            category = "drafts"
        path = self.project_dir(project_id) / category
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifact_dir(
        self, project_id: str, category: str,
        artifact_id: str,
    ) -> Path:
        safe_id = "".join(
            c if (c.isalnum() or c in "-_") else "_"
            for c in str(artifact_id)
        )[:64]
        path = (
            self.category_dir(project_id, category) / safe_id
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def index_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "artifacts.json"

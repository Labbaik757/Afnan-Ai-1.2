"""Read-only query services — the UI/API layer.

UI clients (web, desktop, mobile, CLI, voice) consume these;
they never touch AgentLoop internals.  Responses are
structured, paginated and versioned.

API version: v1
"""

from __future__ import annotations

from typing import Any

from afnan_ai.activity import events as E
from afnan_ai.activity.timeline import (
    ActivityFilter,
    ActivityTimeline,
    ProgressTracker,
    summarize,
)

API_VERSION = "v1"


def _page(
    items: list[Any], *, limit: int, offset: int
) -> dict[str, Any]:
    total = len(items)
    limit = max(1, min(200, int(limit or 50)))
    offset = max(0, int(offset or 0))
    return {
        "api_version": API_VERSION,
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items[offset: offset + limit],
    }


class ActivityQueryService:
    """Search / filter / paginate activity events."""

    def __init__(self, center: Any) -> None:
        self.center = center

    def _all(
        self, task_id: str = "", limit: int = 2000
    ) -> list[E.ActivityEvent]:
        # Replay from the store (bounded).
        return self.center.store.replay(
            after_seq=0, task_id=task_id, limit=limit
        )

    def search(
        self, flt: ActivityFilter
    ) -> dict[str, Any]:
        matched = [
            e.to_dict()
            for e in self._all(task_id=flt.task_id)
            if flt.matches(e)
        ]
        matched.sort(key=lambda d: d["seq"])
        return _page(
            matched, limit=flt.limit, offset=flt.offset
        )

    def timeline(self, task_id: str) -> dict[str, Any]:
        events = self._all(task_id=task_id)
        tl = ActivityTimeline(events)
        result = tl.to_dict()
        result["api_version"] = API_VERSION
        result["task_id"] = task_id
        return result

    def summary(
        self,
        task_id: str,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        events = self._all(task_id=task_id)
        result = summarize(
            task_id, events, progress
        ).to_dict()
        result["api_version"] = API_VERSION
        return result


class TaskQueryService:
    """Task listing, progress and status (read-only)."""

    def __init__(
        self,
        *,
        center: Any,
        task_manager: Any,
        progress: ProgressTracker | None = None,
    ) -> None:
        self.center = center
        self.tasks = task_manager
        self.progress = progress

    def list(
        self,
        *,
        status: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        tasks = self.tasks.list()
        items = []
        for t in tasks:
            d = t.to_dict() if hasattr(t, "to_dict") else dict(t)
            if status and d.get("status") != status:
                continue
            task_id = d.get("task_id", "")
            d["live_status"] = self.center.status.get(task_id)
            if self.progress is not None:
                p = self.progress.get(task_id)
                if p is not None:
                    d["progress"] = p.to_dict()
            # Never expose internals.
            d.pop("checkpoint_ref", None)
            items.append(d)
        items.sort(
            key=lambda d: d.get("updated_at", ""), reverse=True
        )
        return _page(items, limit=limit, offset=offset)

    def get(self, task_id: str) -> dict[str, Any] | None:
        task = self.tasks.get(task_id)
        if task is None:
            return None
        d = (
            task.to_dict()
            if hasattr(task, "to_dict")
            else dict(task)
        )
        d["api_version"] = API_VERSION
        d["live_status"] = self.center.status.get(task_id)
        if self.progress is not None:
            p = self.progress.get(task_id)
            if p is not None:
                d["progress"] = p.to_dict()
        d.pop("checkpoint_ref", None)
        return d


class ArtifactQueryService:
    """Artifacts per task — ArtifactManager stays source of truth."""

    def __init__(
        self,
        *,
        center: Any,
        artifact_manager: Any,
    ) -> None:
        self.center = center
        self.artifacts = artifact_manager

    def for_task(
        self, task_id: str, *, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        names: list[str] = []
        # Prefer the progress view (from artifact_created events).
        if (
            self.center.progress is not None
            and self.center.progress.get(task_id) is not None
        ):
            names = list(
                self.center.progress.get(task_id).artifacts
            )
        items = []
        for name in names:
            try:
                record = self.artifacts.read(name)
            except Exception:
                continue
            d = (
                record.to_dict()
                if hasattr(record, "to_dict")
                else dict(record)
            )
            items.append(
                {
                    "name": d.get("name", name),
                    "type": d.get("type", ""),
                    "created_at": d.get("created_at", ""),
                    "task_id": task_id,
                    "version": d.get("version", 1),
                    "verification": d.get(
                        "verification", "unverified"
                    ),
                }
            )
        return _page(items, limit=limit, offset=offset)

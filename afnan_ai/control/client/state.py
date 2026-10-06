"""In-memory sanitized snapshot of remote agent state.

``RemoteStateStore`` keeps the latest sanitized agent view and task
list plus applies ``task.*`` stream events to the cached tasks, so a
client can render current state without re-fetching everything after
every event.  Secrets are stripped on the way in: keys that look
like credentials are dropped recursively, and tokens are never
stored here.
"""

from __future__ import annotations

import copy
from typing import Any

from afnan_ai.control.models import PROTOCOL_VERSION

_SECRET_KEY_HINTS = (
    "token",
    "secret",
    "password",
    "api_key",
    "apikey",
    "credential",
    "private_key",
    "auth_header",
    "authorization",
)


def _looks_secret(key: str) -> bool:
    lowered = key.lower()
    return any(hint in lowered for hint in _SECRET_KEY_HINTS)


def sanitize(value: Any) -> Any:
    """Recursively drop secret-looking keys from a value."""
    if isinstance(value, dict):
        return {
            k: sanitize(v)
            for k, v in value.items()
            if not (isinstance(k, str) and _looks_secret(k))
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


class RemoteStateStore:
    """Versioned, sanitized, in-memory remote state."""

    def __init__(self) -> None:
        self._agent: dict[str, Any] = {}
        self._tasks: dict[str, dict[str, Any]] = {}

    # -- updates -----------------------------------------------------------

    def update_agent(self, view: dict[str, Any]) -> None:
        """Replace the cached agent view (sanitized)."""
        if isinstance(view, dict):
            self._agent = sanitize(view)

    def update_tasks(self, views: list[dict[str, Any]]) -> None:
        """Replace the cached task list (sanitized)."""
        self._tasks = {}
        for view in views or []:
            if not isinstance(view, dict):
                continue
            clean = sanitize(view)
            task_id = str(clean.get("task_id", ""))
            if task_id:
                self._tasks[task_id] = clean

    def update_event(self, event: dict[str, Any]) -> None:
        """Fold a stream event into the cached state.

        ``task.*`` events update the matching cached task;
        ``agent.status`` events refresh the agent view.  Events
        without a task payload are ignored.
        """
        if not isinstance(event, dict):
            return
        category = str(event.get("category", ""))
        data = event.get("data")
        if not isinstance(data, dict):
            return
        if category.startswith("task."):
            task = data.get("task")
            if not isinstance(task, dict):
                return
            clean = sanitize(task)
            task_id = str(clean.get("task_id", ""))
            if not task_id:
                return
            if category == "task.created":
                self._tasks[task_id] = clean
            elif task_id in self._tasks:
                self._tasks[task_id].update(clean)
            else:
                self._tasks[task_id] = clean
        elif category == "agent.status":
            agent = data.get("agent")
            if isinstance(agent, dict):
                self._agent = sanitize(agent)

    # -- access --------------------------------------------------------------

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Return a copy of one cached task, or None."""
        task = self._tasks.get(str(task_id))
        return copy.deepcopy(task) if task is not None else None

    def snapshot(self) -> dict[str, Any]:
        """Return the versioned sanitized snapshot."""
        return {
            "protocol_version": PROTOCOL_VERSION,
            "agent": copy.deepcopy(self._agent),
            "tasks": copy.deepcopy(list(self._tasks.values())),
        }

    def clear(self) -> None:
        """Drop all cached state."""
        self._agent = {}
        self._tasks = {}

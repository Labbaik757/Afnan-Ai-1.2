"""Browser history and task-session management.

The BrowserController records every navigation event (navigate, back,
forward, tab create/select/close) in a bounded, redacted history.
:class:`SessionManager` exposes that history as structured session
state that tools can record into AgentState, and can save / load /
restore sessions as JSON so a later task can recover the previous
task's browsing context (tabs, purposes, visited URLs).

Session data never contains credentials or tokens: URLs are redacted
when recorded, cookies and storage are never read, and restoring a
session only re-opens tab URLs on a best-effort basis — a page that no
longer exists is reported, never faked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["SessionManager"]


class SessionManager:
    """Structured view over the controller's browsing session."""

    def __init__(self, controller: Any) -> None:
        self._controller = controller

    # -- queries -----------------------------------------------------------

    def history(
        self, tab_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        entries = self._controller.session_history()
        if tab_id:
            entries = [e for e in entries if e["tab_id"] == tab_id]
        return entries[-max(1, limit):]

    def current(self) -> dict[str, Any]:
        tabs = self._controller.list_tabs()
        active = self._controller.current_page()
        return {
            "tabs": tabs,
            "active_tab_id": active.get("tab_id"),
            "active_url": active.get("url"),
            "active_title": active.get("title"),
            "history_entries": len(self._controller.session_history()),
        }

    def export(self) -> dict[str, Any]:
        """A JSON-serialisable snapshot of the whole session."""
        return {
            "tabs": self._controller.list_tabs(),
            "history": self._controller.session_history(),
        }

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> dict[str, Any]:
        data = self.export()
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "path": str(target),
            "tabs": len(data["tabs"]),
            "history_entries": len(data["history"]),
        }

    def load(self, path: str) -> dict[str, Any]:
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"No saved session at {source}.")
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "tabs" not in data:
            raise ValueError("Saved session file is malformed.")
        return data

    def restore(self, path: str) -> dict[str, Any]:
        """Re-open the saved session's tabs (best effort).

        Each tab is recreated with its URL and purpose; pages that fail
        to load are listed under ``failed`` instead of aborting the
        whole restore — an expired/gone session degrades gracefully.
        """
        return self.restore_data(self.load(path))

    def restore_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Re-open tabs from an in-memory session snapshot."""
        restored: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        for tab in data.get("tabs") or []:
            url = str(tab.get("url") or "")
            if not url or url == "about:blank":
                continue
            try:
                created = self._controller.new_tab(
                    url, purpose=str(tab.get("purpose") or "")
                )
                restored.append(created)
            except Exception as exc:  # best effort per tab
                failed.append({"url": url, "error": str(exc)})
        return {
            "restored_tabs": restored,
            "failed": failed,
            "history_entries_recovered": len(data.get("history") or []),
        }

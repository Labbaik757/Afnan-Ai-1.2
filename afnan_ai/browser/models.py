"""Afnan browser data models — engine-independent, serializable.

These are the runtime's own records of browser state.  They
deliberately know nothing about Playwright (or any engine):
a future Afnan Chromium adapter produces exactly the same
models.  Everything here round-trips through ``to_dict`` /
``from_dict`` so records can live in AgentState, checkpoints
and session files — engine handles never do.

URLs and free text are redacted on the way in (credentials in
query strings, tokens, ... never persist).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from afnan_ai.redaction import redact_text


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_url(url: str) -> str:
    return redact_text(str(url or ""))


@dataclass
class BrowserProfile:
    """One isolated browser profile (own storage, own tabs)."""

    profile_id: str
    name: str
    storage_dir: str = ""
    preferences: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    persistent: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "storage_dir": self.storage_dir,
            "preferences": dict(self.preferences),
            "created_at": self.created_at,
            "persistent": self.persistent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrowserProfile":
        return cls(
            profile_id=str(data.get("profile_id") or data.get("name") or ""),
            name=str(data.get("name") or ""),
            storage_dir=str(data.get("storage_dir") or ""),
            preferences=dict(data.get("preferences") or {}),
            created_at=str(data.get("created_at") or _now()),
            persistent=bool(data.get("persistent", True)),
        )


@dataclass
class BrowserTab:
    """One tab as the runtime tracks it (engine mirror)."""

    tab_id: str
    url: str = ""
    title: str = ""
    profile_id: str = "default"
    purpose: str = ""
    active: bool = False
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tab_id": self.tab_id,
            "url": _clean_url(self.url),
            "title": self.title,
            "profile_id": self.profile_id,
            "purpose": self.purpose,
            "active": self.active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrowserTab":
        return cls(
            tab_id=str(data.get("tab_id") or ""),
            url=_clean_url(str(data.get("url") or "")),
            title=str(data.get("title") or ""),
            profile_id=str(data.get("profile_id") or "default"),
            purpose=str(data.get("purpose") or ""),
            active=bool(data.get("active", False)),
            created_at=str(data.get("created_at") or _now()),
            updated_at=str(data.get("updated_at") or _now()),
        )


@dataclass
class BrowserWindow:
    """A browser window grouping tabs (one window today)."""

    window_id: str = "main"
    tab_ids: list[str] = field(default_factory=list)
    profile_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "tab_ids": list(self.tab_ids),
            "profile_id": self.profile_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrowserWindow":
        return cls(
            window_id=str(data.get("window_id") or "main"),
            tab_ids=[str(t) for t in data.get("tab_ids") or []],
            profile_id=str(data.get("profile_id") or "default"),
        )


@dataclass
class BrowserPage:
    """A snapshot view of one page (URL/title/content summary)."""

    tab_id: str
    url: str = ""
    title: str = ""
    text_excerpt: str = ""
    loading: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tab_id": self.tab_id,
            "url": _clean_url(self.url),
            "title": self.title,
            "text_excerpt": redact_text(self.text_excerpt)[:500],
            "loading": self.loading,
        }


@dataclass
class BrowserElement:
    """One interactive element (mirrors controller ElementInfo)."""

    ref: str
    tag: str = ""
    text: str = ""
    visible: bool = True
    enabled: bool = True
    editable: bool = False
    value: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "tag": self.tag,
            "text": redact_text(self.text)[:200],
            "visible": self.visible,
            "enabled": self.enabled,
            "editable": self.editable,
            "value": self.value,
        }


@dataclass
class BrowserObservation:
    """A structured observation of one tab at one moment."""

    tab_id: str
    url: str = ""
    title: str = ""
    elements: list[BrowserElement] = field(default_factory=list)
    page_changed: bool = False
    observed_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tab_id": self.tab_id,
            "url": _clean_url(self.url),
            "title": self.title,
            "elements": [e.to_dict() for e in self.elements],
            "page_changed": self.page_changed,
            "observed_at": self.observed_at,
        }


@dataclass
class BrowserAction:
    """One action performed (or attempted) through the runtime."""

    action: str
    tab_id: str = ""
    url: str = ""
    at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "tab_id": self.tab_id,
            "url": _clean_url(self.url),
            "at": self.at,
        }


@dataclass
class BrowserResult:
    """Structured outcome of a runtime-level operation."""

    success: bool
    action: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "action": self.action,
            "message": self.message,
            "data": self.data,
            "error": self.error,
        }


@dataclass
class BrowserSession:
    """The runtime's recoverable record of one browser session.

    This — never engine objects — is what persists across
    restarts: session/profile/tab identities, URLs and titles,
    enough to identify recoverable tabs after a crash.
    """

    session_id: str
    profile_id: str = "default"
    status: str = "stopped"  # stopped | running | crashed
    tabs: list[BrowserTab] = field(default_factory=list)
    window: BrowserWindow = field(default_factory=BrowserWindow)
    started_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    adapter_name: str = ""

    def touch(self) -> None:
        self.updated_at = _now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "profile_id": self.profile_id,
            "status": self.status,
            "tabs": [t.to_dict() for t in self.tabs],
            "window": self.window.to_dict(),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "adapter_name": self.adapter_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrowserSession":
        return cls(
            session_id=str(data.get("session_id") or ""),
            profile_id=str(data.get("profile_id") or "default"),
            status=str(data.get("status") or "stopped"),
            tabs=[
                BrowserTab.from_dict(t)
                for t in data.get("tabs") or []
            ],
            window=BrowserWindow.from_dict(data.get("window") or {}),
            started_at=str(data.get("started_at") or _now()),
            updated_at=str(data.get("updated_at") or _now()),
            adapter_name=str(data.get("adapter_name") or ""),
        )

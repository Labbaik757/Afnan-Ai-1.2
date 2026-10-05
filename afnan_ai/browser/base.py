"""Browser control types and structured errors.

These are the shared types of the BrowserController system: tab
and page snapshots (always plain, JSON-serializable dicts via
``to_dict``) and the structured error form every browser failure
uses.  No browser operation in Afnan crashes with a bare driver
exception or pretends to have succeeded: failures come back as a
:class:`BrowserError` record (carried by :class:`BrowserException`),
and the browser Tools translate them into structured ToolResults.

Error codes:

* ``browser_unavailable`` — no usable browser backend (e.g.
  Playwright or its browser binaries are not installed)
* ``connection_failed`` — connecting to a running browser failed
* ``browser_not_started`` — the operation needs a running browser
* ``invalid_tab`` — unknown tab id, or no tab is open
* ``invalid_url`` — the URL is empty or unusable
* ``navigation_failed`` — the browser could not load the page
* ``operation_failed`` — any other browser-side failure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BrowserErrorCode(str, Enum):
    BROWSER_UNAVAILABLE = "browser_unavailable"
    CONNECTION_FAILED = "connection_failed"
    BROWSER_NOT_STARTED = "browser_not_started"
    INVALID_TAB = "invalid_tab"
    INVALID_URL = "invalid_url"
    NAVIGATION_FAILED = "navigation_failed"
    OPERATION_FAILED = "operation_failed"


@dataclass
class BrowserError:
    """Structured, serializable description of a browser failure."""

    code: BrowserErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": dict(self.details),
        }


class BrowserException(Exception):
    """Exception carrying a structured :class:`BrowserError`."""

    default_code = BrowserErrorCode.OPERATION_FAILED

    def __init__(
        self,
        message: str,
        *,
        code: BrowserErrorCode | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.error = BrowserError(
            code=code or self.default_code,
            message=message,
            details=dict(details or {}),
        )
        super().__init__(message)

    @property
    def code(self) -> BrowserErrorCode:
        return self.error.code

    def to_dict(self) -> dict[str, Any]:
        return self.error.to_dict()


@dataclass
class TabInfo:
    """Snapshot of one browser tab."""

    tab_id: str
    url: str = ""
    title: str = ""
    active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tab_id": self.tab_id,
            "url": self.url,
            "title": self.title,
            "active": bool(self.active),
        }


@dataclass
class PageState:
    """Snapshot of what a tab is currently showing."""

    tab_id: str
    url: str = ""
    title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tab_id": self.tab_id,
            "url": self.url,
            "title": self.title,
        }

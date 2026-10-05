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
* ``element_not_found`` — no element matches the locator
* ``invalid_locator`` — no usable way to find the element was given
* ``invalid_element`` — element exists but cannot be used this way
  (hidden, disabled, not editable, not a <select>, ...)
* ``stale_element`` — the element reference belongs to an older
  page (the page changed since it was found); find it again
* ``timeout`` — the browser did not complete the action in time
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
    RATE_LIMITED = "rate_limited"
    HUMAN_REQUIRED = "human_required"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_DENIED = "approval_denied"
    ELEMENT_NOT_FOUND = "element_not_found"
    INVALID_LOCATOR = "invalid_locator"
    INVALID_ELEMENT = "invalid_element"
    STALE_ELEMENT = "stale_element"
    TIMEOUT = "timeout"
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
    purpose: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tab_id": self.tab_id,
            "url": self.url,
            "title": self.title,
            "active": bool(self.active),
            "purpose": self.purpose,
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


@dataclass
class ElementInfo:
    """Identity and basic properties of one page element.

    ``ref`` is the controller-issued reference (``el_1``, ``el_2``,
    ...) that later interactions can target; it stays valid until
    the page changes (navigation/reload), after which using it is
    a structured ``stale_element`` error.
    """

    ref: str
    tab_id: str
    tag: str = ""
    text: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    enabled: bool = True
    editable: bool = False
    value: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "tab_id": self.tab_id,
            "tag": self.tag,
            "text": self.text,
            "attributes": dict(self.attributes),
            "visible": bool(self.visible),
            "enabled": bool(self.enabled),
            "editable": bool(self.editable),
            "value": self.value,
        }

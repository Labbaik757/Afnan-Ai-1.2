"""Structured errors for the Computer Use layer."""

from __future__ import annotations

from typing import Any


class ComputerError(Exception):
    """A structured computer-use failure (never a crash)."""

    CODES = frozenset({
        "unsupported_operation", "backend_unavailable",
        "window_not_found", "application_not_found",
        "element_not_found", "stale_element", "window_mismatch",
        "low_confidence", "approval_required", "approval_denied",
        "action_failed", "timeout", "invalid_arguments",
        "application_crashed", "file_error",
    })

    def __init__(
        self, code: str, message: str, **details: Any
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            **self.details,
        }

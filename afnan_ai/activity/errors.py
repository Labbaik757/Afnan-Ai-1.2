"""Structured, user-safe error presentation.

Technical exceptions are never dumped to the user.  Each
known failure mode maps to an error code with a friendly
message, a recoverability flag and a suggested action.
Unknown errors degrade to a generic, safe shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from afnan_ai.redaction import redact_text


@dataclass
class PresentedError:
    error_code: str
    user_message: str
    technical_summary: str = ""
    recoverable: bool = True
    recovery_status: str = "none"  # none|in_progress|recovered|failed
    suggested_action: str = ""
    related_step: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# error_code → (user_message, suggested_action)
_ERROR_CATALOG: dict[str, tuple[str, str]] = {
    "BROWSER_ELEMENT_NOT_FOUND": (
        "Required button nahi mila. Agent alternative "
        "locator se dobara try kar raha hai.",
        "Agar ye bar-bar ho to page dobara load karke retry karein.",
    ),
    "BROWSER_NAVIGATION_FAILED": (
        "Page khul nahi saka. Agent dobara try kar raha hai.",
        "Apna internet connection check karein.",
    ),
    "TOOL_TIMEOUT": (
        "Ye step time par complete nahi hua.",
        "Kuch der ruk kar retry karein.",
    ),
    "APPROVAL_DENIED": (
        "Aap ne is action ki approval deny kar di.",
        "Task ko naye instructions ke saath dobara shuru karein.",
    ),
    "APPROVAL_EXPIRED": (
        "Approval request ka waqt khatam ho gaya.",
        "Task resume karke dobara approval dein.",
    ),
    "PERMISSION_DENIED": (
        "Is action ki permission nahi hai.",
        "Task owner ya policy check karein.",
    ),
    "RESOURCE_LIMIT": (
        "Resource limit cross ho gayi (CPU/memory/tabs).",
        "Heavy tabs band karke retry karein.",
    ),
    "WORKSPACE_UNAVAILABLE": (
        "Task ka workspace abhi available nahi.",
        "Workspace recover hone ka intezar karein.",
    ),
    "CHECKPOINT_CORRUPT": (
        "Saved checkpoint kharab hai; wahan se resume nahi ho sakta.",
        "Task ko shuru se retry karein.",
    ),
    "LLM_UNAVAILABLE": (
        "Language model se rabta nahi ho saka.",
        "Ollama/model running hai ye check karein.",
    ),
    "STT_FAILED": (
        "Awaz samajh nahi aayi.",
        "Saaf awaz mein dobara bolein.",
    ),
    "EMERGENCY_STOP": (
        "Safety stop lagaya gaya; execution foran rok di gayi.",
        "Activity dekh kar faisla karein ke resume karna hai ya nahi.",
    ),
    "UNKNOWN": (
        "Kuch unexpected hua; agent recover karne ki koshish kar raha hai.",
        "Agar bar-bar ho to error details dekhein.",
    ),
}


def present_error(
    code: str,
    *,
    technical: str = "",
    related_step: str = "",
    recovery_status: str = "none",
    recoverable: bool = True,
) -> PresentedError:
    """Build a user-safe error from a code + redacted technical note."""
    code = str(code or "UNKNOWN").upper()
    user_message, suggested = _ERROR_CATALOG.get(
        code, _ERROR_CATALOG["UNKNOWN"]
    )
    return PresentedError(
        error_code=code,
        user_message=user_message,
        technical_summary=redact_text(str(technical))[:300],
        recoverable=recoverable,
        recovery_status=recovery_status,
        suggested_action=suggested,
        related_step=str(related_step)[:120],
    )


def infer_code(error: Any) -> str:
    """Best-effort code from an exception or error dict."""
    if isinstance(error, dict):
        code = str(
            error.get("code") or error.get("error_code") or ""
        ).upper()
        if code:
            return code
        blob = str(error).lower()
    else:
        blob = f"{type(error).__name__}: {error}".lower()
    for known in _ERROR_CATALOG:
        if known.lower() in blob:
            return known
    if "timeout" in blob:
        return "TOOL_TIMEOUT"
    if "approv" in blob and "den" in blob:
        return "APPROVAL_DENIED"
    if "permission" in blob or "forbidden" in blob:
        return "PERMISSION_DENIED"
    return "UNKNOWN"

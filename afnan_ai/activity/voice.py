"""Voice command parsing for ActivityCenter controls.

Natural commands map to the *same* ControlCommand pipeline
the UI uses — voice never bypasses authorization:

    "Task pause karo."        → pause(task_id=current)
    "Task resume karo."       → resume(task_id=current)
    "Is approval ko deny karo." → deny(approval_id=latest pending)
    "Current task ka status batao." → status query (not a command)
    "Task kyun fail hua?"     → error query (not a command)
    "Last successful step kya tha?" → progress query

Commands need a task context (current task id); queries need
read services.  Parsing is keyword-based and deterministic —
no LLM required, no prompt-injection surface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from afnan_ai.activity.control import (
    APPROVE,
    CANCEL,
    DENY,
    EMERGENCY_STOP,
    PAUSE,
    RESUME,
    RETRY,
    STOP,
    ControlCommand,
)


@dataclass
class VoiceIntent:
    kind: str  # command|status_query|error_query|progress_query|unknown
    command: ControlCommand | None = None
    text: str = ""


_PATTERNS: list[tuple[str, str]] = [
    (r"\bpause\b", PAUSE),
    (r"\brok", PAUSE),  # rok do / roko
    (r"\bresume\b", RESUME),
    (r"\bcontinue\b", RESUME),
    (r"\bstop\b", STOP),
    (r"\bcancel\b", CANCEL),
    (r"\bretry\b|dobara", RETRY),
    (r"\bapprove\b|manzoor|ijazat", APPROVE),
    (r"\bdeny\b|reject|inkaar", DENY),
    (r"\bemergency", EMERGENCY_STOP),
]

_STATUS_PATTERNS = r"status|kya ho raha|kahan (hai|tak)|haal"
_ERROR_PATTERNS = r"kyun fail|kyun nahi|error|masla|problem"
_PROGRESS_PATTERNS = r"last .*step|akhri step|kitna hua|progress"


def parse_voice_command(
    text: str,
    *,
    current_task_id: str = "",
    actor: str = "user",
) -> VoiceIntent:
    """Parse a voice utterance into an intent (deterministic)."""
    lowered = str(text or "").lower()

    for pattern, kind in (
        (_STATUS_PATTERNS, "status_query"),
        (_ERROR_PATTERNS, "error_query"),
        (_PROGRESS_PATTERNS, "progress_query"),
    ):
        if re.search(pattern, lowered):
            return VoiceIntent(
                kind=kind, text=str(text)[:300]
            )

    for pattern, command in _PATTERNS:
        if re.search(pattern, lowered):
            return VoiceIntent(
                kind="command",
                command=ControlCommand(
                    command=command,
                    task_id=current_task_id,
                    reason=f"voice: {str(text)[:120]}",
                    actor=actor,
                ),
                text=str(text)[:300],
            )
    return VoiceIntent(kind="unknown", text=str(text)[:300])


def answer_query(
    intent: VoiceIntent,
    *,
    task_query: Any,
    activity_query: Any,
    task_id: str = "",
) -> str:
    """Answer a voice status/error/progress query in brief prose."""
    tid = task_id or ""
    if intent.kind == "status_query" and tid:
        task = task_query.get(tid)
        if task is None:
            return "Koi aisa task nahi mila."
        status = task.get("live_status", "IDLE")
        prog = (task.get("progress") or {}).get(
            "current_step", ""
        )
        return (
            f"Task {status} hai."
            + (f" Abhi: {prog}." if prog else "")
        ).strip()
    if intent.kind == "error_query" and tid:
        events = activity_query.search(
            __import__(
                "afnan_ai.activity.timeline",
                fromlist=["ActivityFilter"],
            ).ActivityFilter(
                task_id=tid,
                types=["action_failed", "task_failed"],
                limit=1,
            )
        )["items"]
        if not events:
            return "Koi failure record nahi mili."
        d = events[0].get("details", {})
        return str(
            d.get("user_message")
            or events[0].get("summary", "Wajah maloom nahi.")
        )[:300]
    if intent.kind == "progress_query" and tid:
        task = task_query.get(tid)
        prog = (task or {}).get("progress") or {}
        done = prog.get("completed_steps", [])
        if not done:
            return "Abhi tak koi step complete nahi hua."
        return f"Akhri kamyab step: {done[-1]}."
    return "Samajh nahi aaya; dobara kahein."

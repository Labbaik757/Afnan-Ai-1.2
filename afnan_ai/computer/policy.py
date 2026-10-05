"""Sensitivity classification + human-approval gate for
desktop actions.

Mirrors the browser layer's fail-safe philosophy: a sensitive
action runs only on an explicit human "yes"; with no approver
configured it does not run at all.  The approver receives
structured, redacted context (action, level, target summary)
— never typed secrets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

LEVELS = ("normal", "sensitive", "destructive", "approval_required")

_DESTRUCTIVE_HOTKEYS = frozenset({
    "alt+f4", "cmd+q", "command+q", "ctrl+alt+delete",
    "ctrl+alt+del", "alt+tab+delete",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_action(
    action: str, *, target: dict[str, Any] | None = None
) -> str:
    """Sensitivity level for a desktop action."""
    action = (action or "").lower()
    target = target or {}
    if target.get("password_field"):
        # Typing into a credential field: a human decides, and
        # the text itself is never logged.
        return "approval_required"
    if action in ("close_application", "close_window"):
        return "destructive"
    if action in ("minimize_window", "maximize_window",
                  "restore_window"):
        return "sensitive"
    if action == "hotkey":
        combo = str(target.get("combo", "")).lower().replace(" ", "")
        if combo in _DESTRUCTIVE_HOTKEYS:
            return "destructive"
        return "sensitive"
    if action == "key_press":
        key = str(target.get("key", "")).lower()
        if key in ("delete", "backspace"):
            return "sensitive"
        return "normal"
    if action in ("type",):
        return "sensitive"
    if action in ("click", "double_click", "right_click"):
        name = str(target.get("name", "")).lower()
        if any(word in name for word in (
            "delete", "remove", "close", "quit", "exit",
            "discard", "send", "pay", "purchase", "confirm",
        )):
            return "sensitive"
        return "normal"
    return "normal"


@dataclass
class GateDecision:
    action: str
    level: str
    outcome: str  # approved | denied | no_approver | not_required
    target: str = ""
    decided_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "level": self.level,
            "outcome": self.outcome,
            "target": self.target[:120],
            "decided_at": self.decided_at,
        }


Approver = Callable[[dict[str, Any]], bool]


class ComputerApprovalGate:
    """Fail-safe approval for sensitive desktop actions."""

    def __init__(self, approver: Approver | None = None):
        self.approver = approver
        self.decisions: list[GateDecision] = []

    def set_approver(self, approver: Approver | None) -> None:
        self.approver = approver

    def check(
        self,
        action: str,
        *,
        level: str,
        target_summary: str = "",
    ) -> GateDecision:
        if level == "normal":
            decision = GateDecision(
                action, level, "not_required", target_summary
            )
            return decision
        if self.approver is None:
            decision = GateDecision(
                action, level, "no_approver", target_summary
            )
            self.decisions.append(decision)
            return decision
        approved = bool(self.approver({
            "action": action,
            "level": level,
            "target": target_summary[:160],
            "kind": "computer_action",
        }))
        decision = GateDecision(
            action,
            level,
            "approved" if approved else "denied",
            target_summary,
        )
        self.decisions.append(decision)
        return decision

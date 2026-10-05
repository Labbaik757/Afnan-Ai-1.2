"""Central risk classification.

Every action the agent (or any subagent/skill/connector)
takes is classified into exactly one of:

* READ_ONLY        — observes, changes nothing
                     (webpage read, search, list)
* LOW_RISK_WRITE   — reversible, scoped writes
                     (draft file create, save draft note)
* SENSITIVE        — needs a human yes
                     (email/message send, publish)
* IRREVERSIBLE     — destructive or account-level
                     (delete important files, purchase,
                      account changes)

Classification is deterministic and central — no
subsystem keeps a private copy.  Unknown actions default
to SENSITIVE (fail cautious, never fail open).
"""

from __future__ import annotations

import re
from typing import Any

from afnan_ai.security.models import RiskLevel

# tool-name → risk.  Checked before the heuristics so
# explicit entries always win.
_TOOL_RISK: dict[str, RiskLevel] = {
    # -- reads ---------------------------------------------------------
    "browser_navigate": RiskLevel.READ_ONLY,
    "browser_read": RiskLevel.READ_ONLY,
    "browser_extract": RiskLevel.READ_ONLY,
    "browser_search": RiskLevel.READ_ONLY,
    "browser_observe": RiskLevel.READ_ONLY,
    "browser_screenshot": RiskLevel.READ_ONLY,
    "computer_observe": RiskLevel.READ_ONLY,
    "computer_locate": RiskLevel.READ_ONLY,
    "computer_screenshot": RiskLevel.READ_ONLY,
    "screen_observe": RiskLevel.READ_ONLY,
    "screen_describe": RiskLevel.READ_ONLY,
    "file_list": RiskLevel.READ_ONLY,
    "file_find_downloads": RiskLevel.READ_ONLY,
    "artifact_read": RiskLevel.READ_ONLY,
    "artifact_list": RiskLevel.READ_ONLY,
    "artifact_verify": RiskLevel.READ_ONLY,
    "connector_list": RiskLevel.READ_ONLY,
    "connector_capabilities": RiskLevel.READ_ONLY,
    "connector_health_check": RiskLevel.READ_ONLY,
    "memory_search": RiskLevel.READ_ONLY,
    "memory_read": RiskLevel.READ_ONLY,
    # -- low-risk writes ----------------------------------------------
    "artifact_create": RiskLevel.LOW_RISK_WRITE,
    "artifact_update": RiskLevel.LOW_RISK_WRITE,
    "artifact_add_source": RiskLevel.LOW_RISK_WRITE,
    "file_save_text": RiskLevel.LOW_RISK_WRITE,
    "file_create_folder": RiskLevel.LOW_RISK_WRITE,
    "browser_download": RiskLevel.LOW_RISK_WRITE,
    "note_save": RiskLevel.LOW_RISK_WRITE,
    "memory_write": RiskLevel.LOW_RISK_WRITE,
    # -- sensitive ------------------------------------------------------
    "browser_click": RiskLevel.SENSITIVE,
    "browser_type": RiskLevel.SENSITIVE,
    "browser_submit": RiskLevel.SENSITIVE,
    "computer_click": RiskLevel.SENSITIVE,
    "computer_type": RiskLevel.SENSITIVE,
    "computer_key_press": RiskLevel.SENSITIVE,
    "computer_hotkey": RiskLevel.SENSITIVE,
    "file_open": RiskLevel.SENSITIVE,
    "connector_execute": RiskLevel.SENSITIVE,
    "artifact_export": RiskLevel.SENSITIVE,
    "email_send": RiskLevel.SENSITIVE,
    "message_send": RiskLevel.SENSITIVE,
    "skill_execute": RiskLevel.SENSITIVE,
    # -- irreversible ---------------------------------------------------
    "file_delete": RiskLevel.IRREVERSIBLE,
    "file_move": RiskLevel.IRREVERSIBLE,
    "artifact_delete": RiskLevel.IRREVERSIBLE,
    "browser_close_tab": RiskLevel.LOW_RISK_WRITE,
    "computer_close_application": RiskLevel.IRREVERSIBLE,
    "purchase": RiskLevel.IRREVERSIBLE,
    "payment_charge": RiskLevel.IRREVERSIBLE,
    "account_change": RiskLevel.IRREVERSIBLE,
    "connector_delete": RiskLevel.IRREVERSIBLE,
}

# Heuristic fallbacks for tools not in the table.
_READ_HINTS = (
    "read", "list", "search", "observe", "get", "check",
    "describe", "verify", "inspect", "view", "fetch",
    "screenshot", "locate", "find",
)
_WRITE_HINTS = (
    "create", "save", "write", "draft", "update",
    "download", "add",
)
_SENSITIVE_HINTS = (
    "send", "publish", "post", "submit", "click", "type",
    "press", "execute", "export", "share", "open",
)
_IRREVERSIBLE_HINTS = (
    "delete", "remove", "purchase", "charge", "pay",
    "terminate", "destroy", "wipe", "format",
)


def classify_action(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> RiskLevel:
    """Classify one action.  Deterministic; unknown tools
    are SENSITIVE (fail cautious)."""
    name = str(tool_name or "").strip().lower()
    if name in _TOOL_RISK:
        risk = _TOOL_RISK[name]
    else:
        risk = _heuristic(name)
    # Argument-driven escalation: some tools change risk
    # with what they are asked to do.
    args = arguments or {}
    risk = _escalate(name, args, risk)
    return risk


def _heuristic(name: str) -> RiskLevel:
    for hint in _IRREVERSIBLE_HINTS:
        if hint in name:
            return RiskLevel.IRREVERSIBLE
    for hint in _SENSITIVE_HINTS:
        if hint in name:
            return RiskLevel.SENSITIVE
    for hint in _WRITE_HINTS:
        if hint in name:
            return RiskLevel.LOW_RISK_WRITE
    for hint in _READ_HINTS:
        if hint in name:
            return RiskLevel.READ_ONLY
    return RiskLevel.SENSITIVE


def _escalate(
    name: str, args: dict[str, Any], risk: RiskLevel
) -> RiskLevel:
    text = " ".join(
        str(v).lower() for v in args.values()
        if isinstance(v, (str, int, float))
    )
    # Deleting outside a sandbox/draft area is worse.
    if "delete" in name or "remove" in name:
        if any(
            p in text
            for p in ("/etc", "c:\\windows", "system32",
                      "/home", "documents")
        ):
            return RiskLevel.IRREVERSIBLE
    # Sending money or buying is always irreversible.
    if re.search(
        r"\b(pay|purchase|buy|charge|order)\b", text
    ) and "send" in name:
        return RiskLevel.IRREVERSIBLE
    # Credential-shaped arguments raise any write.
    if re.search(
        r"(password|passwd|api[_-]?key|secret|token)",
        text,
    ) and risk in (
        RiskLevel.LOW_RISK_WRITE, RiskLevel.READ_ONLY
    ):
        return RiskLevel.SENSITIVE
    return risk


def risk_for_capability(capability: str) -> RiskLevel:
    """Capabilities carry their risk: 'email.send' etc."""
    tail = str(capability or "").split(".")[-1].lower()
    mapping = {
        "read": RiskLevel.READ_ONLY,
        "list": RiskLevel.READ_ONLY,
        "search": RiskLevel.READ_ONLY,
        "observe": RiskLevel.READ_ONLY,
        "write": RiskLevel.LOW_RISK_WRITE,
        "create": RiskLevel.LOW_RISK_WRITE,
        "draft": RiskLevel.LOW_RISK_WRITE,
        "send": RiskLevel.SENSITIVE,
        "publish": RiskLevel.SENSITIVE,
        "execute": RiskLevel.SENSITIVE,
        "delete": RiskLevel.IRREVERSIBLE,
        "purchase": RiskLevel.IRREVERSIBLE,
        "charge": RiskLevel.IRREVERSIBLE,
    }
    return mapping.get(tail, RiskLevel.SENSITIVE)

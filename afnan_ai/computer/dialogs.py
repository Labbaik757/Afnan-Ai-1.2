"""Dialog detection and handling.

Detects confirmation / save / file-picker / auth / permission
/ error dialogs from accessibility metadata + window titles,
and plans the appropriate response.  Sensitive or
irreversible confirmations are never auto-answered: they go
through the approval gate.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from afnan_ai.computer.models import DialogInfo

# kind → title/text patterns
_KIND_PATTERNS: list[tuple[str, list[str]]] = [
    ("save", [r"\bsave\b", r"save as", r"\bsaving\b"]),
    ("file_picker", [r"\bopen\b.*\bfile\b", r"choose file",
                     r"select file", r"\bbrowse\b"]),
    ("confirmation", [r"\bare you sure\b", r"\bconfirm\b",
                      r"\bdelete\b", r"\bremove\b",
                      r"\bdiscard\b"]),
    ("auth", [r"\blog ?in\b", r"\bsign ?in\b", r"\bpassword\b",
              r"\bauthenticat"]),
    ("permission", [r"\ballow\b.*\baccess\b", r"\bpermission\b",
                    r"\bgrant\b"]),
    ("error", [r"\berror\b", r"\bfailed\b", r"\bcrashed\b",
               r"\bnot responding\b"]),
]

_SENSITIVE_PATTERNS = [
    r"\bdelete\b", r"\bremove\b", r"\bdiscard\b",
    r"\bformat\b", r"\bpermanently\b", r"\birreversible\b",
]


class DialogHandler:
    """Detect dialogs and decide safe responses."""

    def __init__(
        self,
        controller: Any,
        *,
        auto_dismiss_info: bool = True,
    ) -> None:
        self.controller = controller
        self.auto_dismiss_info = auto_dismiss_info

    # -- detection ---------------------------------------------------------
    def detect(
        self, observation: dict[str, Any] | None = None
    ) -> list[DialogInfo]:
        """Find dialogs in the current observation."""
        obs = observation or self.controller.observe()
        dialogs: list[DialogInfo] = []
        windows = obs.get("windows", []) or []
        elements = obs.get("elements", []) or []
        for w in windows:
            title = str(w.get("title", ""))
            role = str(w.get("role", "")).lower()
            if "dialog" not in role and not self._looks_dialog(
                title, elements, str(w.get("window_id", ""))
            ):
                continue
            kind = self._classify(title)
            text = self._dialog_text(elements, str(w.get("window_id", "")))
            buttons = self._dialog_buttons(
                elements, str(w.get("window_id", ""))
            )
            sensitive = any(
                re.search(p, f"{title} {text}", re.I)
                for p in _SENSITIVE_PATTERNS
            )
            dialogs.append(
                DialogInfo(
                    dialog_id="dlg-" + uuid.uuid4().hex[:8],
                    kind=kind,
                    title=title,
                    text=text,
                    window_id=str(w.get("window_id", "")),
                    app=str(w.get("app", "")),
                    buttons=buttons,
                    sensitive=sensitive,
                )
            )
        return dialogs

    def _looks_dialog(
        self, title: str, elements: list[dict], window_id: str
    ) -> bool:
        win_elements = [
            e for e in elements
            if str(e.get("window_id", "")) == window_id
        ]
        buttons = [
            e for e in win_elements
            if str(e.get("role", "")).lower() == "button"
        ]
        return len(buttons) >= 2 and bool(title.strip())

    def _classify(self, title: str) -> str:
        for kind, patterns in _KIND_PATTERNS:
            if any(
                re.search(p, title, re.I) for p in patterns
            ):
                return kind
        return "unknown"

    def _dialog_text(
        self, elements: list[dict], window_id: str
    ) -> str:
        parts = []
        for e in elements:
            if str(e.get("window_id", "")) != window_id:
                continue
            if str(e.get("role", "")).lower() in (
                "label", "text", "static_text",
            ):
                name = str(e.get("name", "")).strip()
                if name:
                    parts.append(name)
        return " ".join(parts)[:500]

    def _dialog_buttons(
        self, elements: list[dict], window_id: str
    ) -> list[str]:
        return [
            str(e.get("name", ""))
            for e in elements
            if str(e.get("window_id", "")) == window_id
            and str(e.get("role", "")).lower() == "button"
            and str(e.get("name", "")).strip()
        ][:8]

    # -- response planning ---------------------------------------------------
    def plan_response(
        self, dialog: DialogInfo
    ) -> dict[str, Any]:
        """Decide what to do — never auto-confirm sensitive ones."""
        if dialog.sensitive:
            return {
                "action": "needs_approval",
                "reason": (
                    f"Sensitive {dialog.kind} dialog: "
                    f"{dialog.title[:80]}"
                ),
            }
        if dialog.kind == "error":
            return {
                "action": "dismiss",
                "button": self._pick_button(
                    dialog, ["ok", "close", "dismiss"]
                ),
            }
        if (
            dialog.kind == "info"
            and self.auto_dismiss_info
        ):
            return {
                "action": "dismiss",
                "button": self._pick_button(
                    dialog, ["ok", "close"]
                ),
            }
        return {
            "action": "report",
            "reason": (
                f"{dialog.kind} dialog needs a decision: "
                f"{dialog.title[:80]}"
            ),
        }

    def _pick_button(
        self, dialog: DialogInfo, preferences: list[str]
    ) -> str | None:
        lowered = [b.lower() for b in dialog.buttons]
        for pref in preferences:
            for i, b in enumerate(lowered):
                if pref in b:
                    return dialog.buttons[i]
        return None

    def handle(
        self, dialog: DialogInfo
    ) -> dict[str, Any]:
        """Execute the planned response for one dialog."""
        plan = self.plan_response(dialog)
        if plan["action"] == "needs_approval":
            # Route through the controller's approval gate by
            # raising a structured approval request the
            # runtime surfaces to the user.
            from afnan_ai.computer.errors import ComputerError

            raise ComputerError(
                "approval_required",
                f"Sensitive dialog needs approval: "
                f"{dialog.title[:80]}",
            )
        if plan["action"] == "dismiss" and plan.get("button"):
            return self.controller.act(
                "click", text=plan["button"]
            )
        return {"handled": False, "plan": plan}

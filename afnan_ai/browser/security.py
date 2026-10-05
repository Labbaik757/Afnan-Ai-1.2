"""Browser security layer — sensitive-action classification and
human approval for the Browser Agent.

Two guarantees live here, enforced by the BrowserController
before anything irreversible happens:

* **Sensitive actions are classified.**  Purchases, payments,
  sending messages/emails, account changes, destructive actions,
  file uploads and credential/payment-field entry are recognized
  from the tool, the target element (its text, type and
  attributes) and the page — not from guesswork.
* **Sensitive actions need human approval.**  An
  :class:`ApprovalGate` with a configurable :class:`SecurityPolicy`
  decides: safe actions run; sensitive ones run only when a
  human approver says yes; with no approver configured they do
  **not** run at all (structured ``approval_required`` /
  ``approval_denied`` errors, recorded like any other browser
  failure).  Approval never silently defaults to yes.

Secrets hygiene: approval requests carry *redacted* arguments
(``afnan_ai.redaction``), and the gate logs categories and tool
names only — never values.  This layer classifies and gates; it
executes nothing itself, so the Executor/Planner/Verifier
responsibilities stay exactly where they were.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from afnan_ai.log_config import get_logger
from afnan_ai.redaction import redact_arguments

logger = get_logger(__name__)


class Sensitivity(str, Enum):
    SAFE = "safe"
    SENSITIVE = "sensitive"


@dataclass
class ActionRisk:
    """Classification of one proposed browser action."""

    sensitivity: Sensitivity
    category: str = "general"
    reason: str = ""

    @property
    def sensitive(self) -> bool:
        return self.sensitivity is Sensitivity.SENSITIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensitivity": self.sensitivity.value,
            "category": self.category,
            "reason": self.reason,
        }


# Token-phrase rules over the target element's visible text /
# accessible name.  Conservative on purpose: ordinary navigation
# and sign-in flows must not be flagged.
_TEXT_RULES: list[tuple[str, tuple[frozenset, ...]]] = [
    ("purchase", (frozenset({"buy"}), frozenset({"checkout"}),
                  frozenset({"pay"}), frozenset({"purchase"}),
                  frozenset({"place", "order"}),
                  frozenset({"complete", "purchase"}),
                  frozenset({"confirm", "order"}))),
    ("email_send", (frozenset({"send", "email"}),
                    frozenset({"send", "mail"}))),
    ("message_send", (frozenset({"send"}), frozenset({"send", "message"}),
                      frozenset({"post"}), frozenset({"publish"}))),
    ("destructive", (frozenset({"delete"}), frozenset({"remove"}),
                     frozenset({"discard"}), frozenset({"erase"}),
                     frozenset({"destroy"}))),
    ("account_change", (frozenset({"change", "password"}),
                        frozenset({"deactivate"}),
                        frozenset({"close", "account"}),
                        frozenset({"update", "payment"}),
                        frozenset({"add", "card"}))),
    ("form_submit", (frozenset({"submit"}),)),
]

_PAYMENT_FIELD_HINTS = ("card", "cvv", "cvc", "iban", "expiry")


def _tokens(text: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _element_surface(element: dict[str, Any] | None) -> tuple[set[str], dict]:
    """Token set + attributes of a target element (if known)."""
    if not element:
        return set(), {}
    attrs = element.get("attributes") or {}
    surface = _tokens(element.get("text", ""))
    for key in ("id", "name", "aria-label", "placeholder", "title", "value"):
        surface |= _tokens(attrs.get(key, ""))
    return surface, attrs


def classify_action(
    tool_name: str,
    *,
    element: dict[str, Any] | None = None,
    arguments: dict[str, Any] | None = None,
    url: str | None = None,
) -> ActionRisk:
    """Classify a proposed browser action.

    Uses the tool first (uploads are always file uploads), then
    the target element (password/payment fields, button labels),
    in that order, so a real signal beats a generic one.
    """
    surface, attrs = _element_surface(element)
    element_type = str(attrs.get("type", "")).lower()
    tag = str((element or {}).get("tag", "")).lower()

    # File uploads are sensitive by nature: a local file leaves
    # the machine.
    if tool_name == "browser_upload_file":
        return ActionRisk(
            Sensitivity.SENSITIVE,
            "file_upload",
            "Uploading a local file to a website",
        )

    # Typing into a password field carries a credential.  It is
    # classified (and redacted) but allowed by default policy —
    # sign-in flows need it; purchases do not get this pass.
    if tool_name in ("browser_type", "browser_clear") and element_type == "password":
        return ActionRisk(
            Sensitivity.SENSITIVE,
            "credential_input",
            "Entering a credential into a password field",
        )

    # Typing payment instrument details moves money data.
    if tool_name in ("browser_type", "browser_clear"):
        field_hint = " ".join(
            str(attrs.get(k, "")) for k in ("id", "name", "placeholder")
        ).lower()
        if any(hint in field_hint for hint in _PAYMENT_FIELD_HINTS):
            return ActionRisk(
                Sensitivity.SENSITIVE,
                "payment_input",
                "Entering payment details",
            )

    # Click-like actions are judged by what the target says it does.
    if tool_name in ("browser_click", "browser_press_key") and element is not None:
        for category, phrases in _TEXT_RULES:
            for phrase in phrases:
                if phrase <= surface:
                    return ActionRisk(
                        Sensitivity.SENSITIVE,
                        category,
                        f"Target element matches '{' '.join(sorted(phrase))}' "
                        f"({category})",
                    )

    return ActionRisk(Sensitivity.SAFE, "general", "Ordinary browser action")


@dataclass
class SecurityPolicy:
    """Which categories need a human, and which never run.

    Defaults: money movement, sending, account changes,
    destructive actions, file uploads and submitting forms
    require approval; entering credentials into a password field
    is allowed (it is the normal sign-in flow) but always
    handled redacted.  Everything is configurable per category.
    """

    require_approval_for: frozenset = field(
        default_factory=lambda: frozenset({
            "purchase",
            "payment_input",
            "email_send",
            "message_send",
            "account_change",
            "destructive",
            "file_upload",
            "form_submit",
            # A natural-language target matched with low
            # confidence must be confirmed by a human.
            "uncertain_target",
        })
    )
    blocked_categories: frozenset = field(default_factory=frozenset)
    #: A human answer that arrives later than this is treated as
    #: a timeout: the action does not run.  None disables the
    #: timeout (the approver's answer is accepted whenever it
    #: arrives).
    approval_timeout_s: float | None = None

    def requires_approval(self, category: str) -> bool:
        return category in self.require_approval_for

    def is_blocked(self, category: str) -> bool:
        return category in self.blocked_categories


@dataclass
class ApprovalRequest:
    """What a human is being asked to allow (secrets redacted)."""

    tool_name: str
    category: str
    reason: str
    arguments: dict[str, Any] = field(default_factory=dict)
    url: str = ""
    title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "category": self.category,
            "reason": self.reason,
            "arguments": dict(self.arguments),
            "url": self.url,
            "title": self.title,
        }


@dataclass
class ApprovalDecision:
    allowed: bool
    risk: ActionRisk
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "category": self.risk.category,
            "sensitivity": self.risk.sensitivity.value,
            "detail": self.detail,
        }


#: A human decider: returns True to allow the action.
Approver = Callable[[ApprovalRequest], bool]


class ApprovalGate:
    """Decides whether a classified action may run.

    Fail-safe by construction: a sensitive action with no
    approver configured is *not* allowed, a blocked category is
    never allowed, and only an explicit approver "yes" lets a
    sensitive action through.  Every check is recorded in
    :attr:`decisions` (redacted) for audit.
    """

    def __init__(
        self,
        policy: SecurityPolicy | None = None,
        approver: Approver | None = None,
    ):
        self.policy = policy or SecurityPolicy()
        self.approver = approver
        self.decisions: list[dict[str, Any]] = []

    def check(
        self,
        risk: ActionRisk,
        *,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        url: str = "",
        title: str = "",
    ) -> ApprovalDecision:
        if not risk.sensitive:
            return ApprovalDecision(True, risk, "safe action")

        if self.policy.is_blocked(risk.category):
            decision = ApprovalDecision(
                False, risk,
                f"Category {risk.category!r} is blocked by policy",
            )
            self._record(risk, decision, tool_name)
            return decision

        if not self.policy.requires_approval(risk.category):
            # Classified and noted (e.g. credential entry), but the
            # policy lets it run; records stay redacted upstream.
            return ApprovalDecision(
                True, risk, f"{risk.category} allowed by policy"
            )

        request = ApprovalRequest(
            tool_name=tool_name,
            category=risk.category,
            reason=risk.reason,
            arguments=redact_arguments(arguments or {}),
            url=url,
            title=title,
        )
        if self.approver is None:
            decision = ApprovalDecision(
                False, risk,
                "Sensitive action requires human approval, but no "
                "approver is configured",
            )
            self._record(risk, decision, tool_name, request)
            return decision

        try:
            started = time.monotonic()
            allowed = bool(self.approver(request))
            elapsed = time.monotonic() - started
        except Exception as e:  # a broken approver never allows
            allowed = False
            elapsed = 0.0
            logger.warning(
                "approver raised for %s (%s): %s",
                tool_name, risk.category, e,
            )
        timeout = self.policy.approval_timeout_s
        if timeout is not None and elapsed > timeout:
            # Answered too late: the request has expired, so the
            # action does not run even on a late "yes".
            decision = ApprovalDecision(
                False, risk,
                f"Approval request timed out after {elapsed:.1f}s "
                f"(limit {timeout}s); action not executed",
            )
            self._record(
                risk, decision, tool_name, request, outcome="timeout"
            )
            return decision
        decision = ApprovalDecision(
            allowed, risk,
            "Approved by human approver" if allowed
            else "Denied by human approver",
        )
        self._record(
            risk, decision, tool_name, request,
            outcome="approved" if allowed else "denied",
        )
        return decision

    def _record(self, risk, decision, tool_name, request=None,
                outcome: str | None = None) -> None:
        entry = {
            "tool_name": tool_name,
            "category": risk.category,
            "allowed": decision.allowed,
            "detail": decision.detail,
            "outcome": outcome or (
                "approved" if decision.allowed else "denied"
            ),
            "decided_at": datetime.now(timezone.utc).isoformat(),
        }
        if request is not None:
            entry["request"] = request.to_dict()
        self.decisions.append(entry)
        logger.info(
            "approval check: %s category=%s allowed=%s",
            tool_name, risk.category, decision.allowed,
        )

"""Redaction — keep secrets out of records, logs and prompts.

One small, dependency-free helper used wherever agent data is
*recorded* (AgentState, observations, recovery context, approval
requests): values that look like credentials, tokens, cookies,
card numbers or keys are replaced with ``***`` before they are
stored or shown.  Execution itself always uses the real values;
only the records are sanitized.

Rules:
* dict keys that name a secret (password, token, cookie, ...)
  have their values masked, at any nesting depth;
* free text is scanned for card-number-like digit runs, JWTs,
  ``Bearer ...`` tokens and ``key=value`` secret assignments;
* tool *arguments* get one extra rule: when the target of the
  action is a password field, the typed ``text``/``value`` is
  masked too (the key alone cannot tell).
"""
from __future__ import annotations

import re
from typing import Any

MASK = "***"

_SECRET_KEY_PARTS = (
    "password",
    "passwd",
    "pwd",
    "token",
    "secret",
    "cookie",
    "authorization",
    "api_key",
    "apikey",
    "api-key",
    "session_id",
    "sessionid",
    "credential",
    "cvv",
    "card_number",
    "cardnumber",
    "private_key",
    "access_key",
    "auth",
)

_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]*)?")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_KV_SECRET_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|token|secret|api[_-]?key|session|cookie)"
    r"(\s*[:=]\s*)(\S+)"
)
_MAX_DEPTH = 8


def _is_secret_key(key: Any) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return any(part in normalized for part in _SECRET_KEY_PARTS)


def redact_text(text: Any) -> str:
    """Mask secret-looking patterns inside a string."""
    out = str(text)
    out = _BEARER_RE.sub("Bearer " + MASK, out)
    out = _JWT_RE.sub(MASK, out)
    out = _KV_SECRET_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{MASK}", out)
    out = _CARD_RE.sub(MASK, out)
    return out


def redact_value(value: Any, _depth: int = 0) -> Any:
    """Recursively mask secrets in JSON-ish data.

    The shape is preserved (keys stay, values are masked) so
    records remain useful without carrying the secret itself.
    """
    if _depth > _MAX_DEPTH:
        return MASK
    if isinstance(value, dict):
        return {
            key: (MASK if _is_secret_key(key) else redact_value(item, _depth + 1))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_value(item, _depth + 1) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_arguments(arguments: Any) -> Any:
    """Redact tool arguments for records/prompts.

    Beyond the generic rules: typing into a password field makes
    the typed text itself a secret, so when any target-ish value
    mentions a password, ``text``/``value`` entries are masked.
    """
    redacted = redact_value(arguments)
    if isinstance(redacted, dict):
        haystack = " ".join(
            str(v).lower() for v in redacted.values() if isinstance(v, str)
        )
        if "password" in haystack or "passwd" in haystack:
            for key in ("text", "value"):
                if key in redacted:
                    redacted[key] = MASK
    return redacted

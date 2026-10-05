"""Central SecretRedactor.

One redaction authority for the whole runtime.  Secrets
(API keys, tokens, passwords, cookies, authorization
headers, private keys, sensitive env vars) are detected
and redacted consistently across:

* logs and errors
* screenshots metadata
* tool results
* trajectories
* audit events
* LLM context

Nothing secret-shaped ever leaves the vault boundary in
plaintext.
"""

from __future__ import annotations

import re
from typing import Any

from afnan_ai.redaction import (
    redact_arguments,
    redact_text,
    redact_value,
)

# Extra shapes beyond the base redactor's key-name scan.
_SECRET_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(
        r"\bAKIA[0-9A-Z]{16}\b"  # AWS access key id
    ),
    re.compile(
        r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"  # Slack tokens
    ),
    re.compile(
        r"\bghp_[0-9A-Za-z]{20,}\b"  # GitHub PAT
    ),
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
    ),
    re.compile(
        r"\bBearer\s+[0-9A-Za-z._~+/-]{10,}=*\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bBasic\s+[0-9A-Za-z+/]{10,}={0,2}\b",
        re.IGNORECASE,
    ),
)

_REDACTED = "[REDACTED]"


class SecretRedactor:
    """Stateless, deterministic, total redaction."""

    def redact_text(self, text: Any) -> str:
        out = redact_text(text)
        for pattern in _SECRET_PATTERNS:
            out = pattern.sub(_REDACTED, out)
        return out

    def redact_value(self, value: Any) -> Any:
        scrubbed = redact_value(value)
        return self._scan_value(scrubbed)

    def _scan_value(self, value: Any) -> Any:
        """Apply secret-shape patterns to every string in
        a structure, after key-name redaction."""
        if isinstance(value, str):
            out = value
            for pattern in _SECRET_PATTERNS:
                out = pattern.sub(_REDACTED, out)
            return out
        if isinstance(value, dict):
            return {
                k: self._scan_value(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            scanned = [self._scan_value(v) for v in value]
            return (
                type(value)(scanned)
                if isinstance(value, tuple)
                else scanned
            )
        return value

    def redact_arguments(
        self, arguments: Any
    ) -> Any:
        return redact_arguments(arguments)

    # -- named contexts (same engine, documented coverage) --
    def for_log(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact_text(value)
        return self.redact_value(value)

    def for_error(self, value: Any) -> Any:
        return self.for_log(value)

    def for_tool_result(self, value: Any) -> Any:
        return self.for_log(value)

    def for_trajectory(self, value: Any) -> Any:
        return self.for_log(value)

    def for_audit(self, value: Any) -> Any:
        return self.for_log(value)

    def for_llm(self, value: Any) -> Any:
        return self.for_log(value)

    def for_screenshot_metadata(
        self, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        return self.redact_value(dict(metadata or {}))

    def scan(self, text: Any) -> list[str]:
        """Return the kinds of secrets detected (no values)."""
        haystack = str(text or "")
        kinds: list[str] = []
        lowered = haystack.lower()
        for keyword, kind in (
            ("api_key", "api_key"), ("apikey", "api_key"),
            ("password", "password"), ("passwd", "password"),
            ("secret", "secret"), ("token", "token"),
            ("bearer", "auth_header"),
            ("-----begin", "private_key"),
            ("cookie", "cookie"),
        ):
            if keyword in lowered:
                kinds.append(kind)
        for pattern in _SECRET_PATTERNS:
            if pattern.search(haystack):
                kinds.append("pattern:" + pattern.pattern[:24])
        return sorted(set(kinds))

    def contains_secret(self, text: Any) -> bool:
        return bool(self.scan(text))


# Shared default instance.
DEFAULT_REDACTOR = SecretRedactor()

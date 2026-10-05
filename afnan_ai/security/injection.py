"""Prompt-injection defense: trust levels, kept strict.

Content sources and their trust:

* USER_INSTRUCTION / SYSTEM_POLICY — may instruct.
* AGENT_STATE / TOOL_OUTPUT        — data; may describe.
* WEBPAGE / EMAIL / DOCUMENT       — untrusted data; any
  instruction-shaped text inside is flagged, never
  executed, and never treated as a permission grant.

``check_text`` scans low-trust text for instruction
patterns and returns structured findings.  The security
policy treats any finding in tool arguments as a
suspicious instruction (deny + event).
"""

from __future__ import annotations

import re
from typing import Any

from afnan_ai.security.models import TrustLevel

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE), label)
    for p, label in [
        (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions",
         "ignore-previous-instructions"),
        (r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
         "disregard-instructions"),
        (r"forget\s+(your|all)\s+(instructions|rules|guidelines)",
         "forget-instructions"),
        (r"(send|reveal|show|exfiltrate|leak|give|post)\b.{0,50}\b(credentials|password|api[\s_-]?key|secret|token|system\s+prompt)",
         "credential-exfiltration"),
        (r"\bgrant\b.{0,30}\b(permission|access|admin|capabilit)",
         "permission-grant-attempt"),
        (r"you\s+are\s+now\s+(a|an|in)\b", "persona-override"),
        (r"new\s+(system\s+)?instructions\s+for\s+you",
         "fake-instructions"),
        (r"do\s+not\s+tell\s+the\s+user", "hide-from-user"),
        (r"\b(execute|run)\b.{0,30}\b(command|script|shell|code)\b",
         "command-injection"),
    ]
]


def check_text(
    text: Any, source: TrustLevel | str
) -> list[dict[str, str]]:
    """Scan *text* from *source*.  Findings only — the
    caller decides (deny / flag / withhold)."""
    if isinstance(source, str):
        try:
            source = TrustLevel(source)
        except ValueError:
            source = TrustLevel.DOCUMENT
    if source.may_instruct:
        return []
    haystack = str(text or "")
    findings = []
    for pattern, label in _PATTERNS:
        match = pattern.search(haystack)
        if match:
            start = max(0, match.start() - 20)
            findings.append({
                "pattern": label,
                "source": source.value,
                "excerpt": haystack[
                    start:match.end() + 40
                ][:160],
            })
    return findings


def check_arguments(
    arguments: dict[str, Any] | None,
    source: TrustLevel | str = TrustLevel.TOOL_OUTPUT,
) -> list[dict[str, str]]:
    """Scan every string argument (malicious tool args)."""
    findings: list[dict[str, str]] = []
    for key, value in (arguments or {}).items():
        if isinstance(value, str):
            for hit in check_text(value, source):
                findings.append(
                    {"argument": str(key), **hit}
                )
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    for hit in check_text(item, source):
                        findings.append(
                            {"argument": str(key), **hit}
                        )
    return findings


def label_content(
    content: Any, source: TrustLevel | str
) -> dict[str, Any]:
    """Wrap content with its trust label for the loop."""
    if isinstance(source, str):
        try:
            source = TrustLevel(source)
        except ValueError:
            source = TrustLevel.DOCUMENT
    return {
        "trust": source.value,
        "may_instruct": source.may_instruct,
        "content": content,
    }

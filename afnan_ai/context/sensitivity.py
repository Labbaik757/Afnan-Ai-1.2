"""Context sensitivity: PUBLIC / INTERNAL / SENSITIVE / SECRET.

SECRET context is never rendered into prompts, logs,
activity or trajectory plaintext — only CredentialVault
references travel.  SENSITIVE items are redacted in
user-facing surfaces.  Filtering happens at every
rendering boundary.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.context.models import (
    ContextItemV2,
    Sensitivity,
)
from afnan_ai.redaction import redact_text


def filter_for_prompt(
    items: list[ContextItemV2],
) -> list[ContextItemV2]:
    """Items safe to render into the LLM prompt."""
    return [
        i
        for i in items
        if i.sensitivity
        in (Sensitivity.PUBLIC, Sensitivity.INTERNAL)
        and not i.is_expired()
    ]


def filter_for_activity(
    items: list[ContextItemV2],
) -> list[dict[str, Any]]:
    """User-facing activity view: redacted summaries."""
    out = []
    for item in items:
        if item.sensitivity == Sensitivity.SECRET:
            continue
        text = item.text
        if item.sensitivity == Sensitivity.SENSITIVE:
            text = redact_text(text)[:120]
        out.append(
            {
                "context_id": item.context_id,
                "kind": item.kind.value,
                "text": text[:300],
                "sensitivity": item.sensitivity.value,
            }
        )
    return out


def vault_reference(item: ContextItemV2) -> str:
    """Replace a SECRET item's content with a vault reference."""
    if item.sensitivity != Sensitivity.SECRET:
        return item.text
    return (
        f"[secret:{item.source or 'vault'} ref={item.context_id}]"
    )


def sanitize_items(
    items: list[ContextItemV2],
) -> list[ContextItemV2]:
    """Return copies safe for trajectory/activity storage.

    SECRET items keep metadata only; content becomes a
    vault reference.
    """
    from dataclasses import replace

    clean = []
    for item in items:
        if item.sensitivity == Sensitivity.SECRET:
            clean.append(
                replace(item, text=vault_reference(item))
            )
        else:
            clean.append(item)
    return clean

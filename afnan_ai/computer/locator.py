"""Semantic desktop locator.

Ranks observed windows and UI elements against a natural-
language description ("Save button", "Chrome ka address
bar", "Settings window") using accessibility metadata first,
window information second and visual detections as fallback.

Confidence tiers mirror the browser semantics layer:
>= 0.8 actionable, 0.5-0.8 uncertain (approval required),
< 0.5 low (the caller withholds the element_id so a guess
can never be acted on).
"""

from __future__ import annotations

import re
from typing import Any

from afnan_ai.computer.models import ComputerElement

_KIND_HINTS = {
    "button": {"button", "pushbutton", "menuitem"},
    "address bar": {"textbox", "edit", "entry", "combobox"},
    "search box": {"textbox", "edit", "entry", "searchbox"},
    "text field": {"textbox", "edit", "entry"},
    "input": {"textbox", "edit", "entry", "combobox"},
    "window": {"window"},
    "dialog": {"window", "dialog"},
    "tab": {"tab", "pagetab"},
    "menu": {"menu", "menubar", "menuitem"},
    "checkbox": {"checkbox", "check"},
    "link": {"link"},
    "icon": {"icon", "image", "button"},
}

_SOURCE_BOOST = {"accessibility": 0.05, "window": 0.03, "visual": 0.0}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _kind_hint(description: str) -> set[str]:
    lowered = description.lower()
    hint: set[str] = set()
    for phrase, roles in _KIND_HINTS.items():
        if phrase in lowered:
            hint |= roles
    return hint


def score_candidate(
    description: str, element: ComputerElement
) -> float:
    query = _tokens(description)
    name_tokens = _tokens(element.name)
    if not query or not name_tokens:
        overlap = 0.0
    else:
        overlap = 2 * len(query & name_tokens) / (
            len(query) + len(name_tokens)
        )
    score = 0.7 * overlap + 0.25 * element.confidence
    score += _SOURCE_BOOST.get(element.source, 0.0)
    hint = _kind_hint(description)
    if hint and element.role.lower() in hint:
        score += 0.15
    # An app-name match ("File Explorer", "Chrome") targets the
    # window/app itself.
    if element.app and element.app.lower() in description.lower():
        score += 0.1
    return max(0.0, min(1.0, score))


def rank_candidates(
    description: str,
    elements: list[ComputerElement],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            (score_candidate(description, element), element)
            for element in elements
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    results: list[dict[str, Any]] = []
    for score, element in ranked[: max(1, int(limit))]:
        if score < 0.2:
            continue
        payload = element.to_dict()
        payload["match_confidence"] = round(score, 3)
        if score >= 0.8:
            payload["tier"] = "high"
        elif score >= 0.5:
            payload["tier"] = "uncertain"
        else:
            payload["tier"] = "low"
            # A guess is never actionable.
            payload["element_id"] = ""
        results.append(payload)
    return results

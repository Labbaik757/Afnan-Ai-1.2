"""Semantic element locator — find elements by description.

Given a natural-language description ("Login button",
"Search field", "Next page link"), candidates from the
accessibility tree / DOM are ranked by role fit and name
similarity, each with a confidence score:

* ``>= 0.8`` — actionable match;
* ``0.5 – 0.8`` — uncertain: verify before acting;
* ``< 0.5`` — low confidence: the match's element reference is
  withheld from tool output, so nothing downstream can act on
  the guess automatically.

This is ranking only — interaction still goes through the
controller's validated tools (and its approval gate).
"""
from __future__ import annotations

import re
from typing import Any

from afnan_ai.browser.accessibility import INTERACTIVE_ROLES

HIGH = 0.8
MEDIUM = 0.5

_ROLE_HINTS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("search",), ("searchbox", "textbox")),
    (("field", "input", "box", "textbox"), ("textbox", "searchbox",
                                            "combobox", "spinbutton")),
    (("button", "btn"), ("button",)),
    (("link",), ("link",)),
    (("checkbox", "check"), ("checkbox",)),
    (("dropdown", "select", "combobox"), ("combobox", "listbox")),
    (("menu",), ("menuitem", "button", "navigation")),
    (("heading", "title", "header"), ("heading",)),
    (("tab",), ("tab",)),
    (("option", "choice"), ("option", "radio")),
]

_SYNONYMS = {
    "login": {"login", "sign", "in", "log"},
    "signin": {"sign", "in", "login"},
    "sign": {"sign", "login"},
    "next": {"next", "forward", "continue", "older", "›", ">"},
    "previous": {"previous", "prev", "back", "newer", "‹", "<"},
    "search": {"search", "find", "query", "lookup", "look"},
    "submit": {"submit", "send", "post", "go"},
    "close": {"close", "dismiss", "x"},
    "menu": {"menu", "navigation", "nav"},
    "home": {"home", "main"},
    "settings": {"settings", "preferences", "options"},
    "download": {"download", "save"},
    "upload": {"upload", "attach", "browse"},
}

_STOPWORDS = {"the", "a", "an", "on", "of", "to", "please", "me"}


def _tokens(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z0-9]+", str(text).lower())
        if t not in _STOPWORDS
    ]


def _expand(tokens: list[str]) -> set[str]:
    expanded = set(tokens)
    for token in tokens:
        expanded |= _SYNONYMS.get(token, set())
    return expanded


def desired_roles(tokens: list[str]) -> set[str]:
    roles: set[str] = set()
    for hints, mapped in _ROLE_HINTS:
        if any(h in tokens for h in hints):
            roles |= set(mapped)
    return roles


def rank(description: str, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank *nodes* against *description*, best first.

    Each returned match is the node plus ``confidence`` and
    ``tier`` (``"actionable"`` / ``"uncertain"`` / ``"low"``).
    """
    tokens = _tokens(description)
    wanted = _expand(tokens)
    roles = desired_roles(tokens)
    name_terms = wanted - {
        "button", "link", "field", "input", "box", "textbox",
    }
    results: list[dict[str, Any]] = []
    for node in nodes:
        role = str(node.get("role", "generic"))
        name = str(node.get("name", ""))
        node_tokens = _expand(_tokens(name))
        if roles:
            if role in roles:
                role_score = 0.45
            elif role in INTERACTIVE_ROLES:
                role_score = 0.08
            else:
                role_score = 0.0
        else:
            role_score = 0.25 if role in INTERACTIVE_ROLES else 0.05
        overlap = (
            2 * len(name_terms & node_tokens)
            / max(1, len(name_terms) + len(node_tokens))
            if name_terms else 0.0
        )
        name_score = 0.45 * overlap
        bonus = 0.1 if (
            name and name.lower() in str(description).lower()
        ) else 0.0
        confidence = min(0.99, role_score + name_score + bonus)
        tier = (
            "actionable" if confidence >= HIGH
            else "uncertain" if confidence >= MEDIUM
            else "low"
        )
        results.append({
            **node,
            "confidence": round(confidence, 3),
            "tier": tier,
        })
    results.sort(key=lambda m: m["confidence"], reverse=True)
    return results

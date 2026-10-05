"""Accessibility tree — normalized, structured page semantics.

The browser's accessibility tree is the most reliable source of
"what can a user do on this page": it names buttons, links,
inputs, headings and menus the way assistive technology (and
users) see them.  This module normalizes the driver's raw
snapshot into a flat list of nodes; when the driver has no
accessibility tree, the controller derives an equivalent list
from DOM interactive elements instead — accessibility first,
DOM as the fallback, pixels (ScreenObserver) only after that.

Node shape::

    {node_id, role, name, value, level, depth,
     element_ref, locator: {role, name}, source, tab_id}

``element_ref`` is set when the node maps to a controller
element (DOM-derived nodes), making it directly actionable via
the existing interaction tools.  Snapshot nodes carry a role +
name locator instead, resolvable through find_elements.
"""
from __future__ import annotations

import re
from typing import Any

# Roles that a user can act on (used by semantic ranking).
INTERACTIVE_ROLES = frozenset({
    "button", "link", "textbox", "searchbox", "checkbox", "radio",
    "combobox", "listbox", "option", "menuitem", "tab", "switch",
    "slider", "spinbutton",
})

_TAG_ROLES = {
    "a": "link",
    "button": "button",
    "select": "combobox",
    "textarea": "textbox",
    "nav": "navigation",
    "main": "main",
    "header": "banner",
    "footer": "contentinfo",
    "ul": "list",
    "ol": "list",
    "li": "listitem",
    "img": "img",
    "table": "table",
    "form": "form",
}

_INPUT_ROLES = {
    "text": "textbox",
    "search": "searchbox",
    "email": "textbox",
    "password": "textbox",
    "url": "textbox",
    "tel": "textbox",
    "number": "spinbutton",
    "checkbox": "checkbox",
    "radio": "radio",
    "submit": "button",
    "button": "button",
    "file": "button",
    "range": "slider",
}


def role_for_element(tag: str, attrs: dict[str, Any]) -> str:
    """Best accessibility role for a DOM element."""
    explicit = str(attrs.get("role", "")).strip().lower()
    if explicit:
        return explicit
    tag = (tag or "").lower()
    if tag == "input":
        return _INPUT_ROLES.get(
            str(attrs.get("type", "text")).lower(), "textbox"
        )
    if len(tag) == 2 and tag[0] == "h" and tag[1] in "123456":
        return "heading"
    return _TAG_ROLES.get(tag, "generic")


def name_for_element(element: dict[str, Any]) -> str:
    """The accessible name a user would recognize: aria-label,
    associated label, placeholder, then visible text."""
    attrs = element.get("attributes") or {}
    for key in ("aria-label", "label", "placeholder", "title", "alt"):
        value = str(attrs.get(key, "")).strip()
        if value:
            return value
    text = str(element.get("text", "")).strip()
    if text:
        return text
    return str(attrs.get("value", "")).strip()


def normalize_snapshot(raw: Any) -> list[dict[str, Any]]:
    """Flatten a driver accessibility snapshot into nodes."""
    nodes: list[dict[str, Any]] = []
    counter = [0]

    def walk(node: Any, depth: int) -> None:
        if not isinstance(node, dict):
            return
        counter[0] += 1
        role = str(node.get("role", "generic")).lower()
        name = str(node.get("name", "")).strip()
        entry: dict[str, Any] = {
            "node_id": f"ax_{counter[0]}",
            "role": role,
            "name": name,
            "depth": depth,
            "element_ref": None,
            "locator": {"role": role, "name": name} if name else {"role": role},
            "source": "accessibility",
        }
        if node.get("value") not in (None, ""):
            entry["value"] = str(node["value"])
        if role == "heading" and node.get("level") is not None:
            entry["level"] = node.get("level")
        nodes.append(entry)
        for child in node.get("children") or []:
            walk(child, depth + 1)

    if isinstance(raw, dict) and raw:
        walk(raw, 0)
    elif isinstance(raw, list):
        for item in raw:
            walk(item, 0)
    return nodes


_ARIA_LINE_RE = re.compile(
    r"^(?P<indent>[ ]*)- (?P<role>[a-zA-Z][a-zA-Z0-9-]*)"
    r"(?: \"(?P<name>(?:[^\"\\]|\\.)*)\")?(?P<rest>.*)$"
)
_ARIA_LEVEL_RE = re.compile(r"\[level=(\d+)\]")


def parse_aria_snapshot(text: str) -> dict[str, Any]:
    """Parse Playwright's ``aria_snapshot()`` YAML into a tree.

    The modern Playwright accessibility API returns indented
    YAML lines (``- button \"Sign in\"``); this converts them to
    the ``{role, name, children, level}`` dict tree that
    :func:`normalize_snapshot` already understands, so the
    deprecated ``page.accessibility`` API is never needed.
    Returns a synthetic root node (possibly with no children).
    """
    root: dict[str, Any] = {
        "role": "document", "name": "", "children": [],
    }
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line in str(text or "").splitlines():
        if not line.strip() or not line.lstrip().startswith("- "):
            continue
        match = _ARIA_LINE_RE.match(line)
        if not match:
            continue
        indent = len(match.group("indent")) // 2
        role = match.group("role").lower()
        name = (match.group("name") or "").replace('\\"', '"')
        rest = match.group("rest") or ""
        node: dict[str, Any] = {"role": role, "name": name,
                                 "children": []}
        if role == "text":
            # "- text: some content" carries its text inline
            inline = rest.lstrip(": ").strip()
            if inline:
                node["name"] = inline
        level = _ARIA_LEVEL_RE.search(rest)
        if level:
            node["level"] = int(level.group(1))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack[-1][1]["children"].append(node)
        stack.append((indent, node))
    return root


def derive_nodes(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive accessibility nodes from DOM element dicts (the
    controller's ElementInfo dicts, which carry live refs)."""
    nodes: list[dict[str, Any]] = []
    for i, element in enumerate(elements, start=1):
        attrs = element.get("attributes") or {}
        role = role_for_element(str(element.get("tag", "")), attrs)
        name = name_for_element(element)
        node: dict[str, Any] = {
            "node_id": f"ax_{i}",
            "role": role,
            "name": name,
            "depth": 0,
            "element_ref": element.get("ref"),
            "locator": {"role": role, "name": name} if name else {"role": role},
            "source": "dom",
        }
        if element.get("value"):
            node["value"] = str(element["value"])
        nodes.append(node)
    return nodes

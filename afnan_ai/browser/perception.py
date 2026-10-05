"""Browser perception — one unified view of the page.

The perception layer gives the agent a single, engine-neutral
model of what the browser shows::

    accessibility tree (preferred)  ┐
    DOM / semantic structure        ├─► UnifiedObservation
    visual detection (fallback)     ┘    + PerceptionElement[]

and a single way to act on it::

    locate → validate → act → fresh observation → verify

Accessibility information is always preferred; DOM data fills
its gaps; pixels (via the ScreenObserver) are only the fallback
for canvas/visual-only interfaces.  Every element — whatever
found it — carries the same identity (role, accessible name,
bounding box, state, confidence, source), and actions execute
through the BrowserController, so validation, the challenge
guard and the human approval gate apply exactly as for the
classic browser tools.  Nothing here talks to Playwright or
Chromium, and no generated code is ever executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from afnan_ai.browser.accessibility import (
    name_for_element,
    role_for_element,
)
from afnan_ai.browser.base import BrowserErrorCode, BrowserException
from afnan_ai.browser.security import ActionRisk, Sensitivity
from afnan_ai.browser.semantics import HIGH as SEMANTIC_HIGH
from afnan_ai.browser.semantics import rank as rank_semantic
from afnan_ai.redaction import redact_text

#: Which roles a description's *last* kind-word asks for — used
#: only to break ranking ties, never to change confidences.
_PRIMARY_HINTS: list[tuple[tuple[str, ...], set[str]]] = [
    (("button", "btn"), {"button"}),
    (("link",), {"link"}),
    (("checkbox",), {"checkbox"}),
    (("radio",), {"radio"}),
    (("box", "field", "input", "textbox"), {"textbox", "searchbox"}),
    (("dropdown", "select", "combobox"), {"combobox", "listbox"}),
    (("tab",), {"tab"}),
]


def _primary_roles(tokens: list[str]) -> set[str]:
    for token in reversed(tokens):
        for words, roles in _PRIMARY_HINTS:
            if token in words:
                return set(roles)
    return set()

#: Actions that change page state (subject to confidence gating).
_ACTING = frozenset({
    "click", "double_click", "type", "clear", "select", "check",
    "uncheck", "press_key", "hotkey", "drag",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PerceptionElement:
    """One candidate element in the unified model."""

    element_id: str
    role: str = "generic"
    accessible_name: str = ""
    text: str = ""
    tag: str = ""
    ref: str | None = None        # controller element ref (DOM/AX)
    vis_ref: str | None = None    # ScreenObserver ref (visual)
    locator: dict[str, Any] = field(default_factory=dict)
    frame: str = ""
    tab_id: str = ""
    window_id: str = "main"
    bbox: dict[str, Any] | None = None
    visible: bool = True
    enabled: bool = True
    editable: bool = False
    checked: bool | None = None
    value: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    source: str = "dom"           # accessibility | dom | visual
    actionable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "role": self.role,
            "accessible_name": redact_text(self.accessible_name)[:200],
            "text": redact_text(self.text)[:200],
            "tag": self.tag,
            "ref": self.ref,
            "locator": dict(self.locator),
            "frame": self.frame,
            "tab_id": self.tab_id,
            "window_id": self.window_id,
            "bbox": dict(self.bbox) if self.bbox else None,
            "visible": self.visible,
            "enabled": self.enabled,
            "editable": self.editable,
            "checked": self.checked,
            "value": self.value,
            "confidence": round(float(self.confidence), 3),
            "source": self.source,
            "actionable": self.actionable,
        }


@dataclass
class UnifiedObservation:
    """The browser as one structured, serializable snapshot."""

    tab_id: str
    url: str = ""
    title: str = ""
    text: str = ""
    elements: list[PerceptionElement] = field(default_factory=list)
    dialogs: list[dict[str, Any]] = field(default_factory=list)
    loading: bool = False
    page_changed: bool = False
    window_id: str = "main"
    session_id: str = ""
    sources: dict[str, int] = field(default_factory=dict)
    observed_at: str = field(default_factory=_now)

    def summary(self) -> dict[str, Any]:
        return {
            "tab_id": self.tab_id,
            "url": self.url,
            "title": self.title,
            "element_count": len(self.elements),
            "page_changed": self.page_changed,
            "loading": self.loading,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "browser_observation",
            "tab_id": self.tab_id,
            "window_id": self.window_id,
            "session_id": self.session_id,
            "url": self.url,
            "title": self.title,
            "text": redact_text(self.text)[:2000],
            "elements": [e.to_dict() for e in self.elements],
            "element_count": len(self.elements),
            "dialogs": list(self.dialogs),
            "loading": self.loading,
            "page_changed": self.page_changed,
            "sources": dict(self.sources),
            "observed_at": self.observed_at,
            "observation": {
                "type": "browser_observation",
                "summary": (
                    f"Perceived page {self.title!r} at {self.url}: "
                    f"{len(self.elements)} unified elements "
                    f"(accessibility={self.sources.get('accessibility', 0)}, "
                    f"dom={self.sources.get('dom', 0)}, "
                    f"visual={self.sources.get('visual', 0)}), "
                    f"loading={self.loading}"
                ),
                "url": self.url,
                "title": self.title,
                "page_changed": self.page_changed,
            },
        }


class BrowserPerception:
    """Unified observation + computer actions over a controller."""

    def __init__(self, controller: Any, screen_observer: Any = None):
        self.controller = controller
        self._screen_observer = screen_observer
        self._counter = 0
        self._elements: dict[str, PerceptionElement] = {}
        self._current: dict[str, UnifiedObservation] = {}
        # Element ids are stable per (tab, element signature):
        # re-observing an unchanged page reuses ids (with fresh
        # refs/state), so a located target stays addressable
        # across the observe → act → observe cycle.  Only ids
        # in the latest observation are "live"; anything else
        # is stale and refused.
        self._signatures: dict[str, dict[tuple, str]] = {}
        self._live: dict[str, set[str]] = {}

    # -- visual source (lazy: only built when needed) -------------------
    @property
    def screen_observer(self) -> Any:
        if self._screen_observer is None:
            from afnan_ai.screen import ScreenObserver

            self._screen_observer = ScreenObserver(
                browser_controller=self.controller
            )
        return self._screen_observer

    @screen_observer.setter
    def screen_observer(self, observer: Any) -> None:
        self._screen_observer = observer

    # -- observation ------------------------------------------------------
    def observe(
        self,
        tab_id: str | None = None,
        *,
        max_elements: int = 50,
        include_visual: bool = False,
    ) -> dict[str, Any]:
        """Build the unified observation for a tab.

        Accessibility tree first, DOM gaps filled, visual
        detection only when asked or when nothing structured
        was found (canvas/visual-only pages).
        """
        controller = self.controller
        dom_obs = controller.observe(
            tab_id, max_elements=max_elements
        )
        resolved_tab = dom_obs["tab_id"]
        dom_by_ref = {
            el["ref"]: el for el in dom_obs.get("elements", [])
        }
        # The accessibility tree's internal observation issues
        # its own refs; match its nodes to *this* observation's
        # DOM elements by role + accessible name as well as ref.
        dom_by_signature: dict[tuple[str, str], dict[str, Any]] = {}
        for dom in dom_by_ref.values():
            key = (
                role_for_element(
                    str(dom.get("tag", "")),
                    dom.get("attributes") or {},
                ),
                name_for_element(dom),
            )
            dom_by_signature.setdefault(key, dom)
        elements: list[PerceptionElement] = []
        covered_refs: set[str] = set()
        sources = {"accessibility": 0, "dom": 0, "visual": 0}
        seen_signatures: dict[tuple, int] = {}

        def signature_for(role: str, name: str, tag: str) -> tuple:
            base = (role, name, tag)
            occurrence = seen_signatures.get(base, 0)
            seen_signatures[base] = occurrence + 1
            return base + (occurrence,)

        tree = {"nodes": []}
        try:
            tree = controller.accessibility_tree(tab_id=resolved_tab)
        except Exception:
            tree = {"nodes": []}
        for node in tree.get("nodes", []):
            role = str(node.get("role", "generic"))
            name = str(node.get("name", ""))
            if role in ("generic", "text") and not name:
                continue
            ref = node.get("element_ref")
            dom = (
                dom_by_ref.get(ref) if ref else None
            ) or dom_by_signature.get((role, name))
            if dom is not None:
                ref = dom["ref"]
                covered_refs.add(ref)
            source = str(node.get("source", "accessibility"))
            element = self._element_for(
                resolved_tab,
                signature_for(
                    role, name, (dom or {}).get("tag", "")
                ),
                role=role,
                accessible_name=name,
                text=(dom or {}).get("text", name),
                tag=(dom or {}).get("tag", ""),
                ref=ref,
                locator=dict(node.get("locator") or {}),
                visible=(dom or {}).get("visible", True),
                enabled=(dom or {}).get("enabled", True),
                editable=(dom or {}).get("editable", False),
                value=(dom or {}).get("value", ""),
                attributes=dict((dom or {}).get("attributes") or {}),
                confidence=0.95 if source == "accessibility" else 0.85,
                source=source if source in sources else "dom",
                actionable=bool(ref),
            )
            if dom is not None:
                element.bbox = controller.element_box(
                    {"ref": ref}, tab_id=resolved_tab
                )
            elif element.locator:
                # Accessibility-only node: resolve a live ref
                # through the controller so the preferred
                # source stays actionable.
                try:
                    found = controller.find_elements(
                        element.locator, tab_id=resolved_tab,
                        limit=1,
                    )
                    if found:
                        element.ref = found[0]["ref"]
                        element.actionable = True
                        element.tag = found[0].get("tag", "")
                        element.text = found[0].get("text", name)
                        element.visible = found[0].get(
                            "visible", True
                        )
                        element.enabled = found[0].get(
                            "enabled", True
                        )
                        element.editable = found[0].get(
                            "editable", False
                        )
                        element.bbox = controller.element_box(
                            {"ref": element.ref},
                            tab_id=resolved_tab,
                        )
                except Exception:
                    pass
            elements.append(element)
            sources[element.source] += 1

        for ref, dom in dom_by_ref.items():
            if ref in covered_refs:
                continue
            role = role_for_element(
                str(dom.get("tag", "")), dom.get("attributes") or {}
            )
            name = name_for_element(dom)
            element = self._element_for(
                resolved_tab,
                signature_for(role, name, str(dom.get("tag", ""))),
                role=role,
                accessible_name=name,
                text=dom.get("text", ""),
                tag=dom.get("tag", ""),
                ref=ref,
                locator={},
                visible=dom.get("visible", True),
                enabled=dom.get("enabled", True),
                editable=dom.get("editable", False),
                value=dom.get("value", ""),
                attributes=dict(dom.get("attributes") or {}),
                confidence=0.8,
                source="dom",
                actionable=True,
            )
            element.bbox = controller.element_box(
                {"ref": ref}, tab_id=resolved_tab
            )
            elements.append(element)
            sources["dom"] += 1

        # Visual fallback: canvas UIs, visual-only controls, or
        # whenever the caller explicitly wants pixels fused in.
        if include_visual or not elements:
            for element in self._visual_elements(
                resolved_tab, signature_for
            ):
                elements.append(element)
                sources["visual"] += 1

        loading = False
        try:
            probe = controller.spa_state(resolved_tab)
            loading = probe.get("ready_state") not in (
                "complete", None,
            )
        except Exception:
            loading = False

        session_id = ""
        try:
            session_id = controller.runtime.session.session_id
        except Exception:
            session_id = ""

        observation = UnifiedObservation(
            tab_id=resolved_tab,
            url=dom_obs.get("url", ""),
            title=dom_obs.get("title", ""),
            text=dom_obs.get("text", ""),
            elements=elements[: max(1, int(max_elements))],
            dialogs=list(dom_obs.get("dialogs") or []),
            loading=loading,
            page_changed=bool(dom_obs.get("page_changed")),
            session_id=session_id,
            sources=sources,
        )
        # Ids from this observation are the tab's live set;
        # anything older that vanished is stale from now on.
        self._live[resolved_tab] = {
            e.element_id for e in observation.elements
        }
        for element in observation.elements:
            if (
                element.ref
                and element.confidence < SEMANTIC_HIGH
            ):
                # Route later actions on this target through the
                # controller's approval gate (same mechanism the
                # semantic locator uses).
                self.controller._semantic_confidence[element.ref] = (
                    element.confidence
                )
        self._current[resolved_tab] = observation
        return observation.to_dict()

    def _element_for(
        self, tab_id: str, signature: tuple, **kwargs: Any
    ) -> PerceptionElement:
        """Create or refresh the element for a stable signature."""
        mapping = self._signatures.setdefault(tab_id, {})
        element_id = mapping.get(signature)
        if element_id is None:
            self._counter += 1
            element_id = f"pe_{self._counter}"
            mapping[signature] = element_id
        element = PerceptionElement(
            element_id=element_id,
            tab_id=kwargs.pop("tab_id", tab_id),
            **kwargs,
        )
        self._elements[element_id] = element
        return element

    def _visual_elements(
        self, tab_id: str, signature_for=None
    ) -> list[PerceptionElement]:
        try:
            screen = self.screen_observer.observe(origin="browser")
        except Exception:
            return []
        elements: list[PerceptionElement] = []
        for vis in screen.visual_elements:
            region = vis.region
            bbox = (
                {
                    "x": region.x, "y": region.y,
                    "width": region.width, "height": region.height,
                }
                if region is not None else None
            )
            signature = (
                ("visual", str(vis.kind or "generic"),
                 str(vis.label or ""),
                 region.x if region else 0,
                 region.y if region else 0)
                if signature_for is None
                else signature_for(
                    str(vis.kind or "generic"),
                    str(vis.label or ""), "visual",
                )
            )
            elements.append(self._element_for(
                tab_id,
                signature,
                role=str(vis.kind or "generic"),
                accessible_name=str(vis.label or ""),
                text=str(vis.label or ""),
                tag="",
                ref=None,
                vis_ref=vis.ref,
                locator={},
                bbox=bbox,
                confidence=float(vis.confidence),
                source="visual",
                actionable=bbox is not None,
            ))
        return elements

    # -- semantic location --------------------------------------------------
    def locate(
        self,
        description: str,
        *,
        tab_id: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Resolve a natural-language target to ranked candidates.

        Fallback chain: accessibility/DOM ranking first; if no
        candidate reaches even uncertain confidence, pixels are
        fused in and the ranking runs again.  Low-confidence
        candidates expose no element id, so nothing downstream
        can act on a guess.
        """
        observed = self.observe(tab_id)
        ranked = self._rank(description, observed)
        if not any(c["confidence"] >= 0.5 for c in ranked):
            observed = self.observe(tab_id, include_visual=True)
            ranked = self._rank(description, observed)
        candidates = []
        for entry in ranked[: max(1, int(limit))]:
            element = self._elements.get(entry["element_ref"])
            if (
                element is not None
                and element.ref
                and entry["confidence"] < SEMANTIC_HIGH
                and entry["tier"] != "low"
            ):
                # Mid-confidence target: the controller's
                # approval gate will demand a human yes before
                # any action on this ref (same mechanism as
                # browser_find_semantic).
                self.controller._semantic_confidence[element.ref] = (
                    float(entry["confidence"])
                )
            public = {
                "element_id": (
                    entry["element_ref"]
                    if entry["tier"] != "low" else None
                ),
                "role": entry["role"],
                "name": entry["name"],
                "confidence": entry["confidence"],
                "tier": entry["tier"],
                "source": (
                    element.source if element is not None else ""
                ),
                "actionable": bool(
                    element is not None
                    and element.actionable
                    and entry["tier"] != "low"
                ),
            }
            candidates.append(public)
        return {
            "description": description,
            "tab_id": observed["tab_id"],
            "candidates": candidates,
            "observation": observed["observation"],
        }

    def _rank(
        self, description: str, observed: dict[str, Any]
    ) -> list[dict[str, Any]]:
        nodes = [
            {
                "role": e["role"],
                "name": e["accessible_name"] or e["text"],
                "element_ref": e["element_id"],
            }
            for e in observed["elements"]
        ]
        ranked = rank_semantic(description, nodes)
        # Tie-break equal confidences towards the kind of
        # element the description names last ("Search button"
        # → the button, not the search box sharing its name).
        import re as _re

        tokens = [
            t for t in _re.findall(r"[a-z0-9]+", description.lower())
            if t not in ("the", "a", "an", "on", "of", "to")
        ]
        primary = _primary_roles(tokens)
        if primary:
            ranked.sort(
                key=lambda m: (
                    m["confidence"],
                    1 if m["role"] in primary else 0,
                ),
                reverse=True,
            )
        return ranked

    # -- computer actions -----------------------------------------------------
    def act(
        self,
        action: str,
        *,
        element_id: str | None = None,
        tab_id: str | None = None,
        text: str | None = None,
        value: str | None = None,
        key: str | None = None,
        x: float | None = None,
        y: float | None = None,
        dx: int = 0,
        dy: int = 0,
        expect_text: str | None = None,
        expect_url_contains: str | None = None,
    ) -> dict[str, Any]:
        """Validate, execute and re-observe one computer action.

        The action never runs on a stale, missing or wrong-tab
        target; low-confidence targets go through the human
        approval gate; and the result carries a *fresh*
        observation — executing the action is never itself
        treated as task success.
        """
        action = str(action or "").strip().lower()
        controller = self.controller
        element = (
            self._elements.get(element_id) if element_id else None
        )
        if element_id and (
            element is None
            or element_id
            not in self._live.get(element.tab_id, set())
        ):
            raise BrowserException(
                f"Element {element_id!r} is unknown or stale; "
                "observe the page again before acting",
                code=BrowserErrorCode.STALE_ELEMENT,
                details={"element_id": element_id},
            )
        # Resolve the tab and refuse cross-tab actions.
        if tab_id is None and element is not None:
            tab_id = element.tab_id
        page = controller.current_page(tab_id)
        resolved_tab = page["tab_id"]
        if element is not None and element.tab_id != resolved_tab:
            raise BrowserException(
                f"Element {element.element_id} belongs to tab "
                f"{element.tab_id}, not {resolved_tab}; refusing "
                "to act on the wrong tab",
                code=BrowserErrorCode.INVALID_TAB,
                details={
                    "element_id": element.element_id,
                    "element_tab": element.tab_id,
                    "tab_id": resolved_tab,
                },
            )
        # Target freshness: the element must still exist as
        # observed (controller refs go stale on page change).
        if element is not None and element.ref:
            controller.inspect_element(
                {"ref": element.ref}, tab_id=resolved_tab
            )
        if (
            element is not None
            and action in _ACTING
            and element.confidence < SEMANTIC_HIGH
            and not element.ref
        ):
            self._require_visual_approval(
                element, action, page
            )

        before = self._current.get(resolved_tab)
        before_summary = (
            before.summary() if before is not None else {}
        )
        result = self._execute(
            action, element, resolved_tab,
            text=text, value=value, key=key,
            x=x, y=y, dx=dx, dy=dy,
        )
        after_dict = self.observe(resolved_tab)
        after = self._current[resolved_tab]
        page_changed = (
            before is None
            or before.url != after.url
            or before.title != after.title
            or before.text[:200] != after.text[:200]
        )
        target_still_exists = None
        if element is not None:
            target_still_exists = any(
                e.role == element.role
                and (e.accessible_name or e.text)
                == (element.accessible_name or element.text)
                for e in after.elements
            )
        verified: bool | None = None
        if expect_text is not None:
            verified = expect_text in after.text
        elif expect_url_contains is not None:
            verified = expect_url_contains in after.url
        return {
            "action": action,
            "tab_id": resolved_tab,
            "element_id": element.element_id if element else None,
            "source": element.source if element else None,
            "result": result,
            "before": before_summary,
            "after": after.summary(),
            "page_changed": page_changed,
            "target_still_exists": target_still_exists,
            "verified": verified,
            "after_observation": after_dict,
            "observation": after_dict["observation"],
        }

    def _require_visual_approval(
        self, element: PerceptionElement, action: str,
        page: dict[str, Any],
    ) -> None:
        risk = ActionRisk(
            sensitivity=Sensitivity.SENSITIVE,
            category="uncertain_target",
            reason=(
                "Visual target "
                f"{element.accessible_name or element.role!r} was "
                f"detected from pixels with confidence "
                f"{element.confidence:.2f} (< 0.80); confirm it "
                "is the right target before acting"
            ),
        )
        decision = self.controller.approval_gate.check(
            risk,
            tool_name="browser_computer_act",
            arguments={"action": action},
            url=page.get("url", ""),
            title=page.get("title", ""),
        )
        if not decision.allowed:
            raise BrowserException(
                f"{decision.detail}: browser_computer_act "
                "(uncertain_target) was not executed",
                code=BrowserErrorCode.APPROVAL_REQUIRED,
                details={
                    "category": "uncertain_target",
                    "element_id": element.element_id,
                },
            )

    def _execute(
        self, action: str, element: PerceptionElement | None,
        tab_id: str, **kwargs: Any,
    ) -> dict[str, Any]:
        controller = self.controller
        ref_target = (
            {"ref": element.ref}
            if element is not None and element.ref else None
        )

        def need_ref() -> dict[str, Any]:
            if ref_target is None:
                raise BrowserException(
                    f"Action {action!r} needs a DOM/accessibility "
                    "target; this element is visual-only",
                    code=BrowserErrorCode.OPERATION_FAILED,
                    details={"action": action},
                )
            return ref_target

        if action == "click":
            if ref_target is not None:
                return controller.click(ref_target, tab_id=tab_id)
            return self._click_visual(element, tab_id, 1)
        if action == "double_click":
            if ref_target is not None:
                return controller.double_click(
                    ref_target, tab_id=tab_id
                )
            return self._click_visual(element, tab_id, 2)
        if action == "type":
            if kwargs.get("text") is None:
                raise BrowserException(
                    "The type action requires text",
                    code=BrowserErrorCode.OPERATION_FAILED,
                )
            return controller.type_text(
                need_ref(), str(kwargs["text"]), tab_id=tab_id
            )
        if action == "clear":
            return controller.clear_field(need_ref(), tab_id=tab_id)
        if action == "select":
            if kwargs.get("value") is None:
                raise BrowserException(
                    "The select action requires a value",
                    code=BrowserErrorCode.OPERATION_FAILED,
                )
            return controller.select_option(
                need_ref(), str(kwargs["value"]), tab_id=tab_id
            )
        if action in ("check", "uncheck"):
            return controller.set_checked(
                need_ref(), action == "check", tab_id=tab_id
            )
        if action == "press_key":
            if not kwargs.get("key"):
                raise BrowserException(
                    "The press_key action requires a key",
                    code=BrowserErrorCode.OPERATION_FAILED,
                )
            return controller.press_key(
                str(kwargs["key"]), ref_target, tab_id=tab_id
            )
        if action == "hotkey":
            combo = str(kwargs.get("key") or "")
            if not combo:
                raise BrowserException(
                    "The hotkey action requires a key combination",
                    code=BrowserErrorCode.OPERATION_FAILED,
                )
            return controller.press_key(combo, None, tab_id=tab_id)
        if action == "scroll":
            if ref_target is not None:
                return controller.scroll_page(
                    target=ref_target, tab_id=tab_id
                )
            return controller.scroll_page(
                dx=int(kwargs.get("dx") or 0),
                dy=int(kwargs.get("dy") or 0),
                tab_id=tab_id,
            )
        if action == "mouse_move":
            if element is not None and element.bbox:
                cx, cy = self._center(element)
                return controller.mouse_move(cx, cy, tab_id=tab_id)
            if kwargs.get("x") is None or kwargs.get("y") is None:
                raise BrowserException(
                    "mouse_move needs an element or x/y coordinates",
                    code=BrowserErrorCode.OPERATION_FAILED,
                )
            return controller.mouse_move(
                float(kwargs["x"]), float(kwargs["y"]),
                tab_id=tab_id,
            )
        if action == "drag":
            if (
                element is not None and element.bbox
                and kwargs.get("x") is not None
                and kwargs.get("y") is not None
            ):
                cx, cy = self._center(element)
                return controller.drag(
                    cx, cy, float(kwargs["x"]),
                    float(kwargs["y"]), tab_id=tab_id,
                )
            raise BrowserException(
                "drag needs an element with a bounding box and "
                "a target x/y",
                code=BrowserErrorCode.OPERATION_FAILED,
            )
        if action == "focus":
            return controller.focus_element(
                need_ref(), tab_id=tab_id
            )
        if action == "hover":
            return controller.hover(need_ref(), tab_id=tab_id)
        raise BrowserException(
            f"Unknown computer action {action!r}",
            code=BrowserErrorCode.OPERATION_FAILED,
            details={"action": action},
        )

    def _click_visual(
        self, element: PerceptionElement | None, tab_id: str,
        click_count: int,
    ) -> dict[str, Any]:
        if element is None or not element.bbox:
            raise BrowserException(
                "This action needs an element with a bounding box",
                code=BrowserErrorCode.OPERATION_FAILED,
            )
        cx, cy = self._center(element)
        return self.controller.click_at(
            cx, cy, tab_id=tab_id, click_count=click_count
        )

    @staticmethod
    def _center(
        element: PerceptionElement,
    ) -> tuple[float, float]:
        box = element.bbox or {}
        return (
            float(box.get("x", 0)) + float(box.get("width", 0)) / 2,
            float(box.get("y", 0)) + float(box.get("height", 0)) / 2,
        )

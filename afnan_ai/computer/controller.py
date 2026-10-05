"""ComputerController — validated desktop observation and
action.

Observe → Validate → Execute → Observe Again → Verify for
every action:

* observation fuses window state, accessibility metadata and
  ScreenObserver visual detections into one structured
  ComputerObservation (raw screenshots never enter results);
* semantic locate ranks targets with confidence; low
  confidence is never actionable, uncertain targets and
  sensitive actions go through the human-approval gate;
* actions are refused on the wrong window (an element bound
  to an inactive window is never clicked blindly), stale
  targets fail structurally, and typing into password fields
  never echoes the secret back.

The controller contains no orchestration: it observes and
performs single validated actions for the AgentLoop.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from afnan_ai.computer.backend import ComputerBackend
from afnan_ai.computer.errors import ComputerError
from afnan_ai.computer.locator import rank_candidates
from afnan_ai.computer.models import (
    ComputerElement,
    ComputerObservation,
    WindowInfo,
)
from afnan_ai.computer.policy import (
    ComputerApprovalGate,
    classify_action,
)
from afnan_ai.redaction import redact_text


class ComputerController:
    def __init__(
        self,
        backend: ComputerBackend,
        *,
        screen_observer: Any = None,
        gate: ComputerApprovalGate | None = None,
        approver: Any = None,
    ):
        self.backend = backend
        self.screen_observer = screen_observer
        self.gate = gate or ComputerApprovalGate(approver)
        self._id_by_signature: dict[str, str] = {}
        self._counter = 0
        self._elements: dict[str, ComputerElement] = {}
        self._observation: ComputerObservation | None = None
        self._known_window_ids: set[str] = set()

    # ---------------------------------------------------------
    # Observation
    # ---------------------------------------------------------
    def observe(
        self, *, include_visual: bool = True,
        screenshot_path: str | None = None,
    ) -> dict[str, Any]:
        try:
            width, height = self.backend.screen_size()
        except ComputerError:
            width, height = 0, 0
        try:
            cx, cy = self.backend.cursor_position()
            cursor = {"x": cx, "y": cy}
        except ComputerError:
            cursor = None
        windows = self.backend.list_windows()
        active = next((w for w in windows if w.active), None)
        if active is None:
            try:
                active = self.backend.active_window()
            except ComputerError:
                active = None
        elements: list[ComputerElement] = []
        live: dict[str, ComputerElement] = {}
        # Accessibility metadata wins over pixels.  Elements
        # are gathered per window so targets on inactive
        # windows are known (and correctly refused).
        raw_elements: list[dict[str, Any]] = []
        for window in windows:
            try:
                for item in self.backend.accessibility_elements(
                    window.window_id
                ):
                    raw_elements.append({
                        **item,
                        "window_id": item.get(
                            "window_id", window.window_id
                        ),
                        "app": item.get("app", window.app),
                    })
            except ComputerError:
                # A crashed/unresponsive window contributes no
                # elements; the rest of the desktop still
                # observes fine.
                continue
        counts: dict[str, int] = {}
        for raw in raw_elements:
            role = str(raw.get("role", ""))
            name = str(raw.get("name", ""))
            window_id = str(
                raw.get("window_id")
                or (active.window_id if active else "")
            )
            base = f"{window_id}|{role}|{name}"
            counts[base] = counts.get(base, 0) + 1
            signature = f"{base}|{counts[base]}"
            element_id = self._id_by_signature.get(signature)
            if element_id is None:
                self._counter += 1
                element_id = f"el_{self._counter}"
                self._id_by_signature[signature] = element_id
            center = None
            bbox = raw.get("bbox")
            if bbox:
                center = {
                    "x": int(bbox.get("x", 0))
                    + int(bbox.get("width", 0)) // 2,
                    "y": int(bbox.get("y", 0))
                    + int(bbox.get("height", 0)) // 2,
                }
            element = ComputerElement(
                element_id=element_id,
                role=role,
                name=name,
                source="accessibility",
                confidence=float(raw.get("confidence", 0.95)),
                bbox=bbox,
                center=center,
                window_id=window_id,
                app=str(
                    raw.get("app")
                    or (active.app if active else "")
                ),
                editable=bool(raw.get("editable", False)),
                password_field=bool(
                    raw.get("password_field", False)
                ),
                value=str(raw.get("value", "")),
            )
            elements.append(element)
            live[element_id] = element
        # Windows themselves are locatable targets.
        for window in windows:
            element = ComputerElement(
                element_id=f"win:{window.window_id}",
                role="window",
                name=window.title or window.app,
                source="window",
                confidence=0.9,
                bbox=window.bbox,
                window_id=window.window_id,
                app=window.app,
            )
            elements.append(element)
            live[element.element_id] = element
        # Visual fallback (canvas / custom-drawn UI).
        screenshot_saved = ""
        if include_visual and self.screen_observer is not None:
            try:
                visual = self.screen_observer.observe()
            except Exception:
                visual = None
            if visual is not None:
                for item in visual.elements:
                    region = item.region
                    bbox = (
                        region.to_dict()
                        if region is not None else None
                    )
                    center = bbox.get("center") if bbox else None
                    element = ComputerElement(
                        element_id=item.ref,
                        role=item.kind or "region",
                        name=item.label or item.kind or "region",
                        source="visual",
                        confidence=float(item.confidence),
                        bbox=bbox,
                        center=center,
                        window_id=(
                            active.window_id if active else ""
                        ),
                        app=active.app if active else "",
                    )
                    elements.append(element)
                    live[item.ref] = element
        fingerprint = hashlib.sha256(
            "|".join([
                active.window_id if active else "",
                ",".join(sorted(w.title for w in windows)),
                ",".join(sorted(live.keys())),
            ]).encode()
        ).hexdigest()[:16]
        previous_windows = self._known_window_ids
        new_windows = [
            w.to_dict()
            for w in windows
            if w.window_id not in previous_windows
        ] if previous_windows else []
        self._known_window_ids = {w.window_id for w in windows}
        self._elements = live
        observation = ComputerObservation(
            screen_width=width,
            screen_height=height,
            cursor=cursor,
            windows=windows,
            active_window_id=(
                active.window_id if active else ""
            ),
            active_app=active.app if active else "",
            elements=elements,
            fingerprint=fingerprint,
            screenshot_saved=screenshot_saved,
        )
        self._observation = observation
        payload = observation.to_dict()
        if new_windows:
            payload["new_windows"] = new_windows
        payload["observation"] = self._summary_text(observation)
        return payload

    @staticmethod
    def _summary_text(observation: ComputerObservation) -> str:
        active = next(
            (w for w in observation.windows
             if w.window_id == observation.active_window_id),
            None,
        )
        title = active.title if active else "(none)"
        return (
            f"Desktop: app={observation.active_app or '(unknown)'} "
            f"window={title!r} windows={len(observation.windows)} "
            f"elements={len(observation.elements)}"
        )[:300]

    def screenshot(
        self, path: str | None = None
    ) -> dict[str, Any]:
        """Capture a screenshot (path reported, never bytes)."""
        data = self.backend.capture_screenshot()
        saved = ""
        if data and path:
            from pathlib import Path

            target = Path(path).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            saved = str(target)
        return {
            "captured": bool(data),
            "saved_to": saved,
            "bytes": len(data) if data else 0,
        }

    # ---------------------------------------------------------
    # Semantic location
    # ---------------------------------------------------------
    def locate(
        self, description: str, *, limit: int = 5
    ) -> dict[str, Any]:
        observation = self._observation or self.observe()
        candidates = rank_candidates(
            description, observation.elements, limit=limit
        )
        return {
            "description": description,
            "candidates": candidates,
            "observation": self._summary_text(observation),
        }

    # ---------------------------------------------------------
    # Applications & windows
    # ---------------------------------------------------------
    def list_applications(self) -> dict[str, Any]:
        return {
            "applications": [
                app.to_dict()
                for app in self.backend.list_applications()
            ]
        }

    def open_application(
        self, name: str, *, wait_s: float = 3.0
    ) -> dict[str, Any]:
        app = self.backend.launch_application(name)
        deadline = time.monotonic() + max(0.0, wait_s)
        window: WindowInfo | None = None
        while True:
            for candidate in self.backend.list_windows():
                if (
                    name.lower() in candidate.title.lower()
                    or name.lower() in candidate.app.lower()
                ):
                    window = candidate
                    break
            if window is not None or time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        result: dict[str, Any] = {"application": app.to_dict()}
        result["window"] = window.to_dict() if window else None
        if window is None:
            result["note"] = (
                "Application launched; its window was not "
                "detected yet — observe again"
            )
        return result

    def close_application(self, name: str) -> dict[str, Any]:
        decision = self.gate.check(
            "close_application",
            level="destructive",
            target_summary=f"Close application {name}",
        )
        if decision.outcome not in ("approved", "not_required"):
            raise ComputerError(
                "approval_required"
                if decision.outcome == "no_approver"
                else "approval_denied",
                f"Closing application {name!r} requires human "
                "approval",
                decision=decision.to_dict(),
            )
        self.backend.close_application(name)
        return {"closed": name, "decision": decision.to_dict()}

    def focus_window(self, window_id: str) -> dict[str, Any]:
        self.backend.focus_window(window_id)
        after = self.observe()
        return {
            "focused": window_id,
            "active_window": after.get("active_window"),
            "observation": after["observation"],
        }

    def wait_for_ui_change(
        self, *, timeout_s: float = 5.0
    ) -> dict[str, Any]:
        baseline = (
            self._observation.fingerprint
            if self._observation
            else self.observe()["fingerprint"]
        )
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            current = self.observe()
            if current["fingerprint"] != baseline:
                return {
                    "changed": True,
                    "observation": current["observation"],
                }
            if time.monotonic() >= deadline:
                return {
                    "changed": False,
                    "observation": current["observation"],
                }
            time.sleep(0.1)

    # ---------------------------------------------------------
    # Validated actions
    # ---------------------------------------------------------
    def _resolve_element(
        self, element_id: str | None
    ) -> ComputerElement | None:
        if not element_id:
            return None
        element = self._elements.get(element_id)
        if element is None:
            raise ComputerError(
                "stale_element",
                f"Element {element_id!r} is not on the current "
                "screen; re-observe and locate again",
            )
        return element

    def _require_active_window(
        self, element: ComputerElement
    ) -> None:
        if not element.window_id or not self._observation:
            return
        if element.window_id != self._observation.active_window_id:
            raise ComputerError(
                "window_mismatch",
                "Target belongs to window "
                f"{element.window_id!r} but the active window "
                f"is {self._observation.active_window_id!r}; "
                "focus the right window first — the action was "
                "NOT executed",
                element_window=element.window_id,
                active_window=self._observation.active_window_id,
            )

    def _check_gate(
        self, action: str, *, target_info: dict[str, Any],
        summary: str,
    ) -> None:
        level = classify_action(action, target=target_info)
        if level == "normal":
            return
        decision = self.gate.check(
            action, level=level, target_summary=summary
        )
        if decision.outcome not in ("approved", "not_required"):
            raise ComputerError(
                "approval_required"
                if decision.outcome == "no_approver"
                else "approval_denied",
                f"Desktop action {action!r} ({level}) requires "
                "human approval",
                decision=decision.to_dict(),
            )

    def _check_confidence(self, element: ComputerElement) -> None:
        if element.confidence >= 0.8:
            return
        if element.confidence < 0.5:
            raise ComputerError(
                "low_confidence",
                f"Target {element.name!r} confidence "
                f"{element.confidence:.2f} is too low to act on; "
                "re-observe or ask a human",
                confidence=element.confidence,
            )
        decision = self.gate.check(
            "uncertain_target",
            level="approval_required",
            target_summary=(
                f"{element.role} {element.name!r} "
                f"(confidence {element.confidence:.2f})"
            ),
        )
        if decision.outcome != "approved":
            raise ComputerError(
                "approval_required"
                if decision.outcome == "no_approver"
                else "approval_denied",
                f"Uncertain target {element.name!r} needs human "
                "approval before acting",
                decision=decision.to_dict(),
            )

    def act(
        self,
        action: str,
        *,
        element_id: str | None = None,
        text: str | None = None,
        key: str | None = None,
        keys: list[str] | None = None,
        x: int | None = None,
        y: int | None = None,
        to_x: int | None = None,
        to_y: int | None = None,
        dx: int = 0,
        dy: int = 0,
        expect_text: str | None = None,
        expect_window: str | None = None,
    ) -> dict[str, Any]:
        action = (action or "").lower()
        before_fingerprint = (
            self._observation.fingerprint
            if self._observation
            else self.observe()["fingerprint"]
        )
        element = self._resolve_element(element_id)
        if action in ("click", "double_click", "right_click",
                      "type") and element is not None:
            self._require_active_window(element)
            self._check_confidence(element)
        target_info: dict[str, Any] = {}
        if element is not None:
            target_info = {
                "name": element.name,
                "password_field": element.password_field,
            }
        if keys:
            target_info["combo"] = "+".join(keys)
        if key:
            target_info["key"] = key
        self._check_gate(
            action,
            target_info=target_info,
            summary=(
                f"{action} on "
                f"{(element.name if element else 'screen')!r}"
            ),
        )
        sensitive_text = bool(
            element and element.password_field
        )
        # Execute.
        if action in ("click", "double_click", "right_click"):
            if element is not None and element.center:
                cx, cy = element.center["x"], element.center["y"]
            elif x is not None and y is not None:
                # Raw coordinates are a validated fallback only:
                # without a semantic target they need approval.
                decision = self.gate.check(
                    "coordinate_click",
                    level="approval_required",
                    target_summary=f"Click at ({x}, {y})",
                )
                if decision.outcome != "approved":
                    raise ComputerError(
                        "approval_required"
                        if decision.outcome == "no_approver"
                        else "approval_denied",
                        "Coordinate-only clicking needs human "
                        "approval; locate a semantic target "
                        "instead",
                    )
                cx, cy = x, y
            else:
                raise ComputerError(
                    "invalid_arguments",
                    f"{action} needs an element or coordinates",
                )
            self.backend.mouse_click(
                cx, cy,
                button=(
                    "right" if action == "right_click" else "left"
                ),
                click_count=2 if action == "double_click" else 1,
            )
        elif action == "type":
            if text is None:
                raise ComputerError(
                    "invalid_arguments", "type needs text"
                )
            if element is not None and element.center:
                self.backend.mouse_click(
                    element.center["x"], element.center["y"]
                )
            self.backend.type_text(text)
        elif action == "key_press":
            if not key:
                raise ComputerError(
                    "invalid_arguments", "key_press needs key"
                )
            self.backend.key_press(key)
        elif action == "hotkey":
            if not keys:
                raise ComputerError(
                    "invalid_arguments", "hotkey needs keys"
                )
            self.backend.hotkey(list(keys))
        elif action == "move_mouse":
            if x is None or y is None:
                raise ComputerError(
                    "invalid_arguments", "move_mouse needs x, y"
                )
            self.backend.mouse_move(x, y)
        elif action == "scroll":
            if element is not None and element.center:
                self.backend.mouse_move(
                    element.center["x"], element.center["y"]
                )
            self.backend.scroll(int(dx), int(dy))
        elif action == "drag":
            if element is not None and element.center:
                sx, sy = element.center["x"], element.center["y"]
            elif x is not None and y is not None:
                sx, sy = x, y
            else:
                raise ComputerError(
                    "invalid_arguments",
                    "drag needs a start element or coordinates",
                )
            if to_x is None or to_y is None:
                raise ComputerError(
                    "invalid_arguments", "drag needs to_x, to_y"
                )
            self.backend.mouse_drag(sx, sy, int(to_x), int(to_y))
        else:
            raise ComputerError(
                "invalid_arguments", f"Unknown action {action!r}"
            )
        # Observe again + verify.
        after = self.observe()
        result: dict[str, Any] = {
            "action": action,
            "success": True,
            "screen_changed": (
                after["fingerprint"] != before_fingerprint
            ),
            "active_app": after.get("active_app"),
            "observation": after["observation"],
            "after_observation": after,
        }
        if element is not None:
            result["target_still_exists"] = (
                element.element_id in self._elements
            )
        if sensitive_text:
            result["typed"] = "***"
        elif action == "type" and text is not None:
            result["typed_length"] = len(text)
        verified: bool | None = None
        if expect_text:
            haystacks = [after["observation"]] + [
                e["name"]
                for e in after["elements"]
            ]
            verified = any(
                expect_text.lower() in h.lower()
                for h in haystacks
            )
        if expect_window:
            active_after = after.get("active_window") or {}
            verified = expect_window.lower() in str(
                active_after.get("title", "")
            ).lower() or expect_window.lower() in str(
                after.get("active_app", "")
            ).lower()
        if verified is not None:
            result["verified"] = verified
        return result

    # ---------------------------------------------------------
    # Verifier support
    # ---------------------------------------------------------
    def verifier_observation(
        self, step: Any = None
    ) -> dict[str, Any] | None:
        """Fresh desktop summary for the Verifier (no pixels)."""
        try:
            payload = self.observe()
        except Exception:
            return None
        return {
            "kind": "computer",
            "active_app": payload.get("active_app"),
            "active_window": payload.get("active_window"),
            "elements": [
                e["name"] for e in payload["elements"][:12]
            ],
        }

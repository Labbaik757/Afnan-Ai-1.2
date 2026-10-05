"""Unified autonomous browser workflow.

:class:`BrowserWorkflow` turns a natural-language browser goal
into one complete, evidence-checked run by composing the pieces
that already exist — it plans, executes and judges *nothing*
itself:

* the **Planner** receives a state seeded with a browser
  briefing (open tabs with purposes, the current page, the
  active profile), so its plan acts on the right tab and does
  not redo finished work;
* the **Executor** runs every step through the ToolRegistry;
* the **Verifier** (with BrowserReliability) re-observes the
  page after every action and judges the actual state, so a
  click that changed nothing is never counted as success;
* the **RecoveryManager** replans on failure/uncertainty with
  the full state (observations, completed steps, failed
  actions) and never blindly repeats a failed action;
* the **BrowserController** supplies observations, tabs,
  profiles, downloads and every safety guard (approval gate,
  challenge pause, rate limiting) exactly as before.

Step, time and recovery limits are enforced by the orchestrator
(``max_iterations`` / ``max_duration_s`` /
``max_recovery_attempts``).  The workflow adds the browser-side
framing around that loop: profile/session setup, the briefing,
and a final answer composed from recorded evidence — extracted
content, search results, verified steps, downloads — never from
what the actions merely claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..log_config import get_logger
from ..state import AgentState
from .base import BrowserException
from .controller import BrowserController

__all__ = ["BrowserTaskResult", "BrowserWorkflow"]

logger = get_logger(__name__)

_FINDING_TYPES = (
    "search_results",
    "opened_result",
    "page_content",
    "pagination",
)


@dataclass
class BrowserTaskResult:
    """The outcome of one autonomous browser goal."""

    goal: str
    status: str  # completed | failed | planning_failed | ...
    answer: str = ""
    error: dict[str, Any] | None = None
    iterations: int = 0
    recovery_attempts: int = 0
    verified_steps: int = 0
    total_steps: int = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    tabs: list[dict[str, Any]] = field(default_factory=list)
    downloads: list[dict[str, Any]] = field(default_factory=list)
    state: Any = None  # AgentState, kept out of to_dict

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "status": self.status,
            "answer": self.answer,
            "error": self.error,
            "iterations": self.iterations,
            "recovery_attempts": self.recovery_attempts,
            "verified_steps": self.verified_steps,
            "total_steps": self.total_steps,
            "evidence": self.evidence,
            "tabs": self.tabs,
            "downloads": self.downloads,
        }


class BrowserWorkflow:
    """Runs browser goals through the existing Agent pipeline."""

    def __init__(
        self,
        *,
        orchestrator: Any,
        controller: BrowserController,
    ) -> None:
        if orchestrator is None or controller is None:
            raise ValueError(
                "BrowserWorkflow needs the Agent orchestrator "
                "and the BrowserController"
            )
        self.orchestrator = orchestrator
        self.controller = controller

    # -- public entry point -------------------------------------------

    def run_goal(
        self,
        goal: str,
        *,
        profile: str | None = None,
        state: AgentState | None = None,
        max_iterations: int | None = None,
        max_duration_s: float | None = None,
        loop: bool = False,
        max_steps: int = 20,
        batch_limit: int = 3,
        max_replans: int = 6,
        max_llm_calls: int = 12,
        max_repeated_actions: int = 2,
    ) -> BrowserTaskResult:
        """Plan, execute, observe, verify and (if needed) recover
        one browser goal, then answer from the evidence.

        With ``loop=True`` the task runs as a true
        observation-driven loop: the Planner proposes only the
        next few actions from the current state, every action is
        verified against a fresh observation, and the Planner
        re-decides after each batch (bounded by max_steps,
        max_replans, max_llm_calls and max_repeated_actions)
        instead of executing one long pre-generated plan.
        """
        goal = str(goal or "").strip()
        if not goal:
            raise ValueError("run_goal needs a non-empty goal")
        try:
            if profile:
                self.controller.select_profile(profile)
            task_state = state or AgentState.create(goal)
            self._brief(task_state)
            if loop:
                result = self.orchestrator.run_loop(
                    goal,
                    state=task_state,
                    max_steps=max_steps,
                    batch_limit=batch_limit,
                    max_replans=max_replans,
                    max_llm_calls=max_llm_calls,
                    max_repeated_actions=max_repeated_actions,
                    max_duration_s=max_duration_s,
                )
            else:
                result = self.orchestrator.run(
                    goal,
                    state=task_state,
                    max_iterations=max_iterations,
                    max_duration_s=max_duration_s,
                )
        except BrowserException as exc:
            return BrowserTaskResult(
                goal=goal,
                status="failed",
                answer=(
                    f"Browser task could not start: "
                    f"{exc.error.message}"
                ),
                error={
                    "code": exc.error.code.value,
                    "message": exc.error.message,
                },
            )
        return self._compose(goal, result)

    # -- briefing -------------------------------------------------------

    def brief(self, state: AgentState | None = None) -> dict[str, Any]:
        """A snapshot of the browser *before* acting: open tabs
        (with purposes), the current page and the active profile.

        Recorded into the task state so the Planner's first plan
        — and every recovery replan — starts from what the
        browser actually looks like instead of guessing.
        """
        snapshot = self._snapshot()
        if state is not None:
            self._record(state, snapshot)
        return snapshot

    def _brief(self, state: AgentState) -> None:
        self._record(state, self._snapshot())

    def _record(
        self, state: AgentState, snapshot: dict[str, Any]
    ) -> None:
        lines = [snapshot["summary"]]
        for tab_line in snapshot["tab_lines"]:
            lines.append(tab_line)
        if snapshot["page_line"]:
            lines.append(snapshot["page_line"])
        state.add_observation(
            "Browser briefing before acting:\n" + "\n".join(lines),
            source="browser",
            metadata={"browser_briefing": snapshot["data"]},
        )

    def _snapshot(self) -> dict[str, Any]:
        controller = self.controller
        runtime = controller.runtime
        data: dict[str, Any] = {
            "profile": controller.current_profile,
            "tabs": [],
            "current_page": None,
            "browser_session": {
                "session_id": runtime.session.session_id,
                "adapter": runtime.name,
                "profile_id": runtime.session.profile_id,
                "status": runtime.session.status,
                # Structured runtime events (browser_started,
                # tab_created, navigation_completed,
                # browser_crashed, ...) as state evidence —
                # types only, never page data or credentials.
                "recent_events": [
                    event["type"] for event in runtime.events(8)
                ],
            },
        }
        tab_lines: list[str] = []
        try:
            for tab in controller.list_tabs():
                info = (
                    tab if isinstance(tab, dict) else tab.to_dict()
                )
                data["tabs"].append(info)
                marker = " (active)" if info.get("active") else ""
                purpose = info.get("purpose") or "-"
                tab_lines.append(
                    f"- tab {info.get('tab_id')}{marker} "
                    f"purpose={purpose}: {info.get('url')} "
                    f"— {info.get('title')}"
                )
        except Exception:
            pass

        page_line = ""
        try:
            observed = controller.observe()
            elements = observed.get("elements") or []
            labels = [
                str(e.get("text") or "").strip()
                for e in elements[:8]
                if str(e.get("text") or "").strip()
            ]
            data["current_page"] = {
                "url": observed.get("url"),
                "title": observed.get("title"),
                "visible_actions": labels,
            }
            text = str(observed.get("text") or "").strip()
            page_line = (
                f"Current page: {observed.get('url')} — "
                f"{observed.get('title')}. {text[:240]}"
            )
            if labels:
                page_line += f" Visible actions: {', '.join(labels)}"
        except Exception:
            pass

        try:
            network = controller.network_status()
            if network.get("failed_requests"):
                data["network_failures"] = network["failed_requests"]
                page_line += (
                    f" Note: {network['failed_requests']} network "
                    "request(s) already failed on this page."
                )
        except Exception:
            pass

        summary = (
            f"Browser state: profile "
            f"{data['profile']!r}, {len(data['tabs'])} tab(s) open."
        )
        return {
            "summary": summary,
            "tab_lines": tab_lines,
            "page_line": page_line,
            "data": data,
        }

    # -- final answer -----------------------------------------------------

    def _compose(self, goal: str, result: Any) -> BrowserTaskResult:
        state = result.state
        evidence = _evidence_from_state(state)
        verified, total = _verification_counts(state)
        tabs: list[dict[str, Any]] = []
        try:
            tabs = [
                t if isinstance(t, dict) else t.to_dict()
                for t in self.controller.list_tabs()
            ]
        except Exception:
            pass
        downloads: list[dict[str, Any]] = []
        try:
            downloads = self.controller.download_manager.list_downloads()
        except Exception:
            pass

        answer_parts = [result.summary()]
        findings = [
            e for e in evidence if e["type"] in _FINDING_TYPES
        ]
        if findings:
            answer_parts.append(
                "Findings: "
                + "; ".join(
                    e["summary"] for e in findings[-4:]
                )
            )
        excerpt = _extracted_excerpt(state)
        if excerpt:
            answer_parts.append(f"Extracted: {excerpt}")
        if verified:
            answer_parts.append(
                f"Verified {verified}/{total} step(s) against the "
                "actual page state."
            )
        if downloads:
            names = ", ".join(
                str(d.get("filename") or d.get("id"))
                for d in downloads
            )
            answer_parts.append(f"Downloads: {names}.")

        return BrowserTaskResult(
            goal=goal,
            status=result.status.value
            if hasattr(result.status, "value")
            else str(result.status),
            answer=" ".join(answer_parts),
            error=result.error,
            iterations=result.iterations,
            recovery_attempts=len(result.recovery_attempts or []),
            verified_steps=verified,
            total_steps=total,
            evidence=evidence,
            tabs=tabs,
            downloads=downloads,
            state=state,
        )


def _evidence_from_state(
    state: AgentState | None,
) -> list[dict[str, Any]]:
    """Observation evidence recorded while the task ran."""
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    if state is None:
        return evidence
    for observation in state.observations:
        meta = observation.metadata or {}
        summary = meta.get("observation")
        if not isinstance(summary, dict) or not summary.get("type"):
            continue
        entry = {
            "type": summary["type"],
            "summary": str(
                summary.get("summary") or observation.text
            ),
            "source": observation.source,
        }
        key = (entry["type"], entry["summary"])
        if key in seen:
            continue
        seen.add(key)
        evidence.append(entry)
    return evidence


def _verification_counts(
    state: AgentState | None,
) -> tuple[int, int]:
    if state is None:
        return 0, 0
    records = state.metadata.get("verifications") or []
    verified = sum(
        1 for r in records if r.get("status") == "verified"
    )
    return verified, len(records)


def _extracted_excerpt(state: AgentState | None) -> str:
    """The most recent extracted page text (bounded, redacted at
    the tool layer) for the final answer."""
    if state is None:
        return ""
    for record in reversed(state.tool_results):
        if not record.success or not isinstance(record.output, dict):
            continue
        if record.tool not in (
            "browser_extract_content",
            "browser_open_result",
        ):
            continue
        document = record.output.get("content")
        if not isinstance(document, dict):
            document = record.output  # extract returns the doc itself
        text = _document_text(document)
        if text:
            return text[:400]
    return ""


def _document_text(document: dict[str, Any]) -> str:
    """Readable body text from a cleaned document dict."""
    chunks = document.get("chunks") or []
    if chunks and isinstance(chunks[0], dict):
        text = str(chunks[0].get("text") or "")
        if text.strip():
            return " ".join(text.split())
    paragraphs = document.get("paragraphs") or []
    if paragraphs:
        return " ".join(
            " ".join(str(p).split()) for p in paragraphs
        )
    return " ".join(str(document.get("text") or "").split())

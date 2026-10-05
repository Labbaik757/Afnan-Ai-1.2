"""BrowserReliability — observation-grounded confirmation and
structured recovery advice for browser tasks.

The BrowserController knows how to drive a browser; this layer
answers two higher-level questions without executing anything
itself:

* "What does the page actually look like right now?" —
  :meth:`BrowserReliability.observe_state` returns a fresh
  page observation for ``browser_*`` steps only.  Wired into the
  Verifier as its observation provider, it makes browser task
  completion depend on the *actual page state*, not on a tool
  merely reporting success.

* "That browser action failed or did not produce the expected
  change — what should happen next?" —
  :meth:`BrowserReliability.advise` classifies the failure
  (element not found, timeout, stale element, navigation
  failure, unexpected popup, page state mismatch, ...) and
  returns a structured recovery strategy plus the observed page
  evidence (current URL/title and candidate locators taken from
  the elements that really exist).  Wired into the Verifier as
  its failure advisor, the advice is recorded with the
  verification in AgentState, so the existing Recovery/
  Replanning system generates an *alternative* action instead
  of blindly repeating the failed one.  Nothing here executes
  a tool, plans, or judges outcomes — those stay with the
  Executor, Planner/RecoveryManager and Verifier.
"""
from __future__ import annotations

from typing import Any

from afnan_ai.browser.controller import BrowserController
from afnan_ai.log_config import get_logger

logger = get_logger(__name__)

# failure kind -> (strategy name, guidance for the replanner)
_STRATEGIES: dict[str, tuple[str, str]] = {
    "element_not_found": (
        "relocate_from_observation",
        "The target element was not found on the page. Re-observe "
        "the page and pick a target from the elements that "
        "actually exist (see the candidate locators); do not "
        "retry the same locator unchanged.",
    ),
    "timeout": (
        "wait_for_condition_first",
        "The action timed out, most likely because the page "
        "content was not ready yet. First wait for the element or "
        "text to appear with browser_wait_for, then perform the "
        "action; do not repeat the same action immediately.",
    ),
    "stale_element": (
        "refind_element",
        "The page changed after the element was found, so its "
        "reference went stale. Find the element again on the "
        "current page and use the fresh reference.",
    ),
    "navigation_failed": (
        "reobserve_and_choose_new_route",
        "Navigation failed. Observe the page you are actually "
        "on and choose a different route from there instead of "
        "repeating the same navigation.",
    ),
    "browser_not_started": (
        "launch_browser_first",
        "No browser is running. Launch the browser first "
        "(browser_launch), then continue the task.",
    ),
    "invalid_tab": (
        "adopt_and_select_tab",
        "The tab reference is invalid. List the open tabs "
        "(popups are adopted automatically) and select the "
        "intended tab before continuing.",
    ),
    "state_mismatch": (
        "reobserve_and_replan",
        "The action ran but the page does not show the expected "
        "result. Use the observed page state below to choose a "
        "different action; the action that ran did not produce "
        "the expected change, so do not repeat it as-is.",
    ),
    "action_failed": (
        "reobserve_and_replan",
        "The browser action failed. Use the observed page state "
        "below to choose a different action instead of repeating "
        "the failed one.",
    ),
}


def _classify_failure(reason: str, evidence: dict[str, Any]) -> str:
    text = f"{reason} {evidence.get('error') or ''}".lower()
    if "no element matches" in text or "element_not_found" in text:
        return "element_not_found"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "stale" in text or "older version" in text:
        return "stale_element"
    if "navigation" in text or "navigat" in text:
        return "navigation_failed"
    if (
        "not running" in text
        or "browser_not_started" in text
        or "no browser" in text
    ):
        return "browser_not_started"
    if "invalid tab" in text:
        return "invalid_tab"
    if "observed state does not confirm" in text:
        return "state_mismatch"
    return "action_failed"


class BrowserReliability:
    """Observation + recovery-advice provider for browser tasks.

    Constructed over a :class:`BrowserController`; both public
    methods are safe to call when no browser is running (they
    return None / evidence-free advice instead of raising).
    """

    def __init__(self, controller: BrowserController):
        self.controller = controller

    # -- Verifier observation provider ---------------------------------
    #: tools whose effect is *not* a page change; verifying them
    #: against page state would be meaningless (a screenshot is
    #: verified by its file, an observation by its content)
    _NON_PAGE_TOOLS = frozenset({
        "browser_launch",
        "browser_connect",
        "browser_shutdown",
        "browser_list_tabs",
        "browser_observe_page",
        "browser_screenshot",
        "browser_current_page",
        "browser_find_elements",
        "browser_inspect_element",
    })

    def observe_state(self, step: Any = None) -> dict[str, Any] | None:
        """Fresh page observation for a ``browser_*`` step.

        Returns None for non-browser steps, for tools whose
        effect is not a page change, and whenever the browser
        cannot be observed, so verification falls back to the
        action's own report.
        """
        tool_name = str(getattr(step, "tool_name", "")) if step else ""
        if step is not None and (
            not tool_name.startswith("browser_")
            or tool_name in self._NON_PAGE_TOOLS
        ):
            return None
        try:
            observation = self.controller.observe()
        except Exception:
            return None
        try:
            observation["tabs"] = self.controller.list_tabs()
        except Exception:
            pass
        return observation

    # -- Verifier failure advisor ----------------------------------------
    def advise(
        self,
        step: Any,
        result: Any,
        observation: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Structured recovery advice for a failed/uncertain
        browser step.  Data only — this never executes, plans or
        judges; the Agent's Recovery system decides what to do.
        """
        if not str(getattr(step, "tool_name", "")).startswith("browser_"):
            return None
        evidence = getattr(result, "evidence", {}) or {}
        kind = _classify_failure(
            str(getattr(result, "reason", "")), evidence
        )
        strategy, guidance = _STRATEGIES[kind]

        advice: dict[str, Any] = {
            "kind": kind,
            "strategy": strategy,
            "advice": guidance,
            "failed_tool": getattr(step, "tool_name", None),
        }

        if isinstance(observation, dict):
            advice["current_url"] = observation.get("url")
            advice["current_title"] = observation.get("title")
            candidates = self._candidate_locators(observation)
            if candidates:
                advice["candidate_locators"] = candidates
            tabs = observation.get("tabs")
            if isinstance(tabs, list) and len(tabs) > 1:
                advice["open_tabs"] = [
                    {
                        "tab_id": t.get("tab_id"),
                        "url": t.get("url"),
                        "title": t.get("title"),
                    }
                    for t in tabs
                ]
                advice["advice"] = (
                    guidance
                    + f" Note: {len(tabs)} tabs are open — a popup "
                    "or new tab may have appeared; use "
                    "browser_select_tab with the intended tab_id "
                    "before continuing."
                )

        logger.info(
            "browser reliability advice: %s (%s) for %s",
            kind,
            strategy,
            advice["failed_tool"],
        )
        return advice

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _candidate_locators(
        observation: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Concrete alternative targets from the real page."""
        candidates: list[dict[str, str]] = []
        for element in observation.get("elements", [])[:25]:
            if not isinstance(element, dict):
                continue
            attrs = element.get("attributes") or {}
            if attrs.get("id"):
                candidates.append({"selector": f"#{attrs['id']}"})
            elif element.get("text"):
                candidates.append({"text": str(element["text"])[:40]})
            if len(candidates) >= 8:
                break
        return candidates

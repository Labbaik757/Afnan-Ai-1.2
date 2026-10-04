"""Verifier — checks executed step results against expectations.

The Planner plans, the Executor executes, and the Verifier judges
the outcome.  It is deliberately independent of both:

* it **never executes a tool** (no registry, no tool calls — it
  cannot re-run anything even by accident),
* it does not plan (that is the Planner's job) and does not dispatch
  work (that is the Executor's job),
* it only *analyzes* what is already there: the step's structured
  execution result, its ``expected_result``, and any evidence in
  :class:`~afnan_ai.state.AgentState` (completed/failed step
  records, tool results, observations).

For each step it returns one of three statuses:

* ``verified`` — the execution succeeded and the actual output
  confirms the expected outcome,
* ``failed`` — the execution failed / was skipped, or the actual
  output clearly contradicts the expected outcome,
* ``uncertain`` — there is not enough (or only partial or
  conflicting) evidence to honestly confirm or deny the outcome.

Every verification is recorded in AgentState — as a ``verifier``
observation, a structured entry in ``state.metadata["verifications"]``
and an annotation on the step's record — *without* adding a tool
result: recording a judgement is not executing a tool.

The analysis is deterministic (no model call), so the same result
always verifies the same way and tests need no LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable

from afnan_ai.executor import ExecutionReport, StepExecutionResult
from afnan_ai.planner import PlanStep, TaskPlan
from afnan_ai.state import AgentState


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


@dataclass
class VerificationResult:
    """Judgement for one executed step."""

    step_id: str
    tool_name: str
    status: VerificationStatus
    expected_result: str = ""
    actual_output: Any = None
    reason: str = ""
    confidence: float = 0.0  # 0.0–1.0, how strong the evidence is
    evidence: dict[str, Any] = field(default_factory=dict)
    verified_at: str = field(default_factory=_now_iso)

    @property
    def success(self) -> bool:
        return self.status == VerificationStatus.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "expected_result": self.expected_result,
            "actual_output": _safe(self.actual_output),
            "reason": self.reason,
            "confidence": round(float(self.confidence), 3),
            "evidence": _safe(dict(self.evidence)),
            "verified_at": self.verified_at,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)


@dataclass
class VerificationReport:
    """Judgements for a whole executed plan."""

    plan_id: str | None = None
    goal: str = ""
    task_id: str | None = None
    results: list[VerificationResult] = field(default_factory=list)
    verified_at: str = field(default_factory=_now_iso)

    @property
    def status(self) -> VerificationStatus:
        """Overall: failed if any step failed, verified only if
        every step verified, otherwise uncertain."""
        if any(r.status == VerificationStatus.FAILED for r in self.results):
            return VerificationStatus.FAILED
        if self.results and all(
            r.status == VerificationStatus.VERIFIED for r in self.results
        ):
            return VerificationStatus.VERIFIED
        return VerificationStatus.UNCERTAIN

    @property
    def success(self) -> bool:
        return self.status == VerificationStatus.VERIFIED

    def count(self, status: VerificationStatus) -> int:
        return sum(1 for r in self.results if r.status == status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "task_id": self.task_id,
            "status": self.status.value,
            "success": self.success,
            "results": [r.to_dict() for r in self.results],
            "verified_at": self.verified_at,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)


def _safe(value: Any) -> Any:
    """Keep plain JSON-ish values; stringify anything exotic."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return str(value)


# ----------------------------------------------------------------------
# Text analysis helpers (deterministic, no model)
# ----------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "should", "will", "would", "can", "could", "to", "of", "for", "on",
    "in", "at", "by", "with", "and", "or", "as", "it", "its", "this",
    "that", "from", "into", "then", "than", "has", "have", "had",
    "successfully", "success", "successful", "done", "ok", "okay",
    "completed", "complete", "finished", "properly", "correctly",
}

_SYNONYMS = {
    "launched": "open", "launch": "open", "launches": "open",
    "launching": "open", "opened": "open", "opens": "open",
    "opening": "open", "started": "open", "starts": "open",
    "starting": "open", "displayed": "show", "displays": "show",
    "shown": "show", "showing": "show", "saved": "save",
    "saving": "save", "created": "create", "creating": "create",
    "searched": "search", "searching": "search", "results": "result",
    "screenshots": "screenshot", "files": "file", "browsers": "browser",
    "windows": "window", "applications": "application", "apps": "app",
    "websites": "website", "pages": "page",
}

# Application-like entities where a contradiction is decisive:
# expected Chrome but the actual application is Edge means failed,
# not merely "uncertain".
_APP_ENTITIES = {"chrome", "edge", "safari", "firefox", "whatsapp", "vscode"}

_APP_ALIASES = {
    "google chrome": "chrome",
    "microsoft edge": "edge",
    "visual studio code": "vscode",
    "vs code": "vscode",
}


def _tokens(text: Any) -> list[str]:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    tokens = []
    for word in words:
        if len(word) < 2:
            continue
        word = _SYNONYMS.get(word, word)
        if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(_SYNONYMS.get(word, word))
    return tokens


def _keywords(text: Any) -> list[str]:
    """Meaningful expected-outcome tokens, in order, de-duplicated."""
    seen: list[str] = []
    for token in _tokens(text):
        if token not in _STOPWORDS and token not in seen:
            seen.append(token)
    return seen


def _actual_tokens(step: PlanStep, actual: "_Actual") -> set[str]:
    output = actual.output
    if isinstance(output, (dict, list)):
        text = json.dumps(output, ensure_ascii=False, default=str)
    else:
        text = "" if output is None else str(output)
    tokens = set(_tokens(text))
    # The tool that ran is part of the actual outcome
    tokens.update(_tokens(step.tool_name.replace("_", " ")))
    # A screenshot tool returning a saved path confirms file creation
    if step.tool_name == "take_screenshot" and isinstance(output, str) and output:
        tokens.update({"screenshot", "file", "save", "create", "image"})
    return tokens


# ----------------------------------------------------------------------
# Normalized "actual result" — from an execution result, a tool
# result (registry or AgentState), a dict, or AgentState alone
# ----------------------------------------------------------------------


@dataclass
class _Actual:
    success: bool | None = None
    output: Any = None
    error: str | None = None
    skipped: bool = False
    source: str = "none"


def _normalize_actual(execution_result: Any) -> _Actual | None:
    if execution_result is None:
        return None
    if isinstance(execution_result, StepExecutionResult):
        error_text = None
        if execution_result.error:
            error_text = execution_result.error.get("message") or str(
                execution_result.error
            )
        return _Actual(
            success=execution_result.success,
            output=execution_result.output,
            error=error_text,
            skipped=execution_result.skipped,
            source="execution_result",
        )
    if isinstance(execution_result, dict):
        error = execution_result.get("error")
        if isinstance(error, dict):
            error = error.get("message") or str(error)
        status = str(execution_result.get("status") or "").lower()
        return _Actual(
            success=execution_result.get("success"),
            output=execution_result.get("output"),
            error=error,
            skipped=bool(execution_result.get("skipped")) or status == "skipped",
            source="dict",
        )
    # ToolResult from the registry or from AgentState (duck-typed)
    success = getattr(execution_result, "success", None)
    if success is not None or hasattr(execution_result, "output"):
        error = getattr(execution_result, "error", None)
        if error is not None and not isinstance(error, str):
            error = getattr(error, "message", None) or str(error)
        return _Actual(
            success=success,
            output=getattr(execution_result, "output", None),
            error=error,
            source=type(execution_result).__name__,
        )
    return None


def _state_evidence(state: AgentState | None, step: PlanStep) -> _Actual | None:
    """Actual outcome reconstructed from AgentState alone (used when
    no execution result object is handed over)."""
    if state is None:
        return None
    for record in state.failed_steps:
        if record.name == step.step_id:
            return _Actual(False, record.result, record.error, source="state")
    for record in state.completed_steps:
        if record.name == step.step_id:
            return _Actual(True, record.result, None, source="state")
    matches = [
        t
        for t in state.tool_results
        if (t.metadata or {}).get("step_id") == step.step_id
        or t.tool == step.tool_name
    ]
    if matches:
        last = matches[-1]
        return _Actual(last.success, last.output, last.error, source="state")
    return None


def _state_contradicts(state: AgentState | None, step: PlanStep) -> bool:
    if state is None:
        return False
    if any(r.name == step.step_id for r in state.failed_steps):
        return True
    return any(
        not t.success
        for t in state.tool_results
        if (t.metadata or {}).get("step_id") == step.step_id
    )


# ----------------------------------------------------------------------
# Verifier
# ----------------------------------------------------------------------


class Verifier:
    """Judge executed steps against their expected results.

    Holds no ToolRegistry and no LLM — by construction it cannot
    execute or re-run a tool, plan a task, or dispatch work.  Pass
    a default ``state`` to record into, or pass one per call.
    """

    #: keyword overlap at/above this is needed to call a step verified
    VERIFIED_THRESHOLD = 0.75

    def __init__(self, state: AgentState | None = None):
        self.state = state

    # -- single step --------------------------------------------------
    def verify_step(
        self,
        step: PlanStep,
        execution_result: Any = None,
        *,
        state: AgentState | None = None,
        record: bool = True,
    ) -> VerificationResult:
        effective_state = state if state is not None else self.state
        actual = _normalize_actual(execution_result)
        if actual is None:
            actual = _state_evidence(effective_state, step)

        result = self._judge(step, actual, effective_state)
        if record and effective_state is not None:
            self._record(effective_state, result)
        return result

    # -- whole plan ------------------------------------------------------
    def verify_plan(
        self,
        plan: TaskPlan,
        execution: ExecutionReport | Iterable[Any],
        *,
        state: AgentState | None = None,
        record: bool = True,
    ) -> VerificationReport:
        effective_state = state if state is not None else self.state
        by_step: dict[str, StepExecutionResult] = {}
        if isinstance(execution, ExecutionReport):
            by_step = {r.step_id: r for r in execution.step_results}
            report_task_id = execution.task_id
            report_plan_id = execution.plan_id
        else:
            for item in execution:
                step_id = getattr(item, "step_id", None)
                if step_id:
                    by_step[step_id] = item
            report_task_id = effective_state.task_id if effective_state else None
            report_plan_id = plan.plan_id

        results = [
            self.verify_step(
                step,
                by_step.get(step.step_id),
                state=effective_state,
                record=record,
            )
            for step in plan.steps
        ]
        return VerificationReport(
            plan_id=report_plan_id or plan.plan_id,
            goal=plan.goal,
            task_id=report_task_id
            or (effective_state.task_id if effective_state else plan.task_id),
            results=results,
        )

    # Aliases in the vocabulary used elsewhere
    verify_execution = verify_plan
    verify_report = verify_plan

    # -- judgement ---------------------------------------------------------
    def _judge(
        self,
        step: PlanStep,
        actual: _Actual | None,
        state: AgentState | None,
    ) -> VerificationResult:
        def make(status, reason, confidence, evidence=None, output=None):
            merged = {"source": actual.source if actual else "none"}
            merged.update(evidence or {})
            return VerificationResult(
                step_id=step.step_id,
                tool_name=step.tool_name,
                status=status,
                expected_result=step.expected_result or "",
                actual_output=output if output is not None else (actual.output if actual else None),
                reason=reason,
                confidence=confidence,
                evidence=merged,
            )

        # No evidence at all -> honest "we cannot tell"
        if actual is None:
            return make(
                VerificationStatus.UNCERTAIN,
                f"No execution result or state evidence for step "
                f"{step.step_id!r}; the outcome cannot be verified",
                0.1,
            )

        # A skipped step produced no outcome -> expected not achieved
        if actual.skipped:
            return make(
                VerificationStatus.FAILED,
                f"Step {step.step_id!r} was skipped, so its expected "
                f"result was not produced",
                1.0,
            )

        # The execution itself failed -> cannot be verified as done
        if actual.success is False:
            detail = f": {actual.error}" if actual.error else ""
            return make(
                VerificationStatus.FAILED,
                f"Step {step.step_id!r} execution failed{detail}",
                1.0,
                {"execution_success": False, "error": actual.error},
            )

        if actual.success is None:
            return make(
                VerificationStatus.UNCERTAIN,
                f"Step {step.step_id!r} result does not say whether "
                f"it succeeded; the outcome cannot be verified",
                0.2,
            )

        # Execution claims success but AgentState recorded a failure
        # for the same step -> conflicting evidence
        if _state_contradicts(state, step):
            return make(
                VerificationStatus.UNCERTAIN,
                f"Execution reports success for step {step.step_id!r} "
                f"but AgentState records a failure for it; evidence "
                f"is conflicting",
                0.3,
                {"execution_success": True, "state_conflict": True},
            )

        expected = (step.expected_result or "").strip()
        if not expected:
            return make(
                VerificationStatus.UNCERTAIN,
                f"Step {step.step_id!r} has no expected_result to "
                f"verify its output against",
                0.3,
                {"execution_success": True},
            )

        if actual.output is None:
            return make(
                VerificationStatus.UNCERTAIN,
                f"Step {step.step_id!r} succeeded but returned no "
                f"observable output to check against {expected!r}",
                0.3,
                {"execution_success": True, "output_missing": True},
            )

        # Output itself flags a non-outcome (e.g. launched=False)
        if isinstance(actual.output, dict):
            for flag in ("launched", "opened", "success"):
                if actual.output.get(flag) is False:
                    return make(
                        VerificationStatus.FAILED,
                        f"Step {step.step_id!r} output reports "
                        f"{flag}=False, contradicting {expected!r}",
                        0.9,
                        {"execution_success": True, "contradicting_flag": flag},
                    )

        keywords = _keywords(expected)
        if not keywords:
            return make(
                VerificationStatus.UNCERTAIN,
                f"Expected result {expected!r} for step {step.step_id!r} "
                f"is too vague to verify against the actual output",
                0.3,
                {"execution_success": True, "keywords": []},
            )

        # Decisive contradiction: a different application actually ran
        contradiction = self._app_contradiction(step, actual)
        if contradiction:
            return make(
                VerificationStatus.FAILED,
                contradiction,
                0.95,
                {"execution_success": True, "keywords": keywords},
            )

        actual_tokens = _actual_tokens(step, actual)
        matched = [k for k in keywords if k in actual_tokens]
        missing = [k for k in keywords if k not in actual_tokens]
        ratio = len(matched) / len(keywords)
        evidence = {
            "execution_success": True,
            "keywords": keywords,
            "matched_keywords": matched,
            "missing_keywords": missing,
            "match_ratio": round(ratio, 3),
        }

        if ratio >= self.VERIFIED_THRESHOLD:
            return make(
                VerificationStatus.VERIFIED,
                f"Actual result confirms {expected!r} "
                f"(matched {matched})",
                max(0.7, ratio),
                evidence,
            )
        if ratio == 0:
            return make(
                VerificationStatus.FAILED,
                f"Actual result does not confirm {expected!r}: none "
                f"of {keywords} appear in the output",
                0.8,
                evidence,
            )
        return make(
            VerificationStatus.UNCERTAIN,
            f"Actual result only partly confirms {expected!r} "
            f"(matched {matched}, missing {missing})",
            0.4 + ratio / 2,
            evidence,
        )

    @staticmethod
    def _app_contradiction(step: PlanStep, actual: _Actual) -> str | None:
        if not isinstance(actual.output, dict):
            return None
        actual_app = actual.output.get("application")
        if not actual_app:
            return None
        actual_norm = _APP_ALIASES.get(
            str(actual_app).strip().lower(), str(actual_app).strip().lower()
        )
        if actual_norm not in _APP_ENTITIES:
            return None
        expected_apps = set(_tokens(step.expected_result)) & _APP_ENTITIES
        if expected_apps and actual_norm not in expected_apps:
            return (
                f"Expected {'/'.join(sorted(expected_apps))} but the "
                f"actual application was {actual_norm!r}"
            )
        return None

    # -- AgentState recording (judgement only — no tool result) ------------
    @staticmethod
    def _record(state: AgentState, result: VerificationResult) -> None:
        entry = result.to_dict()
        verifications = state.metadata.get("verifications")
        if not isinstance(verifications, list):
            verifications = []
            state.metadata["verifications"] = verifications
        verifications.append(entry)

        state.add_observation(
            f"Verification {result.status.value} for step "
            f"{result.step_id} ({result.tool_name}): {result.reason}",
            source="verifier",
            metadata={"verification": entry},
        )
        # Annotate the step's own record when there is one
        for record_ in list(state.completed_steps) + list(state.failed_steps):
            if record_.name == result.step_id:
                record_.metadata["verification"] = entry

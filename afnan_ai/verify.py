"""Verifier-Repair Loop — produce → verify → repair for plans and answers.

The orchestrator generates multi-step plans with an LLM.  Small or
sloppy models hallucinate tool names, forget required arguments, or
order steps nonsensically ("click submit" before "fill form").  This
module adds a cheap verification gate:

    plan = llm.generate_plan(goal)
    result = verifier.verify_plan(plan, goal)
    if not result.passed and result.retryable:
        plan = regenerate_with_feedback(feedback=result.issues)

``verify_then_repair`` runs that whole loop (bounded attempts) and
returns the best attempt.  ``AnswerVerifier`` applies the same idea to
chat replies before they are spoken.

Stdlib only.  Works with :class:`afnan_ai.tools.ToolRegistry` and any
:class:`afnan_ai.llm.LLMProvider`.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

try:  # optional imports — the verifier degrades gracefully without them
    from afnan_ai.tools.registry import ToolRegistry
except Exception:  # pragma: no cover - import shim
    ToolRegistry = None  # type: ignore[assignment]

try:
    from afnan_ai.planner import PlanStep, TaskPlan
except Exception:  # pragma: no cover - import shim
    PlanStep = None  # type: ignore[assignment]
    TaskPlan = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: sanity bound — plans longer than this are rejected
MAX_PLAN_STEPS = 20

#: keywords that mark a step as "filling something in"
_FILL_KEYWORDS = ("fill", "type", "input", "enter", "write", "set_value",
                  "send_keys", "paste")

#: keywords that mark a step as "submitting / finalizing"
_SUBMIT_KEYWORDS = ("submit", "send_form", "click_submit", "press_enter",
                    "finalize")

#: answers that match these patterns are generic dodges, not real answers
_DODGE_PATTERNS = (
    r"\bi don'?t know\b",
    r"\bas an ai\b",
    r"\bi can'?t (help|answer|do)\b",
    r"\bi'?m (sorry|unable)\b",
    r"\bno idea\b",
)

#: common stopwords — ignored when extracting question keywords
_STOPWORDS = frozenset(
    "a an the and or but if then else when what which who whom whose "
    "where why how is are was were be been being do does did done have "
    "has had having will would shall should can could may might must "
    "of to in on at for with by from as it its this that these those "
    "i you he she we they me him her us them my your his our their "
    "mein kya hai ke ki ka ko se ne par bhi".split()
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Outcome of verifying one plan."""

    passed: bool
    issues: list[str] = field(default_factory=list)
    retryable: bool = True
    checked_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": list(self.issues),
            "retryable": self.retryable,
            "checked_steps": self.checked_steps,
        }


@dataclass
class RepairOutcome:
    """Outcome of :func:`verify_then_repair`."""

    plan: list[dict[str, Any]]
    result: VerificationResult
    attempts: int
    repaired: bool  # True when a later attempt passed
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan,
            "result": self.result.to_dict(),
            "attempts": self.attempts,
            "repaired": self.repaired,
            "issues": list(self.issues),
        }


# ---------------------------------------------------------------------------
# Step normalization
# ---------------------------------------------------------------------------


def _normalize_step(step: Any) -> dict[str, Any] | None:
    """Convert a PlanStep / dict / anything into a plain step dict.

    Returns None when the step cannot be interpreted at all.
    """
    if PlanStep is not None and isinstance(step, PlanStep):
        return {
            "step_id": step.step_id,
            "description": step.description,
            "tool_name": step.tool_name,
            "arguments": dict(step.arguments or {}),
        }
    if isinstance(step, dict):
        tool = step.get("tool_name", step.get("tool", ""))
        args = step.get("arguments", step.get("params", {}))
        # Keep non-dict arguments as-is so _verify_step can flag them;
        # only dict() them when they really are mappings.
        if isinstance(args, dict):
            args = dict(args)
        return {
            "step_id": step.get("step_id", step.get("id", "")),
            "description": step.get("description", step.get("desc", "")),
            "tool_name": tool,
            "arguments": args,
        }
    return None


def _normalize_plan(plan: Any) -> list[dict[str, Any]] | None:
    """Accept TaskPlan / list[PlanStep] / list[dict]; None if unusable."""
    if TaskPlan is not None and isinstance(plan, TaskPlan):
        steps = plan.steps
    elif isinstance(plan, (list, tuple)):
        steps = list(plan)
    else:
        return None
    normalized: list[dict[str, Any]] = []
    for step in steps:
        norm = _normalize_step(step)
        if norm is None:
            return None
        normalized.append(norm)
    return normalized


# ---------------------------------------------------------------------------
# Plan verifier
# ---------------------------------------------------------------------------


class PlanVerifier:
    """Verifies LLM-generated plans against the real tool registry."""

    def __init__(self, tools_registry: Any, llm: Any = None,
                 max_steps: int = MAX_PLAN_STEPS) -> None:
        self.registry = tools_registry
        self.llm = llm
        self.max_steps = max_steps

    # -- public ----------------------------------------------------------

    def verify_plan(self, plan: Any, goal: str) -> VerificationResult:
        """Check a plan; never raises on bad input, reports issues."""
        issues: list[str] = []
        registry = self.registry

        # Registry itself unusable — nothing the LLM can fix.
        tool_names: set[str] = set()
        if registry is not None:
            try:
                tool_names = set(registry.names())
            except Exception:
                tool_names = set()
        if not tool_names:
            return VerificationResult(
                passed=False,
                issues=["tool registry is empty or unavailable; "
                        "no plan can be verified"],
                retryable=False,
                checked_steps=0,
            )

        steps = _normalize_plan(plan)
        if steps is None:
            return VerificationResult(
                passed=False,
                issues=["plan is not a list of steps "
                        f"(got {type(plan).__name__})"],
                retryable=True,
                checked_steps=0,
            )
        if not steps:
            return VerificationResult(
                passed=False,
                issues=["plan is empty; the goal needs at least one step"],
                retryable=True,
                checked_steps=0,
            )
        if len(steps) > self.max_steps:
            issues.append(
                f"plan has {len(steps)} steps, exceeding the sanity bound "
                f"of {self.max_steps}; trim or split the task")

        for index, step in enumerate(steps):
            self._verify_step(step, index, tool_names, issues)

        self._verify_ordering(steps, issues)
        self._verify_duplicates(steps, issues)

        passed = not issues
        return VerificationResult(
            passed=passed,
            issues=issues,
            retryable=True,  # registry is usable; regeneration can fix these
            checked_steps=len(steps),
        )

    def feedback_text(self, result: VerificationResult) -> str:
        """Build the 'try again avoiding these mistakes' prompt chunk."""
        if result.passed:
            return ""
        lines = ["Your last plan failed verification for these reasons:"]
        for issue in result.issues:
            lines.append(f"- {issue}")
        lines.append("Generate a corrected plan that avoids ALL of these "
                     "mistakes. Use ONLY tools from the available list.")
        return "\n".join(lines)

    # -- per-step checks -------------------------------------------------

    def _verify_step(self, step: dict[str, Any], index: int,
                     tool_names: set[str], issues: list[str]) -> None:
        label = f"step {index + 1}"
        tool_name = str(step.get("tool_name") or "").strip()
        if not tool_name:
            issues.append(f"{label}: missing tool name")
            return
        if tool_name not in tool_names:
            suggestion = self._suggest_tool(tool_name, tool_names)
            hint = f" Did you mean {suggestion!r}?" if suggestion else ""
            issues.append(
                f"{label}: unknown tool {tool_name!r} (hallucinated).{hint} "
                f"Available tools include: "
                f"{', '.join(sorted(tool_names)[:12])}"
                f"{'...' if len(tool_names) > 12 else ''}")
            return
        # Tool exists — check its required arguments.
        tool = self.registry.get_or_none(tool_name)
        arguments = step.get("arguments") or {}
        if not isinstance(arguments, dict):
            issues.append(f"{label}: arguments must be an object, "
                          f"got {type(arguments).__name__}")
            return
        if tool is not None:
            try:
                tool.validate_arguments(dict(arguments))
            except Exception as exc:  # missing/invalid args
                issues.append(f"{label}: tool {tool_name!r}: {exc}")

    @staticmethod
    def _suggest_tool(name: str, tool_names: set[str]) -> str | None:
        matches = difflib.get_close_matches(
            name, sorted(tool_names), n=1, cutoff=0.6)
        return matches[0] if matches else None

    # -- ordering checks -------------------------------------------------

    def _verify_ordering(self, steps: list[dict[str, Any]],
                         issues: list[str]) -> None:
        """No 'submit' before any 'fill'; generic logical-order sanity."""
        def kinds(step: dict[str, Any]) -> set[str]:
            text = (str(step.get("tool_name") or "") + " "
                    + str(step.get("description") or "")).lower()
            found = set()
            if any(k in text for k in _FILL_KEYWORDS):
                found.add("fill")
            if any(k in text for k in _SUBMIT_KEYWORDS):
                found.add("submit")
            return found

        seen_fill = False
        for index, step in enumerate(steps):
            k = kinds(step)
            if "fill" in k:
                seen_fill = True
            if "submit" in k and not seen_fill:
                issues.append(
                    f"step {index + 1}: submits/finalizes before any "
                    f"fill/type step — fill the form first")
                seen_fill = True  # only report the first occurrence

    def _verify_duplicates(self, steps: list[dict[str, Any]],
                           issues: list[str]) -> None:
        """Flag immediately repeated identical steps (likely a loop)."""
        for index in range(1, len(steps)):
            prev, cur = steps[index - 1], steps[index]
            if (prev.get("tool_name") == cur.get("tool_name")
                    and prev.get("arguments") == cur.get("arguments")):
                issues.append(
                    f"step {index + 1}: repeats the previous step "
                    f"({cur.get('tool_name')!r} with identical arguments) — "
                    f"probable loop")


# ---------------------------------------------------------------------------
# Plan generation + repair loop
# ---------------------------------------------------------------------------


def _default_plan_prompt(goal: str, tool_names: Sequence[str],
                         feedback: str | None) -> str:
    catalog = ", ".join(tool_names)
    prompt = (
        "You are a precise task planner. Break the goal into concrete steps.\n"
        f"Available tools (use ONLY these exact names): {catalog}\n"
        f"Goal: {goal}\n"
        "Reply with ONLY a JSON array. Each step is an object with:\n"
        '  {"step_id": "1", "description": "...", '
        '"tool_name": "<exact tool name>", "arguments": {...}}\n'
        "Keep it short: no more than 10 steps."
    )
    if feedback:
        prompt += f"\n\n{feedback}"
    return prompt


def _extract_json_array(text: str) -> list[dict[str, Any]] | None:
    """Pull the first JSON array out of free-form LLM output."""
    if not text:
        return None
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except Exception:
        return None
    return data if isinstance(data, list) else None


def _default_generate_plan(llm: Any, goal: str, tool_names: Sequence[str],
                           feedback: str | None = None
                           ) -> list[dict[str, Any]]:
    """Ask the LLM for a JSON plan.  Returns [] when unparseable."""
    prompt = _default_plan_prompt(goal, tool_names, feedback)
    try:
        if hasattr(llm, "generate"):
            raw = llm.generate(prompt)
        else:  # LLMProvider.chat
            raw = llm.chat([{"role": "user", "content": prompt}])
    except Exception:
        return []
    parsed = _extract_json_array(str(raw))
    if parsed is None:
        return []
    steps: list[dict[str, Any]] = []
    for item in parsed:
        norm = _normalize_step(item)
        if norm is not None:
            steps.append(norm)
    return steps


def _attempt_score(plan: list[dict[str, Any]],
                   result: VerificationResult,
                   tool_names: Sequence[str]) -> tuple[int, int, int]:
    """Lower is better: (issues, -valid_tools, -checked_steps)."""
    valid_tools = sum(
        1 for step in plan
        if str(step.get("tool_name") or "") in tool_names)
    return (len(result.issues), -valid_tools, -result.checked_steps)


def verify_then_repair(
    llm: Any,
    verifier: PlanVerifier,
    goal: str,
    max_attempts: int = 3,
    plan_fn: Callable[..., list[dict[str, Any]]] | None = None,
) -> RepairOutcome:
    """Produce → verify → repair loop.

    Generates a plan, verifies it, and — when it fails but is retryable —
    regenerates with the verifier's feedback appended.  Bounded by
    *max_attempts*; returns the best attempt (fewest issues) when none
    passes cleanly.

    *plan_fn* defaults to asking *llm* for a JSON plan; pass a custom
    callable ``(llm, goal, tool_names, feedback) -> plan`` to override.
    """
    plan_fn = plan_fn or _default_generate_plan
    try:
        tool_names: Sequence[str] = verifier.registry.names()
    except Exception:
        tool_names = []

    best_plan: list[dict[str, Any]] = []
    best_result = VerificationResult(
        passed=False, issues=["no plan generated"], retryable=False)
    best_score: tuple[int, int, int] | None = None
    feedback: str | None = None
    attempts = 0

    for _ in range(max(1, max_attempts)):
        attempts += 1
        try:
            plan = plan_fn(llm, goal, tool_names, feedback)
        except Exception as exc:
            plan = []
            best_result = VerificationResult(
                passed=False,
                issues=[f"plan generation raised: {exc}"],
                retryable=True,
            )
            feedback = verifier.feedback_text(best_result)
            continue
        if not isinstance(plan, list):
            plan = []
        result = verifier.verify_plan(plan, goal)

        score = _attempt_score(plan, result, tool_names)
        if best_score is None or score < best_score:
            best_score = score
            best_plan, best_result = plan, result

        if result.passed:
            return RepairOutcome(
                plan=plan, result=result, attempts=attempts,
                repaired=attempts > 1, issues=[],
            )
        if not result.retryable:
            break
        feedback = verifier.feedback_text(result)

    return RepairOutcome(
        plan=best_plan,
        result=best_result,
        attempts=attempts,
        repaired=False,
        issues=list(best_result.issues),
    )


# ---------------------------------------------------------------------------
# Answer verifier (chat replies before speaking)
# ---------------------------------------------------------------------------


def _question_keywords(question: str) -> list[str]:
    words = re.findall(r"[a-zA-Z\u0600-\u06FF]{4,}", question.lower())
    return [w for w in words if w not in _STOPWORDS]


class AnswerVerifier:
    """Checks that a reply actually addresses the question.

    Heuristics run first (cheap, no LLM).  When an *llm* is supplied, a
    passing heuristic answer gets one confirming LLM judgement call.
    """

    def __init__(self, llm: Any = None) -> None:
        self.llm = llm

    def verify_answer(self, question: str, answer: str) -> bool:
        """True when the answer is acceptable to speak."""
        ok, _reasons = self.verify_answer_with_reasons(question, answer)
        return ok

    def verify_answer_with_reasons(
        self, question: str, answer: str
    ) -> tuple[bool, list[str]]:
        """Like :meth:`verify_answer` but also explains failures."""
        reasons: list[str] = []
        text = (answer or "").strip()

        if not text:
            return False, ["answer is empty"]
        if len(text) < 3:
            reasons.append("answer is too short to be meaningful")
            return False, reasons
        lowered = text.lower()
        for pattern in _DODGE_PATTERNS:
            if re.search(pattern, lowered):
                reasons.append("answer is a generic dodge, not a real reply")
                return False, reasons

        keywords = _question_keywords(question or "")
        if keywords:
            hits = sum(1 for kw in keywords if kw in lowered)
            if hits == 0:
                reasons.append(
                    "answer shares no significant words with the question "
                    f"(question keywords: {', '.join(keywords[:6])})")
                return False, reasons

        # Heuristics passed — optional LLM confirmation.
        if self.llm is not None:
            try:
                judgement = self.llm.generate(
                    "Does the following ANSWER address the QUESTION? "
                    "Reply with exactly YES or NO.\n"
                    f"QUESTION: {question}\nANSWER: {answer}"
                )
                if "yes" not in str(judgement).strip().lower()[:10]:
                    reasons.append("LLM judge: answer does not address "
                                   "the question")
                    return False, reasons
            except Exception:
                pass  # judge unavailable — heuristics already passed

        return True, reasons

    def repair_answer(self, question: str, answer: str,
                      max_attempts: int = 2) -> tuple[str, bool]:
        """Regenerate a failing answer with failure reasons appended.

        Returns (answer, repaired_ok).  Needs an LLM; without one the
        original answer is returned unchanged.
        """
        if self.llm is None:
            return answer, self.verify_answer(question, answer)
        current = answer
        for _ in range(max(1, max_attempts)):
            ok, reasons = self.verify_answer_with_reasons(question, current)
            if ok:
                return current, True
            feedback = ("Your previous answer was rejected because:\n"
                        + "\n".join(f"- {r}" for r in reasons)
                        + "\nAnswer the question directly and completely.")
            try:
                current = self.llm.generate(
                    f"QUESTION: {question}\n\n{feedback}")
            except Exception:
                break
        return current, self.verify_answer(question, current)

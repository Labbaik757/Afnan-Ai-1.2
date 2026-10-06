"""Failure classification and root-cause analysis.

Structured taxonomy (hierarchical): a failure is
classified to a FailureKind with a hierarchy path like
["execution_failure", "browser", "click",
"element_not_interactable"], plus phase, severity,
recoverability and a root-cause hypothesis with
confidence — never just the exception message.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.evaluation.models import (
    FailureKind,
    FailureRecord,
)

# Keyword signals → (kind, hierarchy fragment, root cause).
_SIGNALS: list[
    tuple[tuple[str, ...], FailureKind, list[str], str]
] = [
    (
        ("plan", "no steps", "invalid plan"),
        FailureKind.PLANNING_FAILURE,
        ["planning", "decomposition"],
        "planner produced an unusable plan",
    ),
    (
        ("unknown tool", "no such tool", "tool not found"),
        FailureKind.TOOL_SELECTION_FAILURE,
        ["tool_selection", "registry"],
        "requested tool is not registered or was misspelled",
    ),
    (
        ("stale", "element_not_interactable", "not clickable"),
        FailureKind.BROWSER_FAILURE,
        ["execution_failure", "browser", "click",
         "element_not_interactable"],
        "locator quality or stale DOM; page state mismatch",
    ),
    (
        ("timeout", "timed out"),
        FailureKind.TIMEOUT,
        ["execution_failure", "timeout"],
        "operation exceeded its budget; check waits and budgets",
    ),
    (
        ("captcha",),
        FailureKind.ENVIRONMENT_FAILURE,
        ["environment_failure", "captcha"],
        "human verification required; not an agent failure",
    ),
    (
        ("rate limit", "429", "too many requests"),
        FailureKind.RESOURCE_EXHAUSTION,
        ["resource_exhaustion", "rate_limit"],
        "rate budget exhausted; back off or replan",
    ),
    (
        ("permission", "denied", "unauthorized"),
        FailureKind.PERMISSION_FAILURE,
        ["permission_failure", "gate"],
        "action needed approval that was not granted",
    ),
    (
        ("injection", "prompt injection"),
        FailureKind.SECURITY_FAILURE,
        ["security_failure", "injection"],
        "untrusted content treated as instruction",
    ),
    (
        ("secret", "credential", "api key"),
        FailureKind.SECURITY_FAILURE,
        ["security_failure", "secret_exposure"],
        "secret handling boundary violated",
    ),
    (
        ("connector", "api error"),
        FailureKind.CONNECTOR_FAILURE,
        ["connector_failure", "service"],
        "external service failed; not an agent logic failure",
    ),
    (
        ("subagent",),
        FailureKind.SUBAGENT_FAILURE,
        ["subagent_failure", "delegation"],
        "delegated work failed; check scoping and handoff",
    ),
    (
        ("artifact",),
        FailureKind.ARTIFACT_FAILURE,
        ["artifact_failure", "build"],
        "artifact build or verification failed",
    ),
    (
        ("citation",),
        FailureKind.RESEARCH_FAILURE,
        ["research_failure", "citation"],
        "citation integrity check failed",
    ),
    (
        ("memory",),
        FailureKind.MEMORY_FAILURE,
        ["memory_failure", "retrieval"],
        "memory retrieval or promotion failed",
    ),
    (
        ("context", "token"),
        FailureKind.CONTEXT_FAILURE,
        ["context_failure", "budget"],
        "context budget exceeded or compression lost data",
    ),
    (
        ("recovery", "replan"),
        FailureKind.RECOVERY_FAILURE,
        ["recovery_failure", "strategy"],
        "recovery strategy did not resolve the fault",
    ),
    (
        ("verify", "verification"),
        FailureKind.VERIFICATION_FAILURE,
        ["verification_failure", "judgement"],
        "verifier rejected the outcome; inspect evidence",
    ),
    (
        ("computer", "desktop", "xdotool"),
        FailureKind.COMPUTER_USE_FAILURE,
        ["execution_failure", "computer_use"],
        "desktop backend or locator failed",
    ),
]


class FailureClassifier:
    """Classify failures from error text + context."""

    def classify(
        self,
        *,
        error: str,
        phase: str = "",
        action: str = "",
        run_id: str = "",
        case_id: str = "",
        trajectory_ref: str = "",
        environment: str = "",
        recovery_attempted: bool = False,
        recovery_outcome: str = "",
    ) -> FailureRecord:
        lowered = f"{error} {action}".lower()
        kind = FailureKind.UNKNOWN_FAILURE
        hierarchy: list[str] = ["unknown"]
        root_cause = "unclassified failure; inspect trajectory"
        confidence = 0.3
        for keywords, k, hier, cause in _SIGNALS:
            if any(kw in lowered for kw in keywords):
                kind = k
                hierarchy = hier
                root_cause = cause
                confidence = 0.75
                break

        recoverable = kind not in (
            FailureKind.SECURITY_FAILURE,
            FailureKind.USER_INPUT_FAILURE,
        )
        severity = "high" if kind in (
            FailureKind.SECURITY_FAILURE,
            FailureKind.PERMISSION_FAILURE,
        ) else "medium"

        return FailureRecord(
            run_id=run_id,
            case_id=case_id,
            kind=kind,
            phase=phase,
            action=action,
            error=error,
            trajectory_ref=trajectory_ref,
            environment=environment,
            severity=severity,
            recoverable=recoverable,
            recovery_attempted=recovery_attempted,
            recovery_outcome=recovery_outcome,
            root_cause=root_cause,
            root_cause_confidence=confidence,
            hierarchy=hierarchy,
        )

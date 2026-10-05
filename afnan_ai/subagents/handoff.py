"""Result handoff verification.

A subagent's result is never trusted automatically.  The
parent — or a dedicated verifier agent — checks the
structured handoff against the expected result:

* output present and non-empty,
* evidence recorded (sources, tool results),
* confidence above threshold,
* no contradiction between evidence and output
  (keyword-overlap check, deterministic),
* no injection markers in the output.

The handoff is marked verified / failed / uncertain.  Only
verified handoffs are merged into the parent's result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from afnan_ai.subagents.models import (
    SubAgentHandoff,
    VerificationState,
)

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on",
    "for", "with", "is", "are", "was", "were", "be", "it",
    "this", "that", "as", "at", "by", "from",
})


def _keywords(text: Any) -> set[str]:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return {
        w for w in words
        if len(w) > 2 and w not in _STOPWORDS
    }


def _scan_injection():
    from afnan_ai.agent_loop import scan_for_injection

    return scan_for_injection


@dataclass
class VerificationVerdict:
    handoff_id: str
    state: VerificationState
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "handoff_id": self.handoff_id,
            "state": self.state.value,
            "reasons": list(self.reasons),
            "confidence": self.confidence,
        }


class HandoffVerifier:
    """Deterministic handoff checks (no model calls)."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.5,
        require_evidence: bool = True,
    ) -> None:
        self.min_confidence = min_confidence
        self.require_evidence = require_evidence

    def verify(
        self,
        handoff: SubAgentHandoff,
        *,
        expected: str = "",
    ) -> VerificationVerdict:
        reasons: list[str] = []
        scan = _scan_injection()

        if handoff.status != "completed":
            return VerificationVerdict(
                handoff.subagent_id,
                VerificationState.FAILED,
                [f"subagent did not complete "
                 f"({handoff.status})"],
                0.0,
            )
        if handoff.output in (None, "", [], {}):
            reasons.append("empty output")
        if self.require_evidence and not handoff.evidence:
            reasons.append("no evidence recorded")
        if handoff.confidence < self.min_confidence:
            reasons.append(
                f"confidence {handoff.confidence:.2f} below "
                f"{self.min_confidence:.2f}"
            )
        if scan(str(handoff.output)):
            return VerificationVerdict(
                handoff.subagent_id,
                VerificationState.FAILED,
                ["injection markers in subagent output"],
                0.0,
            )
        # Contradiction check: output keywords should overlap
        # evidence keywords; expected-result keywords should
        # appear in the output.
        output_keys = _keywords(handoff.output)
        evidence_keys = _keywords(" ".join(handoff.evidence))
        if (
            output_keys
            and evidence_keys
            and not (output_keys & evidence_keys)
        ):
            reasons.append(
                "output shares no keywords with evidence"
            )
        if expected:
            expected_keys = _keywords(expected)
            if (
                expected_keys
                and not (expected_keys & output_keys)
            ):
                reasons.append(
                    "output does not address the expected "
                    "result"
                )
        if reasons:
            # Uncertain, not failed: the parent may still use
            # it with caution or ask for a re-run.
            return VerificationVerdict(
                handoff.subagent_id,
                VerificationState.UNCERTAIN,
                reasons,
                handoff.confidence,
            )
        return VerificationVerdict(
            handoff.subagent_id,
            VerificationState.VERIFIED,
            ["output present", "evidence recorded",
             "no contradictions"],
            handoff.confidence,
        )

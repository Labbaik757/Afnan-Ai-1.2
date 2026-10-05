"""Skill learning — from verified success to candidates.

The learner watches completed tasks and notices when the
same verified tool sequence succeeds repeatedly.  Such a
pattern becomes a *candidate* — never a live skill:

* failed workflows never become candidates,
* only verified steps count (a step that merely "ran" is
  not evidence),
* candidate steps carry tool names only — never raw
  external text, never model assumptions,
* candidates are versioned drafts with provenance (the
  task ids they were learned from), so every promotion is
  auditable,
* promotion still goes through full validation and
  explicit registration (human approval for sensitive+).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from afnan_ai.log_config import get_logger
from afnan_ai.skills.composer import compose_skill
from afnan_ai.skills.models import SkillCandidate, SkillStep

logger = get_logger(__name__)


class SkillLearner:
    """Collect verified patterns; propose candidates."""

    def __init__(
        self,
        *,
        min_successes: int = 2,
        max_pattern_len: int = 6,
        min_pattern_len: int = 2,
    ) -> None:
        self.min_successes = max(1, min_successes)
        self.max_pattern_len = max_pattern_len
        self.min_pattern_len = min_pattern_len
        # pattern (tool tuple) -> {"count": n, "tasks": [ids]}
        self._patterns: dict[tuple[str, ...], dict] = {}
        self._seen_candidates: set[tuple[str, ...]] = set()

    def observe_task(
        self,
        task_id: str,
        steps: list[dict[str, Any]],
        *,
        outcome: str,
    ) -> None:
        """Feed one finished task.

        ``steps``: ``{"tool": name, "verified": bool}`` in
        execution order.  Only COMPLETED tasks with verified
        steps contribute patterns.
        """
        if outcome != "completed":
            return
        verified_tools = [
            str(s.get("tool", ""))
            for s in steps
            if s.get("verified") and str(s.get("tool", ""))
        ]
        if len(verified_tools) < self.min_pattern_len:
            return
        pattern = tuple(
            verified_tools[: self.max_pattern_len]
        )
        record = self._patterns.setdefault(
            pattern, {"count": 0, "tasks": []}
        )
        if task_id not in record["tasks"]:
            record["tasks"].append(task_id)
            record["count"] += 1

    def candidates(self) -> list[SkillCandidate]:
        """Patterns seen often enough to propose as skills."""
        out: list[SkillCandidate] = []
        for pattern, record in self._patterns.items():
            if (
                record["count"] >= self.min_successes
                and pattern not in self._seen_candidates
            ):
                self._seen_candidates.add(pattern)
                out.append(SkillCandidate(
                    candidate_id=(
                        "cand_" + "_".join(pattern[:3])
                    ),
                    description=(
                        "Learned workflow: "
                        + " → ".join(pattern)
                    ),
                    steps=[
                        SkillStep(
                            step_id=f"s{i + 1}",
                            tool=tool,
                            arguments={},
                            description=(
                                f"Learned step: {tool}"
                            ),
                        )
                        for i, tool in enumerate(pattern)
                    ],
                    evidence_task_ids=list(record["tasks"]),
                    success_count=record["count"],
                ))
        return out

    def promote(
        self,
        candidate: SkillCandidate,
        *,
        skill_id: str,
        name: str,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
    ):
        """Turn a candidate into a validated, unregistered
        Skill draft (registration stays explicit)."""
        from afnan_ai.skills.registry import SkillRegistry

        skill = compose_skill(
            skill_id=skill_id,
            name=name,
            description=(
                description or candidate.description
            ),
            steps=[
                {
                    "step_id": step.step_id,
                    "tool": step.tool,
                    "arguments": {},
                    "description": (
                        f"Learned from tasks "
                        f"{candidate.evidence_task_ids[:3]}"
                    ),
                }
                for step in candidate.steps
            ],
            input_schema=input_schema,
            dependencies={
                "tools": sorted({
                    step.tool for step in candidate.steps
                }),
            },
            created_by="learner",
        )
        skill.changelog.append(
            f"promoted from candidate {candidate.candidate_id} "
            f"(evidence: {candidate.evidence_task_ids})"
        )
        return skill

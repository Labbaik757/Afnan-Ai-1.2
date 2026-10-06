"""Research planning: decompose complex questions.

Turns one research question into a deterministic,
persisted ResearchPlan: objectives with subquestions,
preferred source classes, freshness, diversity and
budget.  No LLM required for the structural
decomposition — the planner extracts facets from the
question text deterministically; the AgentLoop/LLM can
refine objectives later.
"""

from __future__ import annotations

import re
from typing import Any

from afnan_ai.research.models import (
    ResearchPlan,
    ResearchQuestion,
    ResearchTask,
    SourceClass,
)

# Facet patterns → objective templates.
_FACETS: list[tuple[str, list[str], str]] = [
    (
        "capabilities",
        [r"\bcapabilit", r"\bfeatures?\b", r"\bwhat can\b"],
        "Current capabilities",
    ),
    (
        "limitations",
        [r"\blimitations?\b", r"\bdrawbacks?\b", r"\bcons\b"],
        "Limitations and drawbacks",
    ),
    (
        "latest",
        [
            r"\blatest\b", r"\bcurrent state\b", r"\brecent\b",
            r"\bnew\b", r"\b202[4-9]\b",
        ],
        "Latest developments",
    ),
    (
        "risks",
        [r"\brisks?\b", r"\bsecurity\b", r"\bthreats?\b"],
        "Risks and security considerations",
    ),
    (
        "comparison",
        [
            r"\bvs\b", r"\bversus\b", r"\bcompar",
            r"\balternatives?\b",
        ],
        "Comparison with alternatives",
    ),
    (
        "pricing",
        [r"\bpricing\b", r"\bcost\b", r"\bprice\b"],
        "Pricing and cost",
    ),
]

_SOURCE_CLASS_HINTS: list[tuple[SourceClass, list[str]]] = [
    (SourceClass.OFFICIAL_DOC, [r"\bofficial\b", r"\bdocs?\b"]),
    (SourceClass.ACADEMIC, [r"\bstudy\b", r"\bresearch\b",
                            r"\bpaper\b", r"\bacademic\b"]),
    (SourceClass.GOVERNMENT, [r"\bregulation\b", r"\blaw\b",
                              r"\bgovernment\b"]),
    (SourceClass.COMPANY, [r"\bcompany\b", r"\bvendor\b"]),
    (SourceClass.JOURNALISM, [r"\bnews\b", r"\breport\b"]),
]

_DEFAULT_BUDGET = {
    "max_sources": 20,
    "max_searches": 12,
    "max_concurrent_tasks": 4,
    "max_subagents": 3,
    "max_depth": 2,
    "time_budget_s": 1800,
    "max_tokens": 60000,
}


def _match(patterns: list[str], text: str) -> bool:
    return any(
        re.search(p, text, re.IGNORECASE) for p in patterns
    )


def decompose_question(
    question: str,
    *,
    budget: dict[str, Any] | None = None,
    session_id: str = "",
) -> ResearchPlan:
    """Build a deterministic research plan from a question."""
    q = ResearchQuestion(text=question)
    lowered = question.lower()
    objectives: list[dict[str, Any]] = []

    matched_facets = [
        (key, title)
        for key, patterns, title in _FACETS
        if _match(patterns, lowered)
    ]
    if not matched_facets:
        # Generic decomposition: overview + evidence + outlook.
        matched_facets = [
            ("overview", "Overview and key facts"),
            ("evidence", "Independent evidence"),
            ("outlook", "Outlook and open questions"),
        ]

    for key, title in matched_facets:
        source_classes = ["primary", "official_documentation"]
        for cls, patterns in _SOURCE_CLASS_HINTS:
            if _match(patterns, lowered):
                source_classes.append(cls.value)
        source_classes += ["journalism", "technical_analysis"]
        # Deduplicate, preserve order.
        seen: set[str] = set()
        source_classes = [
            c for c in source_classes if not (c in seen or seen.add(c))
        ]
        objectives.append(
            {
                "objective": title,
                "subquestion": f"{title}: {question[:120]}",
                "source_classes": source_classes,
                "freshness_days": 365,
                "rationale": f"facet '{key}' detected",
            }
        )
        q.subquestions.append(
            f"{title}: {question[:120]}"
        )

    plan = ResearchPlan(session_id=session_id)
    plan.objectives = objectives
    plan.budget = {**_DEFAULT_BUDGET, **(budget or {})}

    # One discovery task per objective, then shared
    # acquire/extract/verify/synthesize tasks.
    for i, obj in enumerate(objectives):
        plan.tasks.append(
            ResearchTask(
                session_id=session_id,
                kind="discover",
                description=(
                    f"Discover sources: {obj['objective']}"
                ),
            )
        )
    plan.tasks.append(
        ResearchTask(
            session_id=session_id,
            kind="acquire",
            description="Acquire discovered sources",
        )
    )
    plan.tasks.append(
        ResearchTask(
            session_id=session_id,
            kind="extract",
            description="Extract evidence from sources",
        )
    )
    plan.tasks.append(
        ResearchTask(
            session_id=session_id,
            kind="verify",
            description=(
                "Cross-source verification, contradictions, "
                "citations"
            ),
        )
    )
    plan.tasks.append(
        ResearchTask(
            session_id=session_id,
            kind="synthesize",
            description="Synthesize verified research report",
        )
    )
    return plan


def replan_after_failure(
    plan: ResearchPlan, failed_task: ResearchTask, reason: str
) -> ResearchPlan:
    """Add an alternate-method task after a failure."""
    alternate = ResearchTask(
        session_id=plan.session_id,
        kind=failed_task.kind,
        description=(
            f"Alternate approach for '{failed_task.description[:80]}': "
            f"{reason[:120]}"
        ),
    )
    plan.tasks.append(alternate)
    return plan

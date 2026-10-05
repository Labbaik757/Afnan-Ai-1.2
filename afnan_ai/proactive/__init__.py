"""Proactive intelligence & ideas for Afnan AI.

A controlled intelligence layer — not a replacement for
the central agent.  The ProactiveEngine watches authorized
state (goals, tasks, schedules, memories, recent activity),
detects evidence-based opportunities, and surfaces them as
Ideas for the user to accept or dismiss.  Accepted ideas
become normal TaskManager tasks executed by the existing
AgentLoop; nothing auto-executes except explicitly
configured read-only low-risk actions, and
sensitive/irreversible suggestions always need human
approval.
"""

from afnan_ai.proactive.engine import (
    ProactiveConfig,
    ProactiveEngine,
    ProactiveError,
)
from afnan_ai.proactive.models import (
    RISK_DESTRUCTIVE,
    RISK_READ_ONLY,
    RISK_REVERSIBLE,
    RISK_SENSITIVE,
    Idea,
    IdeaStatus,
    SuggestionType,
)

__all__ = [
    "Idea",
    "IdeaStatus",
    "ProactiveConfig",
    "ProactiveEngine",
    "ProactiveError",
    "RISK_DESTRUCTIVE",
    "RISK_READ_ONLY",
    "RISK_REVERSIBLE",
    "RISK_SENSITIVE",
    "SuggestionType",
]

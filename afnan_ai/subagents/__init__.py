"""Multi-agent / subagent architecture for Afnan AI.

Complex goals divide into specialized, least-privilege
subagents that each run the *existing* AgentLoop against a
scoped tool view — no new orchestration layer, no separate
planner/executor/verifier.

* :class:`SubAgentSpec` — id, role, objective, allowed
  tools/connectors, context scope, risk permissions,
  status, result, verification state.
* :class:`SubAgentManager` — create/start/pause/resume/
  cancel/terminate; isolated task contexts; structured
  handoffs; permissions never exceed the parent's.
* :class:`TaskDecomposer` — single vs multi-agent decision
  and capability-based goal splitting (roles are permission
  bundles, not hard-coded workflows).
* :class:`ScopedToolRegistry` — read-only least-privilege
  tool view with risk and connector filtering plus
  tool-call budgets.
* :class:`ResourceLockManager` — shared-resource locks so
  parallel subagents cannot corrupt the same browser tab,
  file or connector session.
* :class:`SubAgentMailbox` — parent-mediated messaging; no
  hidden channels.
* :class:`HandoffVerifier` — subagent results are verified,
  never trusted.
"""

from afnan_ai.subagents.communication import (
    MessageRejected,
    SubAgentMailbox,
    SubAgentMessage,
)
from afnan_ai.subagents.decomposition import (
    DecompositionProposal,
    TaskDecomposer,
)
from afnan_ai.subagents.handoff import (
    HandoffVerifier,
    VerificationVerdict,
)
from afnan_ai.subagents.locks import (
    ResourceConflict,
    ResourceLockManager,
)
from afnan_ai.subagents.manager import (
    SubAgentError,
    SubAgentManager,
)
from afnan_ai.subagents.models import (
    ROLE_TEMPLATES,
    ResourceLimits,
    SubAgentHandoff,
    SubAgentRecord,
    SubAgentSpec,
    SubAgentStatus,
    VerificationState,
)
from afnan_ai.subagents.scoped_registry import ScopedToolRegistry
from afnan_ai.subagents.security import (
    SecurityIssue,
    SecurityReport,
    SubAgentSecurity,
)

__all__ = [
    "ROLE_TEMPLATES",
    "DecompositionProposal",
    "HandoffVerifier",
    "MessageRejected",
    "ResourceConflict",
    "ResourceLimits",
    "ResourceLockManager",
    "ScopedToolRegistry",
    "SecurityIssue",
    "SecurityReport",
    "SubAgentError",
    "SubAgentHandoff",
    "SubAgentMailbox",
    "SubAgentManager",
    "SubAgentMessage",
    "SubAgentRecord",
    "SubAgentSecurity",
    "SubAgentSpec",
    "SubAgentStatus",
    "TaskDecomposer",
    "VerificationState",
    "VerificationVerdict",
]

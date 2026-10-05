"""Dynamic Tool & Skill Builder for Afnan AI.

Skills are named, versioned, reusable workflows composed of
*existing* registered tools (and other skills) — never
arbitrary model-generated code.

* :class:`Skill` — the abstraction (id, name, description,
  risk, schemas, version, dependencies, steps, verification
  criteria).
* :class:`SkillRegistry` — dynamic register / discover /
  update / version / disable, structured agent-facing
  descriptions, dependency checks, audit trail.  A Skill is
  never a Tool; the registry bridges them one-way via the
  ``SkillTool`` adapter.
* :func:`compose_skill` — build a skill from tool steps.
* :class:`SkillGenerator` — identify capability → search
  existing → generate composition → strict validation →
  risk analysis → sandbox validation → explicit
  registration (the model never executes code).
* :class:`SkillLearner` — verified successful workflows
  become *candidates*, never live skills; promotion stays
  explicit and audited.
* :class:`SkillExecutor` — runs skills through the normal
  ``ToolRegistry.execute`` path (no separate loop);
  sensitive/destructive skills go through the existing
  human-approval mechanism.
* :class:`SandboxedPython` — restricted subprocess for the
  human-gated code-skill path (no network, no secrets,
  timeout, CPU/memory limits).
"""

from afnan_ai.skills.composer import (
    compose_skill,
    research_report_skill,
)
from afnan_ai.skills.executor import (
    SkillExecutionError,
    SkillExecutor,
    SkillTool,
)
from afnan_ai.skills.generator import SkillDraft, SkillGenerator
from afnan_ai.skills.learner import SkillLearner
from afnan_ai.skills.models import (
    Skill,
    SkillCandidate,
    SkillDependencies,
    SkillRisk,
    SkillStatus,
    SkillStep,
)
from afnan_ai.skills.registry import SkillError, SkillRegistry
from afnan_ai.skills.risk import (
    classify_tool,
    derive_risk,
    effective_risk,
    needs_approval,
    risk_report,
)
from afnan_ai.skills.sandbox import (
    SandboxedPython,
    SandboxConfig,
    SandboxResult,
    SkillValidation,
    ValidationIssue,
)

__all__ = [
    "Skill",
    "SkillCandidate",
    "SkillDependencies",
    "SkillDraft",
    "SkillError",
    "SkillExecutionError",
    "SkillExecutor",
    "SkillGenerator",
    "SkillLearner",
    "SkillRegistry",
    "SkillRisk",
    "SkillStatus",
    "SkillStep",
    "SkillTool",
    "SandboxConfig",
    "SandboxResult",
    "SandboxedPython",
    "SkillValidation",
    "ValidationIssue",
    "classify_tool",
    "compose_skill",
    "derive_risk",
    "effective_risk",
    "needs_approval",
    "research_report_skill",
    "risk_report",
]

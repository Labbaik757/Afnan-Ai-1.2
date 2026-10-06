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

from afnan_ai.skills.conditions import (
    StepCondition,
    evaluate_predicate,
    loop_items,
    should_run_step,
)
from afnan_ai.skills.contracts import (
    ContractCheck,
    ContractReport,
    check_postconditions,
    check_preconditions,
)
from afnan_ai.skills.graph import DependencyError, DependencyGraph
from afnan_ai.skills.history import (
    SkillExecutionRecord,
    SkillHistory,
)
from afnan_ai.skills.importer import ImportReport, import_skill
from afnan_ai.skills.integrity import (
    manifest_hash,
    sign_manifest,
    skill_hash,
    verify_signature,
    verify_skill,
)
from afnan_ai.skills.manifest import (
    ExecutionLimits,
    SkillCompatibility,
    SkillManifest,
    SkillSource,
    VerificationStatus,
    manifest_from_skill,
)
from afnan_ai.skills.marketplace import (
    MarketplaceEntry,
    entry_from_manifest,
)
from afnan_ai.skills.observability import emit_activity, emit_audit
from afnan_ai.skills.optimize import (
    OptimizationSuggestion,
    suggest_optimizations,
)
from afnan_ai.skills.security import (
    SecurityFinding,
    SecurityReport,
    analyze_skill,
)
from afnan_ai.skills.toolsmith import (
    ToolBuilder,
    ToolProposal,
    ToolProposalStatus,
)
from afnan_ai.skills.versions import (
    VersionChange,
    bump,
    compare,
    diff_schemas,
    diff_skills,
    is_valid,
    latest,
    parse,
)

__all__ = [
    "ContractCheck",
    "ContractReport",
    "DependencyError",
    "DependencyGraph",
    "ExecutionLimits",
    "ImportReport",
    "MarketplaceEntry",
    "OptimizationSuggestion",
    "SecurityFinding",
    "SecurityReport",
    "Skill",
    "SkillCandidate",
    "SkillCompatibility",
    "SkillDependencies",
    "SkillDraft",
    "SkillError",
    "SkillExecutionError",
    "SkillExecutionRecord",
    "SkillExecutor",
    "SkillGenerator",
    "SkillHistory",
    "SkillLearner",
    "SkillManifest",
    "SkillRegistry",
    "SkillRisk",
    "SkillSource",
    "SkillStatus",
    "SkillStep",
    "SkillTool",
    "SandboxConfig",
    "SandboxResult",
    "SandboxedPython",
    "SkillValidation",
    "StepCondition",
    "ToolBuilder",
    "ToolProposal",
    "ToolProposalStatus",
    "ValidationIssue",
    "VerificationStatus",
    "VersionChange",
    "analyze_skill",
    "bump",
    "check_postconditions",
    "check_preconditions",
    "classify_tool",
    "compare",
    "compose_skill",
    "derive_risk",
    "diff_schemas",
    "diff_skills",
    "effective_risk",
    "emit_activity",
    "emit_audit",
    "entry_from_manifest",
    "evaluate_predicate",
    "import_skill",
    "is_valid",
    "latest",
    "loop_items",
    "manifest_from_skill",
    "manifest_hash",
    "needs_approval",
    "parse",
    "research_report_skill",
    "risk_report",
    "should_run_step",
    "sign_manifest",
    "skill_hash",
    "suggest_optimizations",
    "verify_signature",
    "verify_skill",
]

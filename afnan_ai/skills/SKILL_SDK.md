# Skill SDK — Building Skills for Afnan AI

A **Skill** is a named, versioned, reusable workflow composed of *existing*
registered tools (and other skills). A **Tool** is an atomic capability.
A **Task** is a user-specific execution instance. A **Goal** is a
long-term objective. These responsibilities are never merged.

## The lifecycle

```
Goal / repeated workflow
→ Discover capabilities (registry.search)
→ Compose tools (composer.compose_skill)
→ Generate Skill Definition (SkillGenerator)
→ Validate schema + security (SkillValidator, security.analyze_skill)
→ Sandbox test (SkillSandbox / SandboxedPython)
→ Verify behavior (contracts.check_postconditions)
→ Version (versions.bump, VersionChange)
→ Publish (registry.register — drafts refused)
→ Execute (SkillExecutor via ToolRegistry)
→ Monitor (SkillHistory, observability)
→ Rollback (registry.rollback)
```

A generated skill is **never** auto-activated: validation criteria must
complete first, and sensitive+ risk always needs human approval.

## Manifest format

Every skill carries a `SkillManifest`:

| Field | Meaning |
|---|---|
| skill_id / name / description | Identity |
| version | Semantic `MAJOR.MINOR.PATCH` |
| source | `system` / `developer` / `user_created` / `agent_generated` / `imported` |
| author | Who created it |
| input_schema / output_schema | Strict object schemas (invalid input → validation failure) |
| required_tools / required_connectors | Fail fast when unavailable |
| required_permissions | Declared capabilities; a skill can never elevate its own |
| skill_dependencies | Other skills (cycles refused at registration) |
| risk_level | `read_only` / `reversible` / `sensitive` / `destructive` |
| limits | `max_steps`, `max_subskill_depth`, `timeout_s`, `max_loop_iterations` |
| compatibility | Afnan version, OS, browser/computer needs, tool versions |
| verification_status | `unverified` / `sandbox_tested` / `verified` |
| integrity_hash | Tamper detection (mismatch → auto-disable) |

Source trust: `system` 1.0, `developer` 0.9, `user_created` 0.7,
`agent_generated` 0.4 (strict validation), `imported` 0.0 until verified.

## Writing a skill (declarative DSL)

```python
from afnan_ai.skills import compose_skill

skill = compose_skill(
    skill_id="research_company",
    name="Research Company",
    description="Research a company from public sources.",
    risk="read_only",
    input_schema={
        "type": "object",
        "properties": {
            "company_name": {"type": "string"},
        },
        "required": ["company_name"],
    },
    steps=[
        {"step_id": "search", "tool": "web_search",
         "arguments": {"query": "{{input.company_name}}"}},
        {"step_id": "open", "tool": "page_open",
         "arguments": {"url": "{{steps.search.top_url}}"},
         # conditional: only when results exist
         "condition": {"kind": "when",
                       "predicate": {"path": "steps.search.found",
                                     "equals": True}}},
    ],
)
```

Steps support `skill:<other_id>` for sub-skills (max depth enforced),
`{{input.*}}` / `{{steps.*}}` templates, structured `when`/`skip_when`
conditions over validated state, and bounded `loop` steps
(`over` a list path, `max_iterations` always set).

No arbitrary host code: the model never gets unrestricted code
execution. Code paths go through static validation → dependency
validation → security scan → sandbox → tests → behavior
verification → policy approval.

## Versioning

- `versions.bump(skill.version, "major|minor|patch")`
- `diff_skills(old, new)` classifies behavior / dependency /
  permission / schema / security changes; breaking input changes
  (removed properties, new required fields) force a major bump.
- The active version of a running task is never silently replaced.
- `registry.rollback(skill_id, version)` re-activates a previous
  stable version after compatibility + dependency + security checks.

## Security model

1. **Pre-publish scan** (`security.analyze_skill`): excessive
   permissions, undeclared tools, credential access, suspicious
   transfers, undeclared network, lax limits. Blockers stop
   publication. Agent-generated skills use the strict profile.
2. **Permissions**: declared in the manifest, authorized through
   the existing PermissionManager; skills cannot self-elevate.
3. **Approvals**: sensitive steps → PermissionManager → approval
   request → user decision → policy re-evaluation → execute.
   Approval never grants permanent unrestricted permission.
4. **Prompt injection**: external content can *inspire* a skill
   definition but never *authorizes* it — permissions derive from
   developer/user policy only.
5. **Credentials**: skills get CredentialVault-mediated, scoped
   access — never raw secrets. Secrets are refused in manifests,
   sources, logs, trajectory, ActivityCenter and test output.
6. **Integrity**: `integrity.skill_hash` over the canonical
   manifest; mismatch → automatic disable.
7. **Imports**: signature check → manifest validation →
   dependency analysis → security scan → sandbox → policy →
   approval (sensitive+) → install disabled-by-default.

## Execution contracts

- **Preconditions** (`contracts.check_preconditions`): tools
  available, connectors connected, permissions granted, workspace
  ready, inputs valid, dependencies available, budget sufficient.
  Failure → execution never starts.
- **Postconditions** (`contracts.check_postconditions`): output
  schema, verification criteria evidenced, artifacts produced.
  Failure → `UNCERTAIN` / `FAILED`, never silent success.
- **Recovery**: observe → identify failed capability → retry only
  if safe/idempotent → alternate strategy → replan → verify.
  Failed actions are never blindly repeated.
- **Long-running**: integrates with TaskManager + checkpointing;
  completed steps are not repeated after resume.

## Observability

ActivityCenter: discovered / selected / validated / started,
step started/completed/failed, approval required, recovery,
verification, completed — redacted summaries only.
AuditCenter: permission checks, capability use, security
findings, credential access, publication, version changes,
rollbacks. `history.SkillHistory` keeps per-execution records
(inputs redacted) and `stats()` for optimization.

## Learning & optimization

- `SkillLearner`: verified successful workflows become
  *candidates* — promotion stays explicit and audited.
- `optimize.suggest_optimizations(stats)`: low success rate,
  slow runs, excessive retries, recurring failures → suggestions.
  Applying one creates a **new version**; stable is never
  overwritten.

## Tool vs Skill vs Task vs Goal

| | Atomic? | Reusable? | Versioned? |
|---|---|---|---|
| Tool | yes | yes | no |
| Skill | no (multi-step) | yes | yes |
| Task | no | no (one instance) | no |
| Goal | no | no | no |

`toolsmith.ToolBuilder` proposes *new tools* only as compositions
of existing primitives — never privileged, never wrapping
`shell_exec`-class tools, through sandbox → security → registration.

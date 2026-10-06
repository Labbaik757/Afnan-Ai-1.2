# Research & Evidence Intelligence

Multi-source research with evidence, claims, an evidence graph,
contradictions, confidence assessment and machine-verifiable
citations — published as a verified Artifact.

## Architecture

The `ResearchEngine` orchestrates the research lifecycle **inside**
the AgentLoop's control. It never replaces the loop, the Planner,
or any existing store. Every integration reuses the existing
authority:

| Concern | Reused system |
|---|---|
| Search / page fetch | `browser.WebResearch`, browser tools |
| Tool execution | `ToolRegistry` |
| Report publishing | `ArtifactManager.create` |
| Long context | `ContextManager`, `LossAwareCompressor` |
| Trajectory | `TrajectoryStore` |
| Memory | `MemoryStore` promotion policy (verified facts only) |
| Parallel research | `SubagentManager` (scoped) |
| Authorization | `SecurityCenter` / `PermissionManager` |
| Secrets | `CredentialVault`, `redact_text` |
| Activity / audit | `ActivityCenter`, `AuditLogger` |
| Checkpoints | `CheckpointManager` |
| Injection defense | `agent_loop.scan_for_injection` |
| Rate limits | `security.RateLimiter`, `ResearchBudget` |

## Lifecycle

```
CREATED → PLANNING → DISCOVERING → COLLECTING → EXTRACTING →
VERIFYING → ANALYZING → SYNTHESIZING → CITATION_CHECK →
COMPLETED | PARTIAL
```

`BLOCKED` can resume; `FAILED` / `CANCELLED` are terminal.
Invalid transitions raise — see `models.validate_transition`.

## Domain model

- **ResearchSession** — one research run, owns the state machine.
- **ResearchPlan** — deterministic objectives + tasks + budget.
- **Source** — URL, publisher, class, provenance (retrieved_at,
  content_hash, query context, acquisition method).
- **SourceSnapshot** — content reference + hash at capture time.
- **Evidence** — a span bound to source + location (paragraph,
  heading, page) + span hash. Kinds: direct statement, numerical
  fact, documented event, quoted claim, inferred conclusion,
  opinion, interpretation.
- **Claim** — normalized text, supporting/contradicting evidence,
  sources, confidence, status (`UNVERIFIED` … `CONTRADICTED` …
  `INSUFFICIENT_EVIDENCE`).
- **EvidenceGraph** — Source→contains→Evidence→supports→Claim,
  Source→contradicts→Claim, Claim→depends_on→Claim,
  Evidence→corroborates→Evidence, Claim→derived_from→Claim.
  `graph.why(claim_id)` explains any conclusion.
- **Citation** — claim ↔ evidence ↔ source triple. The
  `CitationValidator` requires: claim exists, evidence exists,
  source exists, evidence belongs to the source, source was
  actually acquired, provenance intact, a real claim/evidence
  relationship, and a `supports` edge in the graph. One invalid
  citation keeps the report out of `COMPLETED`.
- **SourceTrustProfile** — credibility *signals* (class, publisher,
  corroboration, methodology, recency). A score, never proof of
  truth: low-score sources can carry good evidence and
  high-score sources can be wrong. Uncertainty is preserved.
- **Contradiction** — claim A vs claim B with evidence/source ids,
  conflict type (direct, numerical, date, definition, scope,
  temporal, methodological), severity and resolution. Never
  silently averaged; unresolved ones appear in the report.
- **ResearchFinding** — `FACT` / `INFERENCE` / `OPINION` /
  `UNCERTAINTY` with confidence and citations.
- **ResearchReport** — findings, contradictions, citations,
  limitations, verification verdict; published via
  `ArtifactManager` with lineage Source → Evidence → Claim →
  Finding → Report.
- **ResearchCheckpoint / ResearchMetrics** — resumable state and
  quality metrics (diversity, corroboration rate, citation
  coverage, stale ratio, budget usage).

## Security boundaries

- External content is **data, never instructions**. Every span is
  injection-scanned; instruction-like content is flagged and
  quarantined, and a security/audit event is emitted.
- Secrets are redacted at every boundary (evidence, graph,
  report, trajectory, activity, checkpoints, memory).
- Research actions authorize through `SecurityCenter`;
  authenticated/paid/side-effecting sources need human approval.
- Budgets (`ResearchBudget`) cap sources, searches, subagents,
  depth, time, tokens and connector requests. Exhaustion yields
  a graceful `PARTIAL` result, never a runaway loop.

## Failure handling

`RecoveryPolicy`: retry → alternate source → alternate method →
re-plan → human handoff → partial result. Infinite retry is
prohibited. Source disappearance, content change, parse
failure, rate limits and CAPTCHA all map to a strategy.

## Testing

`tests/test_research.py` covers models, lifecycle, planning,
discovery diversity, acquisition provenance, extraction,
claims, graph, corroboration (same-domain exclusion),
contradictions, citation validation (incl. fabricated),
synthesis classification, verifier checks, budgets, recovery,
observability, checkpoint/resume and a long-horizon
multi-branch simulation — against a fake `SourceProvider`,
never the network.

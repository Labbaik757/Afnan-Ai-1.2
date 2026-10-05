"""SubAgentManager — lifecycle, parallel execution, audit.

The manager owns subagent lifecycle (create / start / pause /
resume / cancel / terminate) and the parent's execution
graph.  It never plans or executes itself:

* each subagent runs the *existing* AgentLoop against a
  *scoped* tool view, with its own isolated AgentState,
* independent subagents run in parallel threads; shared
  resources are guarded by the ResourceLockManager,
* results return as structured handoffs that the parent
  verifies instead of trusting,
* approval pauses surface as WAITING_FOR_APPROVAL; the
  subagent can never approve independently — the decision
  flows Subagent → Parent → Human → resume,
* every status change, handoff and verdict lands in the
  audit trail and the execution graph.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from afnan_ai.log_config import get_logger
from afnan_ai.redaction import redact_text, redact_value
from afnan_ai.subagents.communication import SubAgentMailbox
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
from afnan_ai.subagents.models import (
    ResourceLimits,
    SubAgentHandoff,
    SubAgentRecord,
    SubAgentSpec,
    SubAgentStatus,
    VerificationState,
)
from afnan_ai.subagents.scoped_registry import ScopedToolRegistry
from afnan_ai.subagents.security import (
    SubAgentSecurity,
    SecurityReport,
)

logger = get_logger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class SubAgentError(Exception):
    """Structured subagent-management failure."""

    def __init__(
        self, message: str, *, code: str = "subagent_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "message": str(self),
            "details": dict(self.details),
        }


class _CapturingCheckpointer:
    """Wraps a checkpointer to remember the last snapshot
    (needed for approval-pause resume)."""

    def __init__(self, inner: Any | None) -> None:
        self._inner = inner
        self.last_snapshot: dict[str, Any] | None = None

    def save_snapshot(self, **kwargs: Any) -> Any:
        snapshot = None
        if self._inner is not None:
            snapshot = self._inner.save_snapshot(**kwargs)
        self.last_snapshot = snapshot
        return snapshot

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class SubAgentManager:
    """Create, run and supervise subagents for a parent agent."""

    def __init__(
        self,
        *,
        tool_registry: Any,
        loop_factory: Callable[..., Any],
        security: SubAgentSecurity,
        verifier: HandoffVerifier | None = None,
        mailbox: SubAgentMailbox | None = None,
        locks: ResourceLockManager | None = None,
        audit_path: str | Path | None = None,
        max_parallel: int = 4,
        security_center: Any = None,
    ) -> None:
        self.tools = tool_registry
        self.loop_factory = loop_factory
        self.security = security
        self.verifier = verifier or HandoffVerifier()
        self.mailbox = mailbox or SubAgentMailbox()
        self.locks = locks or ResourceLockManager()
        self.max_parallel = max(1, max_parallel)
        self._audit_path = (
            Path(str(audit_path)).expanduser()
            if audit_path else None
        )
        self._records: dict[str, SubAgentRecord] = {}
        self._runtimes: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.decomposer = TaskDecomposer()
        # Global emergency stop: a trip terminates every
        # live subagent.  Only an explicit authorized reset
        # lets new ones start.
        self._security_center = None
        self._security_cb_token = None
        self.set_security_center(security_center)

    def set_security_center(self, center: Any) -> None:
        """Attach (or replace) the central SecurityCenter.

        Used when the manager is constructed before the
        center exists.  Emergency-callback registration is
        idempotent: only one callback per manager."""
        if center is None:
            return
        self._security_center = center
        if self._security_cb_token is None:
            try:
                self._security_cb_token = (
                    center.emergency.register(
                        self._emergency_terminate
                    )
                )
            except Exception:
                self._security_cb_token = None

    def _emergency_terminate(self) -> None:
        with self._lock:
            ids = list(self._records.keys())
        for sid in ids:
            try:
                self.terminate(sid)
            except Exception:
                pass

    # -- lifecycle ------------------------------------------------------
    def create(self, spec: SubAgentSpec) -> SubAgentRecord:
        """Validate permissions and register a subagent."""
        if not isinstance(spec, SubAgentSpec):
            raise SubAgentError(
                "spec must be a SubAgentSpec",
                code="invalid_spec",
            )
        with self._lock:
            if spec.subagent_id in self._records:
                raise SubAgentError(
                    f"subagent {spec.subagent_id!r} already "
                    "exists",
                    code="duplicate_subagent",
                )
        report = self.security.validate_spec(spec)
        if not report.ok:
            raise SubAgentError(
                "subagent spec failed security validation: "
                + "; ".join(
                    i.message for i in report.issues[:3]
                ),
                code="security_rejected",
                details=report.to_dict(),
            )
        record = SubAgentRecord(spec=spec)
        with self._lock:
            if (
                self._security_center is not None
                and self._security_center.emergency
                .is_tripped()
            ):
                raise SubAgentError(
                    "refused: the global emergency stop is "
                    "tripped; an authorized reset is required "
                    "before new subagents may start",
                    code="emergency_stop",
                )
            self._records[spec.subagent_id] = record
        self._audit("created", record)
        self._grant_center_capabilities(spec)
        return record

    def _grant_center_capabilities(self, spec) -> None:
        """Least-privilege grant in the central
        PermissionManager (when the parent registry has a
        SecurityCenter): the subagent actor gets exactly
        the capabilities its allowed tools imply — never
        more than the agent holds."""
        center = getattr(
            self.tools, "security_center", None
        )
        if center is None:
            return
        try:
            from afnan_ai.security.policy import (
                SecurityPolicyEngine,
            )

            caps: set[str] = set()
            for pattern in spec.allowed_tools or []:
                pattern = str(pattern or "").strip()
                if not pattern:
                    continue
                if pattern.endswith("*"):
                    domain = pattern[:-1].rstrip("_")
                    if domain:
                        caps.add(f"{domain}.*")
                else:
                    caps.add(
                        SecurityPolicyEngine._capability_for(
                            pattern
                        )
                    )
            if caps:
                actor = center.subagent_actor(
                    spec.subagent_id, sorted(caps)
                )
                # subagent_actor() already intersects with
                # the agent's own capabilities.
                _ = actor
        except Exception:
            pass  # grants are best-effort; the scoped
            # registry still enforces its own allow-list

    def get(self, subagent_id: str) -> SubAgentRecord:
        with self._lock:
            record = self._records.get(subagent_id)
        if record is None:
            raise SubAgentError(
                f"unknown subagent {subagent_id!r}",
                code="unknown_subagent",
            )
        return record

    def list_subagents(self) -> list[SubAgentRecord]:
        with self._lock:
            return list(self._records.values())

    # -- execution ------------------------------------------------------
    def execute(
        self, subagent_id: str, *, timeout: float | None = None
    ) -> SubAgentHandoff:
        """Run one subagent synchronously (with retries)."""
        record = self.get(subagent_id)
        spec = record.spec
        last_error: str | None = None
        for attempt in range(spec.limits.max_retries + 1):
            record.attempts = attempt + 1
            try:
                handoff = self._run_once(
                    record, timeout=timeout
                )
            except _ApprovalPause as pause:
                # Not a failure: surface to the parent for the
                # Subagent → Parent → Human flow.
                record.status = (
                    SubAgentStatus.WAITING_FOR_APPROVAL
                )
                record.failure_reason = (
                    f"approval required: {pause.request[:160]}"
                )
                self._audit("waiting_for_approval", record)
                raise
            except SubAgentError:
                raise
            except Exception as e:  # noqa: BLE001
                last_error = str(e)[:300]
                record.failure_reason = last_error
                logger.warning(
                    "subagent %s attempt %d failed: %s",
                    subagent_id, attempt + 1, last_error,
                )
                if attempt >= spec.limits.max_retries:
                    break
                continue
            # A failed handoff (not a crash) is also
            # retried within the budget — the loop's own
            # recovery already avoided blind repeats
            # *within* the attempt, so a fresh attempt is a
            # genuinely different try, not an infinite loop.
            if (
                handoff.status == "failed"
                and attempt < spec.limits.max_retries
            ):
                last_error = handoff.error
                record.failure_reason = last_error
                logger.warning(
                    "subagent %s attempt %d returned failed: "
                    "%s",
                    subagent_id, attempt + 1, last_error,
                )
                continue
            return handoff
        record.status = SubAgentStatus.FAILED
        handoff = SubAgentHandoff(
            subagent_id=subagent_id,
            objective=spec.objective,
            status="failed",
            error=last_error or "unknown failure",
            started_at=record.created_at,
            finished_at=_utcnow(),
        )
        record.handoff = handoff
        self._audit("failed", record)
        return handoff

    def _run_once(
        self, record: SubAgentRecord,
        timeout: float | None = None,
    ) -> SubAgentHandoff:
        from afnan_ai.agent_loop import LoopControl, LoopLimits

        spec = record.spec
        subagent_id = spec.subagent_id
        # 1) Shared resources first — fail fast on conflict.
        try:
            held = self.locks.acquire(
                subagent_id, spec.resources, timeout=30.0
            )
        except ResourceConflict as e:
            record.status = SubAgentStatus.FAILED
            record.failure_reason = str(e)[:200]
            handoff = SubAgentHandoff(
                subagent_id=subagent_id,
                objective=spec.objective, status="failed",
                error=f"resource_conflict: {e}",
                started_at=record.created_at,
                finished_at=_utcnow(),
            )
            record.handoff = handoff
            self._audit("failed", record)
            return handoff
        record.held_resources = held
        # 2) Least-privilege tool view.
        scoped = ScopedToolRegistry(
            self.tools,
            allowed=spec.allowed_tools,
            risk_permissions=spec.risk_permissions,
            allowed_connectors=spec.allowed_connectors,
            max_tool_calls=spec.limits.max_tool_calls,
            owner_id=subagent_id,
        )
        # 3) Scoped context — never the parent's AgentState.
        context_text = self.security.scope_context(spec)
        checkpointer = _CapturingCheckpointer(None)
        control = LoopControl()
        runtime: dict[str, Any] = {
            "control": control,
            "checkpointer": checkpointer,
            "scoped": scoped,
        }
        with self._lock:
            self._runtimes[subagent_id] = runtime
        record.status = SubAgentStatus.RUNNING
        self._audit("started", record)
        started_at = _utcnow()
        loop = self.loop_factory(
            spec, scoped, context_text,
            checkpointer=checkpointer,
        )
        limits = LoopLimits(
            max_steps=spec.limits.max_steps,
            max_duration_s=spec.limits.timeout_s,
        )
        outcome_box: dict[str, Any] = {}

        def _target() -> None:
            try:
                outcome_box["result"] = loop.run(
                    spec.objective,
                    limits=limits,
                    control=control,
                    goal_id=f"sub:{subagent_id}",
                )
            except Exception as e:  # noqa: BLE001
                outcome_box["error"] = e

        thread = threading.Thread(
            target=_target,
            name=f"subagent-{subagent_id}",
            daemon=True,
        )
        thread.start()
        budget = (
            timeout if timeout is not None
            else (spec.limits.timeout_s or 120.0) + 30.0
        )
        thread.join(timeout=budget)
        try:
            if thread.is_alive():
                control.request_stop()
                thread.join(timeout=10.0)
                raise SubAgentError(
                    f"subagent {subagent_id!r} timed out "
                    f"after {budget:.0f}s",
                    code="timeout",
                )
            if "error" in outcome_box:
                raise outcome_box["error"]
            result = outcome_box.get("result")
            if result is None:
                raise SubAgentError(
                    "subagent produced no result",
                    code="no_result",
                )
            return self._finish_run(
                record, result, scoped, started_at,
                checkpointer,
            )
        finally:
            self.locks.release(subagent_id)
            record.held_resources = []
            with self._lock:
                self._runtimes.pop(subagent_id, None)

    def _finish_run(
        self, record: SubAgentRecord, result: Any,
        scoped: ScopedToolRegistry, started_at: str,
        checkpointer: _CapturingCheckpointer,
    ) -> SubAgentHandoff:
        from afnan_ai.orchestrator import OrchestrationStatus

        spec = record.spec
        subagent_id = spec.subagent_id
        error = getattr(result, "error", None)
        code = (error or {}).get("code", "")
        if code in ("approval_required", "approval_denied"):
            # Pause, don't fail: Subagent → Parent → Human.
            with self._lock:
                runtime = self._runtimes.get(subagent_id, {})
                runtime["checkpoint"] = (
                    checkpointer.last_snapshot
                )
            raise _ApprovalPause(
                subagent_id,
                (error or {}).get("message", "approval needed"),
            )
        state = getattr(result, "state", None)
        verified = 0
        if state is not None:
            verified = len(
                getattr(state, "completed_steps", []) or []
            )
        output = None
        evidence: list[str] = []
        if state is not None:
            summaries = []
            for step in (
                getattr(state, "completed_steps", []) or []
            ):
                name = getattr(step, "name", "?")
                step_result = getattr(step, "result", "")
                # Summaries carry evidence, not just names —
                # the handoff verifier checks output against
                # them.  AgentState already redacts secrets.
                summaries.append(
                    f"{name}: {str(step_result)[:160]}"
                )
            output = (
                "; ".join(summaries[:10])
                or "task completed"
            )
            evidence = summaries[:10]
        confidence = (
            min(0.95, 0.5 + 0.1 * verified)
            if result.status == OrchestrationStatus.COMPLETED
            else 0.0
        )
        handoff = SubAgentHandoff(
            subagent_id=subagent_id,
            objective=spec.objective,
            status=(
                "completed"
                if result.status == OrchestrationStatus.COMPLETED
                else "failed"
            ),
            output=output,
            evidence=evidence,
            confidence=confidence,
            tool_calls=scoped.tool_calls,
            steps_verified=verified,
            error=(
                (error or {}).get("message")
                if handoff_status(result) != "completed"
                else None
            ),
            started_at=started_at,
            finished_at=_utcnow(),
        )
        record.handoff = handoff
        # An explicit parent cancel/terminate sticks: a loop
        # finishing right after the decision does not
        # overwrite it.
        if record.status not in (
            SubAgentStatus.CANCELLED,
            SubAgentStatus.TERMINATED,
        ):
            record.status = (
                SubAgentStatus.COMPLETED
                if handoff.status == "completed"
                else SubAgentStatus.FAILED
            )
        # Verify the handoff — never trusted blindly.
        verdict = self.verifier.verify(
            handoff, expected=spec.objective
        )
        handoff.verification = verdict.state.value
        self._audit(
            "completed" if handoff.status == "completed"
            else "failed",
            record,
            extra={"verdict": verdict.to_dict()},
        )
        return handoff

    # -- approval -------------------------------------------------------
    def pending_approvals(self) -> list[dict[str, Any]]:
        """Subagents currently waiting on a human."""
        with self._lock:
            return [
                {
                    "subagent_id": r.spec.subagent_id,
                    "role": r.spec.role,
                    "reason": r.failure_reason,
                }
                for r in self._records.values()
                if r.status
                == SubAgentStatus.WAITING_FOR_APPROVAL
            ]

    def resolve_approval(
        self, subagent_id: str, approved: bool
    ) -> SubAgentHandoff:
        """Parent records the human decision and resumes the
        subagent from its checkpoint.  The subagent never
        approves independently."""
        record = self.get(subagent_id)
        if (
            record.status
            != SubAgentStatus.WAITING_FOR_APPROVAL
        ):
            raise SubAgentError(
                f"subagent {subagent_id!r} is not waiting for "
                "approval",
                code="not_waiting",
            )
        # The human decision flows through the parent's
        # existing approver; here we just resume — the loop
        # re-runs from the checkpoint and the parent's gate
        # sees the human's answer.
        self._audit(
            "approval_resolved", record,
            extra={"approved": approved},
        )
        if not approved:
            record.status = SubAgentStatus.CANCELLED
            handoff = SubAgentHandoff(
                subagent_id=subagent_id,
                objective=record.spec.objective,
                status="cancelled",
                error="human denied approval",
                started_at=record.created_at,
                finished_at=_utcnow(),
            )
            record.handoff = handoff
            return handoff
        with self._lock:
            runtime = self._runtimes.get(subagent_id, {})
            checkpoint = runtime.get("checkpoint")
        if not checkpoint:
            raise SubAgentError(
                "no checkpoint to resume from",
                code="no_checkpoint",
            )
        # Resume with a fresh run from the checkpoint.
        record.status = SubAgentStatus.RUNNING
        return self._resume_from_checkpoint(record, checkpoint)

    def _resume_from_checkpoint(
        self, record: SubAgentRecord, checkpoint: dict
    ) -> SubAgentHandoff:
        from afnan_ai.agent_loop import LoopControl, LoopLimits

        spec = record.spec
        subagent_id = spec.subagent_id
        scoped = ScopedToolRegistry(
            self.tools,
            allowed=spec.allowed_tools,
            risk_permissions=spec.risk_permissions,
            allowed_connectors=spec.allowed_connectors,
            max_tool_calls=spec.limits.max_tool_calls,
            owner_id=subagent_id,
        )
        context_text = self.security.scope_context(spec)
        checkpointer = _CapturingCheckpointer(None)
        control = LoopControl()
        loop = self.loop_factory(
            spec, scoped, context_text,
            checkpointer=checkpointer,
        )
        limits = LoopLimits(
            max_steps=spec.limits.max_steps,
            max_duration_s=spec.limits.timeout_s,
        )
        started_at = _utcnow()
        try:
            result = loop.run(
                spec.objective,
                limits=limits,
                control=control,
                resume_from=checkpoint,
                goal_id=f"sub:{subagent_id}",
            )
        finally:
            with self._lock:
                self._runtimes.pop(subagent_id, None)
        return self._finish_run(
            record, result, scoped, started_at, checkpointer
        )

    # -- pause / cancel / terminate ---------------------------------------
    def pause(self, subagent_id: str) -> None:
        with self._lock:
            runtime = self._runtimes.get(subagent_id)
        if runtime is None:
            raise SubAgentError(
                f"subagent {subagent_id!r} is not running",
                code="not_running",
            )
        runtime["control"].request_pause()
        self.get(subagent_id).status = SubAgentStatus.PAUSED
        self._audit("paused", self.get(subagent_id))

    def cancel(self, subagent_id: str) -> None:
        with self._lock:
            runtime = self._runtimes.get(subagent_id)
        if runtime is not None:
            runtime["control"].request_stop()
        record = self.get(subagent_id)
        record.status = SubAgentStatus.CANCELLED
        self.locks.release(subagent_id)
        self._audit("cancelled", record)

    def terminate(self, subagent_id: str) -> None:
        """Force-stop and forget a subagent's runtime."""
        self.cancel(subagent_id)
        record = self.get(subagent_id)
        record.status = SubAgentStatus.TERMINATED
        with self._lock:
            self._runtimes.pop(subagent_id, None)
        self._audit("terminated", record)

    # -- graph execution ------------------------------------------------------
    def execute_graph(
        self, specs: list[SubAgentSpec]
    ) -> dict[str, SubAgentHandoff]:
        """Run specs honoring depends_on; independent specs in
        a level run in parallel (bounded by max_parallel)."""
        for spec in specs:
            if spec.subagent_id not in self._records:
                self.create(spec)
        levels = self._topological_levels(specs)
        handoffs: dict[str, SubAgentHandoff] = {}
        for level in levels:
            # A failed dependency fails its dependents fast
            # with a structured reason (no wasted work).
            for spec in level:
                failed_deps = [
                    dep for dep in spec.depends_on
                    if dep in handoffs
                    and handoffs[dep].status != "completed"
                ]
                if failed_deps:
                    record = self.get(spec.subagent_id)
                    record.status = SubAgentStatus.FAILED
                    record.failure_reason = (
                        "dependency failed: "
                        + ", ".join(failed_deps)
                    )
                    handoffs[spec.subagent_id] = SubAgentHandoff(
                        subagent_id=spec.subagent_id,
                        objective=spec.objective,
                        status="failed",
                        error=record.failure_reason,
                        started_at=record.created_at,
                        finished_at=_utcnow(),
                    )
            runnable = [
                spec for spec in level
                if spec.subagent_id not in handoffs
            ]
            results = self._run_parallel(runnable)
            handoffs.update(results)
        return handoffs

    def _topological_levels(
        self, specs: list[SubAgentSpec]
    ) -> list[list[SubAgentSpec]]:
        by_id = {s.subagent_id: s for s in specs}
        levels: list[list[SubAgentSpec]] = []
        done: set[str] = set()
        remaining = dict(by_id)
        while remaining:
            level = [
                spec for spec in remaining.values()
                if all(
                    dep in done or dep not in by_id
                    for dep in spec.depends_on
                )
            ]
            if not level:
                raise SubAgentError(
                    "circular depends_on in subagent specs",
                    code="circular_dependency",
                )
            levels.append(level)
            for spec in level:
                done.add(spec.subagent_id)
                del remaining[spec.subagent_id]
        return levels

    def _run_parallel(
        self, specs: list[SubAgentSpec]
    ) -> dict[str, SubAgentHandoff]:
        handoffs: dict[str, SubAgentHandoff] = {}
        if not specs:
            return handoffs
        semaphore = threading.Semaphore(self.max_parallel)
        errors: dict[str, Exception] = {}

        def _one(spec: SubAgentSpec) -> None:
            with semaphore:
                try:
                    handoffs[spec.subagent_id] = self.execute(
                        spec.subagent_id
                    )
                except _ApprovalPause as pause:
                    # Approval pauses propagate as handoffs
                    # marked accordingly (parent resolves).
                    record = self.get(spec.subagent_id)
                    handoffs[spec.subagent_id] = (
                        record.handoff
                        or SubAgentHandoff(
                            subagent_id=spec.subagent_id,
                            objective=spec.objective,
                            status="failed",
                            error=(
                                "approval pause: "
                                f"{pause.request[:120]}"
                            ),
                        )
                    )
                except Exception as e:  # noqa: BLE001
                    errors[spec.subagent_id] = e

        threads = [
            threading.Thread(
                target=_one, args=(spec,),
                name=f"mgr-{spec.subagent_id}", daemon=True,
            )
            for spec in specs
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            first = next(iter(errors.values()))
            raise first
        return handoffs

    # -- decomposition ----------------------------------------------------------
    def decompose(
        self, goal_text: str, **kwargs: Any
    ) -> DecompositionProposal:
        return self.decomposer.decompose(goal_text, **kwargs)

    def should_decompose(self, goal_text: str) -> bool:
        return self.decomposer.should_decompose(goal_text)

    # -- execution graph & audit --------------------------------------------------
    def get_execution_graph(self) -> dict[str, Any]:
        """Parent's complete execution graph for audit."""
        with self._lock:
            records = list(self._records.values())
        return {
            "generated_at": _utcnow(),
            "subagents": [r.to_dict() for r in records],
            "locks": self.locks.status(),
            "mailbox": self.mailbox.audit_log(),
        }

    def merge_results(
        self, handoffs: dict[str, SubAgentHandoff]
    ) -> dict[str, Any]:
        """Merge verified handoffs into one parent result."""
        merged: dict[str, Any] = {}
        unverified: list[str] = []
        for subagent_id, handoff in handoffs.items():
            if (
                handoff.status == "completed"
                and handoff.verification
                == VerificationState.VERIFIED.value
            ):
                merged[subagent_id] = handoff.output
            else:
                unverified.append(subagent_id)
        return {
            "merged": merged,
            "unverified": unverified,
            "total": len(handoffs),
        }

    def _audit(
        self, event: str, record: SubAgentRecord,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self._audit_path is None:
            return
        try:
            from afnan_ai.persistence import JsonFileStore

            store = JsonFileStore(str(self._audit_path))
            data = store.read({"events": []})
            events = data.get("events", [])
            entry: dict[str, Any] = {
                "event": event,
                "subagent_id": record.spec.subagent_id,
                "role": record.spec.role,
                "status": record.status.value,
                "attempts": record.attempts,
                "at": _utcnow(),
            }
            if record.failure_reason:
                entry["failure_reason"] = redact_text(
                    record.failure_reason
                )[:200]
            if extra:
                entry["extra"] = redact_value(extra)
            events.append(entry)
            store.write({"events": events[-500:]})
        except Exception as e:  # noqa: BLE001 - never breaks
            logger.warning("subagent audit failed: %s", e)


class _ApprovalPause(Exception):
    """Internal: a subagent paused for human approval."""

    def __init__(
        self, subagent_id: str, request: str
    ) -> None:
        super().__init__(
            f"subagent {subagent_id!r} waiting for approval"
        )
        self.subagent_id = subagent_id
        self.request = request


def handoff_status(result: Any) -> str:
    from afnan_ai.orchestrator import OrchestrationStatus

    return (
        "completed"
        if getattr(result, "status", None)
        == OrchestrationStatus.COMPLETED
        else "failed"
    )

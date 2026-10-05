"""ConnectorService — the runtime behind the connector tools.

The service is the only core component that talks to
connectors, and it knows nothing connector-specific.  It owns
the whole execution pipeline:

1. **Validate** — connector exists, operation is supported,
   parameters match the operation's schema.
2. **Authorize** — the operation's required scopes are a
   subset of the scopes granted to this connector
   (least privilege).
3. **Connect** — credentials come from the
   :class:`~afnan_ai.connectors.credentials.CredentialStore`
   (never from tool arguments or planner output); expired
   sessions are refreshed once, transparently.
4. **Approve** — sensitive/irreversible operations go through
   the fail-safe :class:`ConnectorApprovalGate`.
5. **Execute** — with a timeout; failures become structured
   :class:`ConnectorError` records (timeout, rate limit,
   expired auth, permission denied, network failure,
   malformed response, outage).
6. **Record** — safe metadata in the result (id, operation,
   status, redacted summary, timestamps), a redacted audit
   entry, and a fresh observation for the Verifier.

The AgentLoop drives multi-step connector workflows through
the normal plan → execute → verify cycle: the Planner picks
``connector_execute`` from the structured capability schema,
the Executor runs it through this service, and the Verifier
judges the safe result.  No connector-specific orchestration
lives in the agent.
"""

from __future__ import annotations

import concurrent.futures
import time
from datetime import datetime, timezone
from typing import Any, Callable

from afnan_ai.connectors.audit import ConnectorAuditLog
from afnan_ai.connectors.auth import AuthSession
from afnan_ai.connectors.base import (
    Connector,
    HealthResult,
    OperationContext,
)
from afnan_ai.connectors.credentials import (
    CredentialStore,
    MemoryCredentialStore,
)
from afnan_ai.connectors.errors import (
    ConnectorError,
    ConnectorErrorCode,
    ConnectorException,
    map_exception,
)
from afnan_ai.connectors.models import (
    AuditRecord,
    ConnectionRecord,
    ConnectionStatus,
    OperationResult,
    OperationSpec,
)
from afnan_ai.connectors.policy import (
    Approver,
    ConnectorApprovalGate,
    OperationRisk,
    RiskPolicy,
    classify_operation,
)
from afnan_ai.connectors.registry import ConnectorRegistry
from afnan_ai.log_config import get_logger
from afnan_ai.redaction import redact_value

logger = get_logger(__name__)

_SUMMARY_CAP = 2000

_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


class ConnectorService:
    """Runtime for connector lifecycle, auth, approval and execution."""

    def __init__(
        self,
        registry: ConnectorRegistry | None = None,
        *,
        credential_store: CredentialStore | None = None,
        gate: ConnectorApprovalGate | None = None,
        audit_log: ConnectorAuditLog | None = None,
        audit_path: str | None = None,
        default_timeout_s: float = 30.0,
    ):
        self.registry = registry or ConnectorRegistry()
        self.credentials: CredentialStore = (
            credential_store or MemoryCredentialStore()
        )
        self.gate = gate or ConnectorApprovalGate()
        self.audit = audit_log or ConnectorAuditLog(audit_path)
        self.default_timeout_s = default_timeout_s
        self._connections: dict[str, ConnectionRecord] = {}
        self._sessions: dict[str, AuthSession] = {}
        self._last_results: dict[tuple[str, str], dict[str, Any]] = {}
        # Central security: the global emergency stop
        # disconnects every live connector connection and
        # refuses new executions until an authorized reset.
        self._security_center = None
        self._security_cb_token = None

    def set_security_center(self, center: Any) -> None:
        if center is None:
            return
        self._security_center = center
        if self._security_cb_token is None:
            try:
                self._security_cb_token = (
                    center.emergency.register(
                        self._emergency_disconnect
                    )
                )
            except Exception:
                self._security_cb_token = None

    def _emergency_disconnect(self) -> None:
        for connector_id in list(self._connections.keys()):
            try:
                self.disconnect(connector_id, _quiet=True)
            except Exception:
                pass

    def _check_emergency(self) -> None:
        center = self._security_center
        if (
            center is not None
            and center.emergency.is_tripped()
        ):
            raise ConnectorException(
                "refused: the global emergency stop is "
                "tripped; an authorized reset is required",
                code=ConnectorErrorCode.EMERGENCY_STOP,
            )
    # -- credentials (secure channel, never tool arguments) ------------
    def provide_credentials(
        self, connector_id: str, values: dict[str, str]
    ) -> None:
        """Store secrets for *connector_id* (secure channel).

        Called from Python (e.g. a vault-backed setup flow), never
        through a tool or the Planner — secrets must not appear in
        planner output, AgentState or logs.  Only field names are
        ever logged.
        """
        self.registry.get(connector_id)  # structured if unknown
        cleaned = {
            str(key): str(value)
            for key, value in (values or {}).items()
            if str(value)
        }
        if not cleaned:
            raise ConnectorException(
                "No credential values provided",
                code=ConnectorErrorCode.MISSING_CREDENTIALS,
                connector=connector_id,
            )
        self.credentials.put(connector_id, cleaned)
        logger.info(
            "credentials provided for %r: fields [%s]",
            connector_id,
            ", ".join(sorted(cleaned)),
        )

    def revoke_credentials(self, connector_id: str) -> bool:
        self.registry.get(connector_id)
        self.disconnect(connector_id, _quiet=True)
        return self.credentials.delete(connector_id)

    def has_credentials(self, connector_id: str) -> bool:
        return self.credentials.has(connector_id)

    # -- least-privilege scopes -------------------------------------------
    def grant_scopes(
        self, connector_id: str, scopes: list[str] | tuple[str, ...]
    ) -> list[str]:
        """Grant a subset of the connector's declared scopes."""
        connector = self.registry.get(connector_id)
        declared = set(connector.declared_scopes or ())
        requested = [str(scope) for scope in (scopes or ())]
        unknown = [s for s in requested if s not in declared]
        if unknown:
            raise ConnectorException(
                f"Connector {connector_id!r} does not declare "
                f"scope(s): {', '.join(unknown)}",
                code=ConnectorErrorCode.INVALID_ARGUMENTS,
                connector=connector_id,
                details={
                    "unknown_scopes": unknown,
                    "declared_scopes": sorted(declared),
                },
            )
        record = self._record(connector_id)
        granted = sorted(set(record.granted_scopes) | set(requested))
        record.granted_scopes = tuple(granted)
        logger.info(
            "granted scopes to %r: %s", connector_id, granted
        )
        return granted

    def revoke_scopes(
        self, connector_id: str, scopes: list[str] | tuple[str, ...]
    ) -> list[str]:
        record = self._record(connector_id)
        revoked = {str(scope) for scope in (scopes or ())}
        record.granted_scopes = tuple(
            s for s in record.granted_scopes if s not in revoked
        )
        return list(record.granted_scopes)

    # -- lifecycle -------------------------------------------------------------
    def connect(self, connector_id: str) -> dict[str, Any]:
        connector = self.registry.get(connector_id)
        record = self._record(connector_id)
        secrets = self.credentials.get(connector_id)
        if not secrets and connector.auth_type.value != "none":
            raise ConnectorException(
                f"No credentials stored for connector "
                f"{connector_id!r}; provide them via "
                "ConnectorService.provide_credentials (never as "
                "tool arguments)",
                code=ConnectorErrorCode.MISSING_CREDENTIALS,
                connector=connector_id,
            )
        context = self._context(connector_id, secrets)
        try:
            session = connector.authenticate(secrets, context)
            self._sessions[connector_id] = session
            self._store_session_secrets(connector_id, context)
            result = connector.connect(context)
        except Exception as e:  # noqa: BLE001 - mapped below
            error = map_exception(
                e, connector=connector_id, operation="connect"
            )
            record.status = ConnectionStatus.ERROR
            record.last_error = error.message
            self._audit(
                connector_id, "connect", "read", "not_required",
                "failed", error_code=error.code.value,
            )
            raise ConnectorException(
                error.message, code=error.code,
                connector=connector_id, operation="connect",
                details=error.details,
            ) from None
        record.status = ConnectionStatus.CONNECTED
        record.connected_at = datetime.now(timezone.utc).isoformat()
        record.last_error = None
        self._audit(
            connector_id, "connect", "read", "not_required", "ok"
        )
        return result.to_dict()

    def disconnect(
        self, connector_id: str, _quiet: bool = False
    ) -> dict[str, Any]:
        connector = self.registry.get(connector_id)
        try:
            connector.disconnect()
        except Exception as e:  # noqa: BLE001 - best effort
            logger.warning(
                "connector %r disconnect raised: %s", connector_id, e
            )
        self._sessions.pop(connector_id, None)
        self.credentials.delete(f"{connector_id}:session")
        record = self._record(connector_id)
        record.status = ConnectionStatus.DISCONNECTED
        record.last_error = None
        if not _quiet:
            self._audit(
                connector_id, "disconnect", "read",
                "not_required", "ok",
            )
        return {"connector_id": connector_id, "status": "disconnected"}

    def health_check(self, connector_id: str) -> dict[str, Any]:
        connector = self.registry.get(connector_id)
        record = self._record(connector_id)
        context = self._context(connector_id)
        started = time.monotonic()
        try:
            result: HealthResult = self._run_with_timeout(
                lambda: connector.health_check(context),
                self.default_timeout_s,
                connector_id,
                "health_check",
            )
        except ConnectorException as e:
            record.status = ConnectionStatus.ERROR
            record.last_error = e.error.message
            raise
        record.last_health_check = datetime.now(
            timezone.utc
        ).isoformat()
        if not result.healthy:
            record.status = ConnectionStatus.ERROR
            record.last_error = result.detail
        payload = result.to_dict()
        payload["latency_s"] = round(time.monotonic() - started, 3)
        return payload

    def connection_info(self, connector_id: str) -> dict[str, Any]:
        self.registry.get(connector_id)
        return self._record(connector_id).to_dict()

    # -- execution ---------------------------------------------------------------
    def execute(
        self,
        connector_id: str,
        operation: str,
        parameters: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Run one connector operation through the full pipeline.

        Returns the safe :class:`OperationResult` dict.  Raises
        :class:`ConnectorException` with a structured
        :class:`ConnectorError` for every expected failure mode.
        """
        started = time.monotonic()
        self._check_emergency()
        connector = self.registry.get(connector_id)
        spec = connector.get_operation(operation)
        params = self._validate_parameters(spec, parameters)
        self._check_scopes(connector_id, spec)
        self._ensure_connected(connector_id)
        risk = classify_operation(connector_id, spec)

        if self.gate.policy.is_blocked(risk.risk):
            self._audit(
                connector_id, operation, risk.risk.value,
                "denied", "failed",
                error_code=ConnectorErrorCode.PERMISSION_DENIED.value,
                duration_s=time.monotonic() - started,
            )
            raise ConnectorException(
                f"Operation {operation!r} on {connector_id!r} is "
                f"blocked by policy (risk {risk.risk.value})",
                code=ConnectorErrorCode.PERMISSION_DENIED,
                connector=connector_id,
                operation=operation,
                details={
                    "risk": risk.risk.value, "blocked_by_policy": True
                },
            )

        approval = "not_required"
        if self.gate.policy.requires_approval(risk.risk):
            approval = "required"
        decision = self.gate.check(
            risk,
            connector_id=connector_id,
            operation=operation,
            parameters=params,
        )
        if not decision.allowed:
            detail = decision.detail
            if "Denied by human" in detail:
                code = ConnectorErrorCode.APPROVAL_DENIED
                approval = "denied"
            else:
                code = ConnectorErrorCode.APPROVAL_REQUIRED
                approval = "required"
            self._audit(
                connector_id, operation, risk.risk.value,
                approval, "failed",
                error_code=code.value,
                duration_s=time.monotonic() - started,
            )
            raise ConnectorException(
                f"Operation {operation!r} on {connector_id!r} not "
                f"allowed: {detail}",
                code=code,
                connector=connector_id,
                operation=operation,
                details={"risk": risk.risk.value},
            )
        if approval == "required":
            approval = "granted"

        result = self._execute_with_refresh(
            connector, operation, params,
            timeout_s or self.default_timeout_s,
        )
        safe = self._safe_result(connector_id, operation, result)
        safe["duration_s"] = round(time.monotonic() - started, 3)
        self._last_results[(connector_id, operation)] = safe
        self._audit(
            connector_id, operation, risk.risk.value, approval,
            "ok" if safe["status"] == "ok" else "failed",
            error_code=(
                None if safe["status"] == "ok"
                else "execution_failed"
            ),
            duration_s=safe["duration_s"],
        )
        return safe

    # -- planner / loop integration -----------------------------------------
    def capabilities_schema(self) -> list[dict[str, Any]]:
        """Structured connector capabilities for the Planner."""
        return self.registry.capabilities_schema()

    def context_section(self) -> str | None:
        """Compact connector summary for the AgentLoop context.

        Returns None when no connectors are registered, so the
        loop prompt stays clean.  Connector details stay in
        ``connector_capabilities``; this is only a pointer.
        """
        if len(self.registry) == 0:
            return None
        lines = ["Available connectors (external services):"]
        lines.extend(self.registry.summary_lines())
        lines.append(
            "Use connector_capabilities for full operation schemas "
            "and connector_execute to run one. Read and low-risk "
            "writes run once authenticated; sensitive/irreversible "
            "operations need human approval."
        )
        return "\n".join(lines)

    def verifier_observation(self, step: Any = None) -> dict[str, Any]:
        """Fresh, safe connector observation for the Verifier."""
        connector_id = ""
        operation = ""
        arguments: dict[str, Any] = {}
        if step is not None:
            arguments = dict(
                getattr(step, "arguments", None)
                or getattr(step, "args", None)
                or {}
            )
            connector_id = str(arguments.get("connector_id", ""))
            operation = str(arguments.get("operation", ""))
        record = (
            self._connections.get(connector_id)
            if connector_id else None
        )
        last = self._last_results.get((connector_id, operation))
        return {
            "kind": "connector",
            "connector_id": connector_id,
            "operation": operation,
            "connection_status": (
                record.status.value if record else "unknown"
            ),
            "last_result": last,
            "registered_connectors": self.registry.ids(),
        }

    def set_approver(self, approver: Approver | None) -> None:
        self.gate.set_approver(approver)

    # -- internals ---------------------------------------------------------------
    def _record(self, connector_id: str) -> ConnectionRecord:
        record = self._connections.get(connector_id)
        if record is None:
            record = ConnectionRecord(connector_id=connector_id)
            self._connections[connector_id] = record
        return record

    def _context(
        self,
        connector_id: str,
        secrets: dict[str, str] | None = None,
    ) -> OperationContext:
        stored = (
            secrets if secrets is not None
            else self.credentials.get(connector_id)
        )
        session_secrets = self.credentials.get(
            f"{connector_id}:session"
        )
        return OperationContext(
            connector_id=connector_id,
            credentials=dict(stored),
            session_secrets=dict(session_secrets),
            timeout_s=self.default_timeout_s,
        )

    def _store_session_secrets(
        self, connector_id: str, context: OperationContext
    ) -> None:
        # Connectors may stash exchanged tokens on the context;
        # anything they put in session_secrets is persisted in
        # the credential store (never in AgentState/memory).
        if context.session_secrets:
            self.credentials.put(
                f"{connector_id}:session", context.session_secrets
            )

    def _validate_parameters(
        self,
        spec: OperationSpec,
        parameters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        params = dict(parameters or {})
        schema = spec.parameters or {}
        properties: dict[str, Any] = schema.get("properties", {})
        required: list[str] = list(schema.get("required", []))
        missing = [key for key in required if params.get(key) is None]
        if missing:
            raise ConnectorException(
                f"Operation {spec.name!r} is missing required "
                f"parameter(s): {', '.join(missing)}",
                code=ConnectorErrorCode.INVALID_ARGUMENTS,
                operation=spec.name,
                details={"missing": missing},
            )
        for key, declaration in properties.items():
            if key not in params or params[key] is None:
                continue
            expected = (declaration or {}).get("type")
            check = _TYPE_CHECKS.get(expected or "")
            if check is not None and not check(params[key]):
                raise ConnectorException(
                    f"Parameter {key!r} for operation {spec.name!r} "
                    f"must be {expected}, got "
                    f"{type(params[key]).__name__}",
                    code=ConnectorErrorCode.INVALID_ARGUMENTS,
                    operation=spec.name,
                    details={
                        "parameter": key, "expected": expected
                    },
                )
        return params

    def _check_scopes(
        self, connector_id: str, spec: OperationSpec
    ) -> None:
        required = set(spec.required_scopes or ())
        if not required:
            return
        granted = set(self._record(connector_id).granted_scopes)
        missing = sorted(required - granted)
        if missing:
            raise ConnectorException(
                f"Operation {spec.name!r} needs scope(s) "
                f"{', '.join(missing)} which were not granted to "
                f"connector {connector_id!r} (least privilege)",
                code=ConnectorErrorCode.PERMISSION_DENIED,
                connector=connector_id,
                operation=spec.name,
                details={
                    "missing_scopes": missing,
                    "granted_scopes": sorted(granted),
                },
            )

    def _ensure_connected(self, connector_id: str) -> None:
        record = self._record(connector_id)
        if record.status == ConnectionStatus.CONNECTED:
            session = self._sessions.get(connector_id)
            if session is not None and session.expired:
                self._refresh(connector_id)
            return
        # Lazy connect on first use (credentials must be stored).
        self.connect(connector_id)

    def _refresh(self, connector_id: str) -> None:
        connector = self.registry.get(connector_id)
        record = self._record(connector_id)
        context = self._context(connector_id)
        try:
            result = self._run_with_timeout(
                lambda: connector.refresh_session(context),
                self.default_timeout_s,
                connector_id,
                "refresh_session",
            )
        except Exception as e:  # noqa: BLE001 - mapped below
            error = map_exception(
                e, connector=connector_id, operation="refresh_session"
            )
            record.status = ConnectionStatus.AUTH_EXPIRED
            record.last_error = error.message
            raise ConnectorException(
                f"Session refresh failed for {connector_id!r}: "
                f"{error.message}",
                code=ConnectorErrorCode.AUTHENTICATION_EXPIRED,
                connector=connector_id,
                operation="refresh_session",
                details=error.details,
            ) from None
        if result.session is not None:
            self._sessions[connector_id] = result.session
        self._store_session_secrets(connector_id, context)
        record.status = ConnectionStatus.CONNECTED
        record.last_error = None
        logger.info("refreshed session for connector %r", connector_id)

    def _execute_with_refresh(
        self,
        connector: Connector,
        operation: str,
        params: dict[str, Any],
        timeout_s: float,
    ) -> OperationResult:
        connector_id = connector.connector_id
        try:
            return self._call_execute(
                connector, operation, params, timeout_s
            )
        except ConnectorException as e:
            if e.code != ConnectorErrorCode.AUTHENTICATION_EXPIRED:
                raise
            # One transparent refresh + one retry; a second
            # expiry is a real failure, not a loop.
            logger.info(
                "operation %s.%s reported expired auth; "
                "refreshing once",
                connector_id, operation,
            )
            self._refresh(connector_id)
            return self._call_execute(
                connector, operation, params, timeout_s
            )

    def _call_execute(
        self,
        connector: Connector,
        operation: str,
        params: dict[str, Any],
        timeout_s: float,
    ) -> OperationResult:
        connector_id = connector.connector_id
        context = self._context(connector_id)
        context.timeout_s = timeout_s
        try:
            result = self._run_with_timeout(
                lambda: connector.execute_operation(
                    operation, params, context
                ),
                timeout_s,
                connector_id,
                operation,
            )
        except ConnectorException:
            raise
        except Exception as e:  # noqa: BLE001 - mapped to structured
            error = map_exception(
                e, connector=connector_id, operation=operation
            )
            record = self._record(connector_id)
            record.last_error = error.message
            raise ConnectorException(
                error.message, code=error.code,
                connector=connector_id, operation=operation,
                details=error.details,
            ) from None
        if not isinstance(result, OperationResult):
            raise ConnectorException(
                f"Connector {connector_id!r} returned a malformed "
                f"result for {operation!r} (expected "
                "OperationResult)",
                code=ConnectorErrorCode.MALFORMED_RESPONSE,
                connector=connector_id,
                operation=operation,
                details={"returned": type(result).__name__},
            )
        return result

    def _run_with_timeout(
        self,
        func: Callable[[], Any],
        timeout_s: float,
        connector_id: str,
        operation: str,
    ) -> Any:
        """Run *func* with a hard timeout (stdlib threads).

        The pool is shut down without waiting: a timed-out
        worker thread is abandoned, never joined, so a hung
        connector cannot stall the agent.
        """
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="afnan-connector"
        )
        try:
            future = pool.submit(func)
            try:
                return future.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise TimeoutError(
                    f"{connector_id}.{operation} exceeded "
                    f"{timeout_s}s"
                ) from None
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _safe_result(
        self,
        connector_id: str,
        operation: str,
        result: OperationResult,
    ) -> dict[str, Any]:
        summary = str(result.summary or "")[:_SUMMARY_CAP]
        hint = str(result.verification_hint or "")[:500]
        safe = OperationResult(
            connector_id=connector_id,
            operation=operation,
            status=result.status,
            summary=summary,
            verification_hint=hint,
            started_at=result.started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        # Backstop redaction: connectors must already return
        # safe summaries; this guarantees it.
        return redact_value(safe.to_dict())

    def _audit(
        self,
        connector_id: str,
        operation: str,
        risk: str,
        approval: str,
        status: str,
        *,
        error_code: str | None = None,
        duration_s: float = 0.0,
    ) -> None:
        self.audit.record(AuditRecord(
            connector_id=connector_id,
            operation=operation,
            risk=risk,
            approval=approval,
            status=status,
            error_code=error_code,
            duration_s=duration_s,
        ))

    # -- convenience ---------------------------------------------------------------
    @property
    def approver(self) -> Approver | None:
        return self.gate.approver


__all__ = ["ConnectorService"]

"""SecurityAuditCenter — the event & audit authority.

Every important event is recorded with:

* timestamp, task_id, actor
* capability, resource
* policy decision (+ policy version)
* approval status
* execution status, verification status
* failure reason

Sensitive payloads are redacted.  Dedicated security
events: permission denied, approval requested/granted/
rejected, capability escalation, credential access,
prompt injection detected, sandbox violation, abnormal
activity, rate limit exceeded, emergency stop.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.security.audit import AuditLogger
from afnan_ai.security.models import SecurityEventType
from afnan_ai.security.redactor import SecretRedactor


class SecurityAuditCenter:
    """Rich audit API over the tamper-evident log."""

    def __init__(
        self,
        audit: AuditLogger | None = None,
        redactor: SecretRedactor | None = None,
        version_provider: Any = None,
    ) -> None:
        self._audit = audit or AuditLogger()
        self._redactor = redactor or SecretRedactor()
        self._version_provider = version_provider

    @property
    def log(self) -> AuditLogger:
        return self._audit

    def log_decision(
        self,
        *,
        actor: str = "",
        capability: str = "",
        action: str = "",
        resource: dict[str, Any] | None = None,
        task_id: str = "",
        policy_decision: str = "",
        policy_version: str = "",
        approval_status: str = "",
        risk: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        return self._audit.log(
            "policy_decision",
            actor=actor,
            action=action,
            target=self._redactor.redact_text(
                str((resource or {}).get("target", ""))
            )[:200],
            task_id=task_id,
            risk_level=risk,
            permission_result=policy_decision,
            approval_status=approval_status,
            details={
                "capability": capability,
                "resource": self._redactor.for_audit(
                    resource or {}
                ),
                "policy_version": policy_version,
                "reason": self._redactor.redact_text(reason),
            },
        )

    def log_execution(
        self,
        *,
        actor: str = "",
        capability: str = "",
        action: str = "",
        task_id: str = "",
        policy_version: str = "",
        execution_status: str = "",
        verification_status: str = "",
        failure_reason: str = "",
    ) -> dict[str, Any]:
        return self._audit.log(
            "execution_record",
            actor=actor,
            action=action,
            task_id=task_id,
            execution_result=execution_status,
            verification_result=verification_status,
            details={
                "capability": capability,
                "policy_version": policy_version,
                "failure_reason": self._redactor.redact_text(
                    failure_reason
                ),
            },
        )

    def security_event(
        self,
        event: SecurityEventType | str,
        *,
        actor: str = "",
        action: str = "",
        task_id: str = "",
        capability: str = "",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = (
            event.value
            if isinstance(event, SecurityEventType)
            else str(event)
        )
        version = ""
        if self._version_provider is not None:
            try:
                version = str(
                    self._version_provider() or ""
                )
            except Exception:
                version = ""
        return self._audit.log(
            name,
            actor=actor,
            action=action,
            task_id=task_id,
            policy_version=version,
            details={
                "capability": capability,
                **self._redactor.for_audit(details or {}),
            },
        )

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._audit.query(**kwargs)

    def verify_chain(self) -> dict[str, Any]:
        return self._audit.verify_chain()

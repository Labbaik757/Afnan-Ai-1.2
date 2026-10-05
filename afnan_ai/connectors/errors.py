"""Connector errors — structured failures for the Connector System.

Every connector failure, wherever it happens (registry lookup,
permission check, approval gate, network call inside a
connector implementation), is represented as a structured
:class:`ConnectorError` instead of leaking a bare provider
exception:

* ``code`` — machine-readable :class:`ConnectorErrorCode`
* ``message`` — human-readable explanation
* ``connector`` / ``operation`` — which connector/operation failed
* ``details`` — redacted, serializable extras
  (``retryable`` hints whether a retry makes sense)

Exception forms (:class:`ConnectorException` and friends) carry
the same structured error on ``.error`` for callers who prefer
``try/except``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ConnectorErrorCode(str, Enum):
    # Registry / configuration
    INVALID_CONNECTOR = "invalid_connector"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    INVALID_ARGUMENTS = "invalid_arguments"
    # Connectivity / lifecycle
    NOT_CONNECTED = "not_connected"
    MISSING_CREDENTIALS = "missing_credentials"
    # Authentication
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHENTICATION_EXPIRED = "authentication_expired"
    # Authorization (least privilege) and human approval
    PERMISSION_DENIED = "permission_denied"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_DENIED = "approval_denied"
    # Reliability
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    MALFORMED_RESPONSE = "malformed_response"
    SERVICE_UNAVAILABLE = "service_unavailable"
    EXECUTION_FAILED = "execution_failed"


#: Error codes that a retry *may* resolve (with backoff), as
#: opposed to codes that need new input first (credentials,
#: scopes, approval, fixed arguments).
RETRYABLE_CODES = frozenset({
    ConnectorErrorCode.RATE_LIMITED,
    ConnectorErrorCode.TIMEOUT,
    ConnectorErrorCode.NETWORK_ERROR,
    ConnectorErrorCode.SERVICE_UNAVAILABLE,
    ConnectorErrorCode.AUTHENTICATION_EXPIRED,
})


class ConnectorError:
    """Structured, serializable description of a connector failure."""

    def __init__(
        self,
        code: ConnectorErrorCode | str,
        message: str,
        *,
        connector: str | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.code = (
            code
            if isinstance(code, ConnectorErrorCode)
            else ConnectorErrorCode(str(code))
        )
        self.message = str(message)
        self.connector = connector
        self.operation = operation
        self.details = dict(details or {})
        self.details.setdefault(
            "retryable", self.code in RETRYABLE_CODES
        )

    @property
    def retryable(self) -> bool:
        return bool(self.details.get("retryable", False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "connector": self.connector,
            "operation": self.operation,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConnectorError":
        return cls(
            data.get("code", ConnectorErrorCode.EXECUTION_FAILED.value),
            data.get("message", ""),
            connector=data.get("connector"),
            operation=data.get("operation"),
            details=dict(data.get("details") or {}),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ConnectorError(code={self.code.value!r}, "
            f"connector={self.connector!r}, "
            f"operation={self.operation!r})"
        )


class ConnectorException(Exception):
    """Exception carrying a structured :class:`ConnectorError`."""

    default_code = ConnectorErrorCode.EXECUTION_FAILED

    def __init__(
        self,
        message: str,
        *,
        code: ConnectorErrorCode | str | None = None,
        connector: str | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.error = ConnectorError(
            code or self.default_code,
            message,
            connector=connector,
            operation=operation,
            details=details,
        )
        super().__init__(message)

    @property
    def code(self) -> ConnectorErrorCode:
        return self.error.code


class ConnectorNotFoundError(ConnectorException):
    default_code = ConnectorErrorCode.INVALID_CONNECTOR


class UnsupportedOperationError(ConnectorException):
    default_code = ConnectorErrorCode.UNSUPPORTED_OPERATION


class AuthenticationError(ConnectorException):
    default_code = ConnectorErrorCode.AUTHENTICATION_FAILED


class PermissionDeniedError(ConnectorException):
    default_code = ConnectorErrorCode.PERMISSION_DENIED


class ApprovalError(ConnectorException):
    """Sensitive/irreversible operation without human approval."""

    default_code = ConnectorErrorCode.APPROVAL_REQUIRED


def map_exception(
    exc: BaseException,
    *,
    connector: str | None = None,
    operation: str | None = None,
) -> ConnectorError:
    """Convert a connector-raised exception into a structured error.

    Connectors should raise :class:`ConnectorException` (or its
    subclasses) directly.  Anything else is mapped by type so a
    crashed connector never leaks a raw traceback-shaped failure
    into the agent:

    * ``TimeoutError`` -> ``timeout`` (retryable)
    * ``ConnectionError`` -> ``network_error`` (retryable)
    * ``PermissionError`` -> ``permission_denied``
    * anything else -> ``execution_failed``
    """
    if isinstance(exc, ConnectorException):
        error = exc.error
        if error.connector is None:
            error.connector = connector
        if error.operation is None:
            error.operation = operation
        return error
    if isinstance(exc, TimeoutError):
        code = ConnectorErrorCode.TIMEOUT
        message = f"Connector operation timed out: {exc}"
    elif isinstance(exc, ConnectionError):
        code = ConnectorErrorCode.NETWORK_ERROR
        message = f"Connector network failure: {exc}"
    elif isinstance(exc, PermissionError):
        code = ConnectorErrorCode.PERMISSION_DENIED
        message = f"Connector permission denied: {exc}"
    else:
        code = ConnectorErrorCode.EXECUTION_FAILED
        message = f"Connector failed: {type(exc).__name__}: {exc}"
    return ConnectorError(
        code,
        message,
        connector=connector,
        operation=operation,
        details={"exception": type(exc).__name__},
    )

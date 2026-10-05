"""Connector tools — the agent's door to external services.

Six tools, all going through the
:class:`~afnan_ai.connectors.service.ConnectorService`:

* ``connector_list`` — registered connectors (id, name, auth,
  connection status, operation count).
* ``connector_capabilities`` — the full structured capability
  schema for one (or all) connectors: operations, risk
  levels, required scopes, parameter schemas.
* ``connector_connect`` / ``connector_disconnect`` —
  lifecycle using credentials already stored via
  ``ConnectorService.provide_credentials`` (the secure
  channel — credentials are never tool arguments).
* ``connector_health_check`` — liveness probe.
* ``connector_execute`` — run one operation through the
  validate → authorize → connect → approve → execute →
  record pipeline.

Connector failures surface as ``ToolExecutionError`` with the
structured connector error under
``details["connector_error"]`` (codes like
``invalid_connector``, ``unsupported_operation``,
``permission_denied``, ``approval_required``,
``authentication_expired``, ``rate_limited``, ``timeout``),
so the Planner/Executor/Verifier/Recovery machinery treats
them like any other tool failure: approval pauses resumably,
nothing is blindly repeated.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.connectors.errors import ConnectorException
from afnan_ai.connectors.service import ConnectorService
from afnan_ai.tools.base import Tool, ToolExecutionError


def _connector_error(exc: ConnectorException, tool_name: str) -> ToolExecutionError:
    error = exc.error
    return ToolExecutionError(
        error.message,
        tool=tool_name,
        details={"connector_error": error.to_dict()},
    )


class _ConnectorTool(Tool):
    """Base: holds the service, maps connector errors to tools."""

    def __init__(self, service: ConnectorService):
        self.service = service

    def _run_service(self, func, arguments: dict[str, Any]) -> Any:
        try:
            return func(**arguments)
        except ConnectorException as e:
            raise _connector_error(e, self.name) from None


class ConnectorListTool(_ConnectorTool):
    name = "connector_list"
    description = (
        "List registered external-service connectors: id, name, "
        "description, authentication type, connection status and "
        "operations. Use connector_capabilities for full schemas."
    )
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        def _list() -> list[dict[str, Any]]:
            items = []
            for connector in self.service.registry.list():
                record = self.service.connection_info(
                    connector.connector_id
                )
                items.append({
                    "connector_id": connector.connector_id,
                    "name": connector.name,
                    "description": connector.description,
                    "auth_type": connector.info().auth_type,
                    "connection_status": record["status"],
                    "granted_scopes": record["granted_scopes"],
                    "operations": [
                        op.name
                        for op in (connector.operations or ())
                    ],
                })
            return items

        return self._run_service(lambda: _list(), arguments)


class ConnectorCapabilitiesTool(_ConnectorTool):
    name = "connector_capabilities"
    description = (
        "Structured capability schema for connectors: operations, "
        "risk levels (read/low_risk_write/sensitive_write/"
        "irreversible_destructive), required permission scopes "
        "and parameter schemas. Omit connector_id for all."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "connector_id": {
                "type": "string",
                "description": "Connector id, e.g. 'email'",
            },
        },
        "required": [],
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        def _caps(connector_id: str | None = None):
            if connector_id:
                connector = self.service.registry.get(
                    connector_id
                )
                return connector.info().to_dict()
            return self.service.capabilities_schema()

        try:
            return _caps(arguments.get("connector_id"))
        except ConnectorException as e:
            raise _connector_error(e, self.name) from None


class ConnectorConnectTool(_ConnectorTool):
    name = "connector_connect"
    description = (
        "Connect a connector using its stored credentials. "
        "Credentials must already be stored via the secure "
        "credential channel (ConnectorService.provide_credentials) "
        "— never pass secrets as tool arguments."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "connector_id": {"type": "string"},
        },
        "required": ["connector_id"],
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._run_service(
            self.service.connect, arguments
        )


class ConnectorDisconnectTool(_ConnectorTool):
    name = "connector_disconnect"
    description = (
        "Disconnect a connector and drop its session "
        "(stored credentials are kept until revoked)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "connector_id": {"type": "string"},
        },
        "required": ["connector_id"],
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._run_service(
            self.service.disconnect, arguments
        )


class ConnectorHealthCheckTool(_ConnectorTool):
    name = "connector_health_check"
    description = (
        "Probe a connector's liveness without side effects."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "connector_id": {"type": "string"},
        },
        "required": ["connector_id"],
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._run_service(
            self.service.health_check, arguments
        )


class ConnectorExecuteTool(_ConnectorTool):
    name = "connector_execute"
    description = (
        "Execute one connector operation (external service call). "
        "Parameters are validated against the operation's schema; "
        "required permission scopes are enforced (least "
        "privilege); read and low-risk writes run once "
        "authenticated, while sensitive/irreversible operations "
        "need human approval first. Returns safe metadata only: "
        "connector, operation, status, redacted summary, "
        "timestamps. Returns a structured connector error on "
        "failure (invalid_connector, unsupported_operation, "
        "permission_denied, approval_required/denied, "
        "authentication_failed/expired, rate_limited, timeout, "
        "network_error, service_unavailable)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "connector_id": {
                "type": "string",
                "description": "Connector id, e.g. 'email'",
            },
            "operation": {
                "type": "string",
                "description": "Operation name from the schema",
            },
            "parameters": {
                "type": "object",
                "description": "Operation parameters (validated)",
            },
        },
        "required": ["connector_id", "operation"],
    }

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._run_service(
            lambda connector_id, operation, parameters=None: (
                self.service.execute(
                    connector_id, operation, parameters
                )
            ),
            arguments,
        )


def create_connector_tools(
    service: ConnectorService,
) -> list[Tool]:
    """Build the six connector tools bound to *service*."""
    return [
        ConnectorListTool(service),
        ConnectorCapabilitiesTool(service),
        ConnectorConnectTool(service),
        ConnectorDisconnectTool(service),
        ConnectorHealthCheckTool(service),
        ConnectorExecuteTool(service),
    ]


__all__ = ["create_connector_tools"]

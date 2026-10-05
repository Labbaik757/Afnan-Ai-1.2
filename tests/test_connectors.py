"""Connector System tests: interface, registry, credentials,
authentication, least-privilege scopes, approval flow, secret
redaction, reliability (timeout/rate-limit/network/malformed),
session refresh, audit trail, AgentLoop integration and an
end-to-end email -> calendar workflow through mock connectors
(no external services)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from afnan_ai.agent import AfnanAgent
from afnan_ai.connectors import (
    AuthSession,
    AuthType,
    ConnectionResult,
    ConnectionStatus,
    Connector,
    ConnectorApprovalGate,
    ConnectorErrorCode,
    ConnectorException,
    ConnectorRegistry,
    ConnectorService,
    HealthResult,
    MemoryCredentialStore,
    OperationResult,
    OperationSpec,
    RiskLevel,
    RiskPolicy,
    UnsupportedOperationError,
    create_connector_tools,
    require_fields,
    session_expiry_in,
)
from afnan_ai.connectors.auth import AuthSession as _AuthSessionCheck
from afnan_ai.connectors.errors import map_exception
from afnan_ai.state import AgentState

from test_browser_advanced import QueueLLM, plan_json, step


SECRET = "SECRET-123"
CAL_SECRET = "CAL-SECRET"


class MockEmailConnector(Connector):
    connector_id = "email"
    name = "Email"
    description = "Mock email service for tests"
    auth_type = AuthType.API_KEY
    declared_scopes = ("email.read", "email.send", "email.delete")
    operations = (
        OperationSpec(
            "search_emails", "Search the inbox", RiskLevel.READ,
            ("email.read",),
            {"type": "object",
             "properties": {"query": {"type": "string"}},
             "required": ["query"]},
        ),
        OperationSpec(
            "send_email", "Send an email", RiskLevel.SENSITIVE_WRITE,
            ("email.send",),
            {"type": "object",
             "properties": {"to": {"type": "string"},
                            "subject": {"type": "string"}},
             "required": ["to"]},
        ),
        OperationSpec(
            "delete_email", "Delete an email forever",
            RiskLevel.DESTRUCTIVE, ("email.delete",),
            {"type": "object",
             "properties": {"message_id": {"type": "string"}},
             "required": ["message_id"]},
        ),
    )

    def __init__(self):
        self.calls: list[tuple] = []

    def authenticate(self, credentials, context):
        require_fields(
            credentials, ("api_key",), connector_id=self.connector_id
        )
        if credentials["api_key"] != SECRET:
            raise self._auth_failed("invalid api key")
        return AuthSession(
            auth_type=self.auth_type, secret_ref=self.connector_id,
            expires_at=session_expiry_in(3600),
            scopes=self.declared_scopes,
        )

    def connect(self, context):
        return ConnectionResult(
            self.connector_id, ConnectionStatus.CONNECTED,
            detail="connected",
        )

    def health_check(self, context):
        return HealthResult(self.connector_id, True, "ok")

    def execute_operation(self, operation, parameters, context):
        self.calls.append((operation, dict(parameters)))
        # The secret reaches the connector transiently, through
        # the per-call context only.
        assert context.credentials.get("api_key") == SECRET
        if operation == "search_emails":
            return self._result(
                operation,
                f"Found 2 emails about '{parameters['query']}'",
                verification_hint="msg-1,msg-2",
            )
        if operation == "send_email":
            return self._result(
                operation, f"Sent email to {parameters['to']}"
            )
        if operation == "delete_email":
            return self._result(
                operation, f"Deleted {parameters['message_id']}"
            )
        raise UnsupportedOperationError(
            f"no such op {operation}", connector=self.connector_id,
            operation=operation,
        )


class MockCalendarConnector(Connector):
    connector_id = "calendar"
    name = "Calendar"
    description = "Mock calendar service for tests"
    auth_type = AuthType.OAUTH2
    declared_scopes = ("calendar.read", "calendar.write")
    operations = (
        OperationSpec(
            "list_events", "List upcoming events", RiskLevel.READ,
            ("calendar.read",),
            {"type": "object", "properties": {}, "required": []},
        ),
        OperationSpec(
            "create_event", "Create a calendar event",
            RiskLevel.SENSITIVE_WRITE, ("calendar.write",),
            {"type": "object",
             "properties": {"title": {"type": "string"}},
             "required": ["title"]},
        ),
    )

    def __init__(self, token_lifetime_s=3600):
        self.token_lifetime_s = token_lifetime_s
        self.refreshes = 0
        self.fail_refresh = False
        self.expire_once = False
        self.calls: list[tuple] = []

    def authenticate(self, credentials, context):
        require_fields(
            credentials, ("client_id", "client_secret"),
            connector_id=self.connector_id,
        )
        if credentials["client_secret"] != CAL_SECRET:
            raise self._auth_failed("bad client secret")
        context.session_secrets["access_token"] = "TOKEN-1"
        return AuthSession(
            auth_type=self.auth_type, secret_ref=self.connector_id,
            expires_at=session_expiry_in(self.token_lifetime_s),
            scopes=self.declared_scopes,
        )

    def connect(self, context):
        return ConnectionResult(
            self.connector_id, ConnectionStatus.CONNECTED
        )

    def refresh_session(self, context):
        self.refreshes += 1
        if self.fail_refresh:
            raise ConnectorException(
                "provider rejected the refresh token",
                code=ConnectorErrorCode.AUTHENTICATION_FAILED,
                connector=self.connector_id,
            )
        context.session_secrets["access_token"] = "TOKEN-2"
        return ConnectionResult(
            self.connector_id, ConnectionStatus.CONNECTED,
            session=AuthSession(
                auth_type=self.auth_type,
                secret_ref=self.connector_id,
                expires_at=session_expiry_in(3600),
            ),
            detail="refreshed",
        )

    def health_check(self, context):
        return HealthResult(self.connector_id, True, "ok")

    def execute_operation(self, operation, parameters, context):
        self.calls.append((operation, dict(parameters)))
        if self.expire_once:
            self.expire_once = False
            raise ConnectorException(
                "token expired mid-call",
                code=ConnectorErrorCode.AUTHENTICATION_EXPIRED,
                connector=self.connector_id, operation=operation,
            )
        if operation == "list_events":
            return self._result(operation, "3 upcoming events")
        return self._result(
            operation, f"Created event '{parameters['title']}'",
            verification_hint="evt-99",
        )


class MockFlakyConnector(Connector):
    connector_id = "flaky"
    name = "Flaky"
    description = "Mock unreliable service for tests"
    auth_type = AuthType.NONE
    operations = (
        OperationSpec("hang", "Hangs forever", RiskLevel.READ),
        OperationSpec("unstable", "Fails on demand", RiskLevel.READ),
        OperationSpec(
            "bad_shape", "Returns a malformed result", RiskLevel.READ
        ),
    )

    def __init__(self):
        self.mode = "ok"
        self.calls = 0
        self.stop_hang = threading.Event()

    def authenticate(self, credentials, context):
        return AuthSession(auth_type=self.auth_type)

    def connect(self, context):
        return ConnectionResult(
            self.connector_id, ConnectionStatus.CONNECTED
        )

    def health_check(self, context):
        return HealthResult(self.connector_id, True, "ok")

    def execute_operation(self, operation, parameters, context):
        self.calls += 1
        if operation == "hang":
            self.stop_hang.wait(30)  # interruptible in tearDown
            raise TimeoutError("hang interrupted")  # pragma: no cover
        if operation == "bad_shape":
            return {"not": "an OperationResult"}  # type: ignore
        if self.mode == "rate_limited":
            raise ConnectorException(
                "slow down", code=ConnectorErrorCode.RATE_LIMITED,
                connector=self.connector_id, operation=operation,
            )
        if self.mode == "network":
            raise ConnectionError("connection reset")
        if self.mode == "boom":
            raise ValueError("unexpected bug")
        return self._result(operation, "flaky ok")


def make_service(**kwargs):
    tmp = tempfile.TemporaryDirectory()
    registry = ConnectorRegistry()
    email = MockEmailConnector()
    calendar = MockCalendarConnector()
    flaky = MockFlakyConnector()
    registry.register_many([email, calendar, flaky])
    service = ConnectorService(
        registry,
        audit_path=os.path.join(tmp.name, "audit.jsonl"),
        **kwargs,
    )
    return service, email, calendar, flaky, tmp


class TestConnectorInterface(unittest.TestCase):
    def test_info_schema_shape(self):
        info = MockEmailConnector().info().to_dict()
        self.assertEqual(info["connector_id"], "email")
        self.assertEqual(info["auth_type"], "api_key")
        ops = {op["name"]: op for op in info["operations"]}
        self.assertEqual(ops["send_email"]["risk"], "sensitive_write")
        self.assertEqual(
            ops["send_email"]["required_scopes"], ["email.send"]
        )
        self.assertEqual(
            ops["search_emails"]["parameters"]["required"], ["query"]
        )

    def test_duplicate_operation_rejected(self):
        class Bad(Connector):
            connector_id = "bad"
            operations = (
                OperationSpec("op", "x"),
                OperationSpec("op", "y"),
            )

            def authenticate(self, c, ctx): raise AssertionError
            def connect(self, ctx): raise AssertionError
            def health_check(self, ctx): raise AssertionError
            def execute_operation(self, o, p, ctx): raise AssertionError

        with self.assertRaises(ConnectorException):
            ConnectorRegistry().register(Bad())

    def test_non_connector_rejected(self):
        with self.assertRaises(ConnectorException):
            ConnectorRegistry().register(object())

    def test_duplicate_id_rejected(self):
        registry = ConnectorRegistry()
        registry.register(MockEmailConnector())
        with self.assertRaises(ConnectorException):
            registry.register(MockEmailConnector())


class TestRegistry(unittest.TestCase):
    def test_invalid_connector_structured(self):
        registry = ConnectorRegistry()
        with self.assertRaises(ConnectorException) as caught:
            registry.get("nope")
        self.assertEqual(
            caught.exception.code, ConnectorErrorCode.INVALID_CONNECTOR
        )

    def test_unsupported_operation_structured(self):
        connector = MockEmailConnector()
        with self.assertRaises(ConnectorException) as caught:
            connector.get_operation("fax")
        self.assertEqual(
            caught.exception.code,
            ConnectorErrorCode.UNSUPPORTED_OPERATION,
        )
        self.assertIn(
            "search_emails",
            caught.exception.error.details["available"],
        )

    def test_discover_is_best_effort(self):
        registry = ConnectorRegistry()
        # No entry points installed in the test env: returns []
        # and never raises.
        self.assertEqual(registry.discover(), [])

    def test_capabilities_schema(self):
        service, _, _, _, tmp = make_service()
        try:
            schema = service.capabilities_schema()
            by_id = {c["connector_id"]: c for c in schema}
            self.assertIn("email", by_id)
            self.assertIn("calendar", by_id)
            email_ops = {
                op["name"]: op["risk"]
                for op in by_id["email"]["operations"]
            }
            self.assertEqual(
                email_ops,
                {
                    "search_emails": "read",
                    "send_email": "sensitive_write",
                    "delete_email": "irreversible_destructive",
                },
            )
        finally:
            tmp.cleanup()


class TestCredentials(unittest.TestCase):
    def test_put_get_delete(self):
        store = MemoryCredentialStore()
        store.put("email", {"api_key": SECRET})
        self.assertTrue(store.has("email"))
        self.assertEqual(store.get("email"), {"api_key": SECRET})
        # namespaces() exposes ids only, never values
        self.assertEqual(store.namespaces(), ["email"])
        self.assertNotIn(SECRET, str(store.namespaces()))
        self.assertTrue(store.delete("email"))
        self.assertFalse(store.has("email"))

    def test_service_rejects_unknown_connector_credentials(self):
        service, _, _, _, tmp = make_service()
        try:
            with self.assertRaises(ConnectorException) as caught:
                service.provide_credentials("nope", {"k": "v"})
            self.assertEqual(
                caught.exception.code,
                ConnectorErrorCode.INVALID_CONNECTOR,
            )
        finally:
            tmp.cleanup()


class TestAuth(unittest.TestCase):
    def test_require_fields(self):
        with self.assertRaises(ConnectorException) as caught:
            require_fields({}, ("api_key",), connector_id="email")
        self.assertEqual(
            caught.exception.code,
            ConnectorErrorCode.MISSING_CREDENTIALS,
        )
        self.assertNotIn("SECRET", str(caught.exception))

    def test_session_expiry(self):
        live = _AuthSessionCheck(
            expires_at=session_expiry_in(3600)
        )
        dead = _AuthSessionCheck(expires_at=session_expiry_in(-10))
        self.assertFalse(live.expired)
        self.assertTrue(dead.expired)
        self.assertFalse(_AuthSessionCheck().expired)

    def test_connect_missing_credentials(self):
        service, _, _, _, tmp = make_service()
        try:
            with self.assertRaises(ConnectorException) as caught:
                service.connect("email")
            self.assertEqual(
                caught.exception.code,
                ConnectorErrorCode.MISSING_CREDENTIALS,
            )
        finally:
            tmp.cleanup()

    def test_connect_bad_credentials(self):
        service, _, _, _, tmp = make_service()
        try:
            service.provide_credentials("email", {"api_key": "wrong"})
            with self.assertRaises(ConnectorException) as caught:
                service.connect("email")
            self.assertEqual(
                caught.exception.code,
                ConnectorErrorCode.AUTHENTICATION_FAILED,
            )
            self.assertNotIn("wrong", str(caught.exception.error.details))
        finally:
            tmp.cleanup()

    def test_connect_health_disconnect(self):
        service, _, _, _, tmp = make_service()
        try:
            service.provide_credentials("email", {"api_key": SECRET})
            connected = service.connect("email")
            self.assertEqual(connected["status"], "connected")
            health = service.health_check("email")
            self.assertTrue(health["healthy"])
            info = service.connection_info("email")
            self.assertEqual(info["status"], "connected")
            service.disconnect("email")
            self.assertEqual(
                service.connection_info("email")["status"],
                "disconnected",
            )
        finally:
            tmp.cleanup()


class TestLeastPrivilege(unittest.TestCase):
    def setUp(self):
        self.service, self.email, _, _, self.tmp = make_service()
        self.service.provide_credentials("email", {"api_key": SECRET})

    def tearDown(self):
        self.tmp.cleanup()

    def test_grant_unknown_scope_rejected(self):
        with self.assertRaises(ConnectorException) as caught:
            self.service.grant_scopes("email", ["root.all"])
        self.assertEqual(
            caught.exception.code, ConnectorErrorCode.INVALID_ARGUMENTS
        )

    def test_operation_without_scope_denied(self):
        # No scopes granted: even a read is refused (least
        # privilege), and the approver cannot override it.
        self.service.set_approver(lambda req: True)
        with self.assertRaises(ConnectorException) as caught:
            self.service.execute(
                "email", "search_emails", {"query": "x"}
            )
        self.assertEqual(
            caught.exception.code, ConnectorErrorCode.PERMISSION_DENIED
        )
        self.assertEqual(self.email.calls, [])

    def test_granted_scope_allows_read(self):
        self.service.grant_scopes("email", ["email.read"])
        result = self.service.execute(
            "email", "search_emails", {"query": "invoice"}
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("invoice", result["summary"])

    def test_revoke_scopes(self):
        self.service.grant_scopes(
            "email", ["email.read", "email.send"]
        )
        remaining = self.service.revoke_scopes("email", ["email.send"])
        self.assertEqual(remaining, ["email.read"])

    def test_invalid_arguments(self):
        self.service.grant_scopes("email", ["email.read"])
        with self.assertRaises(ConnectorException) as caught:
            self.service.execute("email", "search_emails", {})
        self.assertEqual(
            caught.exception.code, ConnectorErrorCode.INVALID_ARGUMENTS
        )
        with self.assertRaises(ConnectorException) as caught:
            self.service.execute(
                "email", "search_emails", {"query": 42}
            )
        self.assertEqual(
            caught.exception.code, ConnectorErrorCode.INVALID_ARGUMENTS
        )

    def test_invalid_connector_and_operation(self):
        with self.assertRaises(ConnectorException) as caught:
            self.service.execute("nope", "x", {})
        self.assertEqual(
            caught.exception.code, ConnectorErrorCode.INVALID_CONNECTOR
        )
        with self.assertRaises(ConnectorException) as caught:
            self.service.execute("email", "fax", {})
        self.assertEqual(
            caught.exception.code,
            ConnectorErrorCode.UNSUPPORTED_OPERATION,
        )


class TestApprovalFlow(unittest.TestCase):
    def setUp(self):
        self.service, self.email, _, _, self.tmp = make_service()
        self.service.provide_credentials("email", {"api_key": SECRET})
        self.service.grant_scopes("email", ["email.send", "email.delete"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_sensitive_needs_approver(self):
        with self.assertRaises(ConnectorException) as caught:
            self.service.execute(
                "email", "send_email", {"to": "a@b.c"}
            )
        self.assertEqual(
            caught.exception.code, ConnectorErrorCode.APPROVAL_REQUIRED
        )
        self.assertEqual(self.email.calls, [])

    def test_denied_by_human(self):
        self.service.set_approver(lambda req: False)
        with self.assertRaises(ConnectorException) as caught:
            self.service.execute(
                "email", "send_email", {"to": "a@b.c"}
            )
        self.assertEqual(
            caught.exception.code, ConnectorErrorCode.APPROVAL_DENIED
        )
        self.assertEqual(self.email.calls, [])

    def test_approved_runs(self):
        seen = {}

        def approver(request):
            seen["op"] = request.operation
            seen["risk"] = request.risk
            # Approval requests carry redacted parameters only.
            self.assertNotIn(SECRET, json.dumps(request.to_dict()))
            return True

        self.service.set_approver(approver)
        result = self.service.execute(
            "email", "send_email",
            {"to": "a@b.c", "api_key": SECRET},
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(seen["op"], "send_email")
        self.assertEqual(seen["risk"], "sensitive_write")
        self.assertNotIn(SECRET, json.dumps(result))

    def test_destructive_blocked_by_policy(self):
        policy = RiskPolicy(blocked=frozenset({RiskLevel.DESTRUCTIVE}))
        self.service.gate = ConnectorApprovalGate(
            lambda req: True, policy=policy
        )
        with self.assertRaises(ConnectorException) as caught:
            self.service.execute(
                "email", "delete_email", {"message_id": "m1"}
            )
        self.assertEqual(
            caught.exception.code, ConnectorErrorCode.PERMISSION_DENIED
        )
        self.assertTrue(
            caught.exception.error.details["blocked_by_policy"]
        )
        self.assertEqual(self.email.calls, [])


class TestSecretRedaction(unittest.TestCase):
    def test_no_secret_in_results_audit_or_gate(self):
        service, email, _, _, tmp = make_service()
        try:
            service.provide_credentials(
                "email", {"api_key": SECRET}
            )
            service.grant_scopes(
                "email", ["email.read", "email.send"]
            )
            service.set_approver(lambda req: True)
            result = service.execute(
                "email", "send_email",
                {"to": "a@b.c", "api_key": SECRET},
            )
            blob = json.dumps(result, default=str)
            self.assertNotIn(SECRET, blob)
            # gate decisions (approval requests) are redacted
            self.assertNotIn(
                SECRET, json.dumps(service.gate.decisions)
            )
            # audit file is redacted
            with open(
                os.path.join(tmp.name, "audit.jsonl"),
                encoding="utf-8",
            ) as handle:
                audit_blob = handle.read()
            self.assertNotIn(SECRET, audit_blob)
            self.assertIn("send_email", audit_blob)
        finally:
            tmp.cleanup()


class TestReliability(unittest.TestCase):
    def setUp(self):
        self.service, _, _, self.flaky, self.tmp = make_service()

    def tearDown(self):
        self.flaky.stop_hang.set()  # release any hung worker
        self.tmp.cleanup()

    def test_timeout_structured(self):
        started = time.monotonic()
        with self.assertRaises(ConnectorException) as caught:
            self.service.execute("flaky", "hang", {}, timeout_s=0.3)
        self.assertEqual(
            caught.exception.code, ConnectorErrorCode.TIMEOUT
        )
        self.assertTrue(caught.exception.error.retryable)
        self.assertLess(time.monotonic() - started, 10)

    def test_rate_limited_no_blind_retry(self):
        self.flaky.mode = "rate_limited"
        with self.assertRaises(ConnectorException) as caught:
            self.service.execute("flaky", "unstable", {})
        self.assertEqual(
            caught.exception.code, ConnectorErrorCode.RATE_LIMITED
        )
        self.assertTrue(caught.exception.error.retryable)
        # The service made exactly one attempt: no blind repeat.
        self.assertEqual(self.flaky.calls, 1)

    def test_network_error_mapped(self):
        self.flaky.mode = "network"
        with self.assertRaises(ConnectorException) as caught:
            self.service.execute("flaky", "unstable", {})
        self.assertEqual(
            caught.exception.code, ConnectorErrorCode.NETWORK_ERROR
        )

    def test_unexpected_exception_mapped(self):
        self.flaky.mode = "boom"
        with self.assertRaises(ConnectorException) as caught:
            self.service.execute("flaky", "unstable", {})
        self.assertEqual(
            caught.exception.code, ConnectorErrorCode.EXECUTION_FAILED
        )
        self.assertNotIn("traceback", str(caught.exception).lower())

    def test_malformed_result(self):
        with self.assertRaises(ConnectorException) as caught:
            self.service.execute("flaky", "bad_shape", {})
        self.assertEqual(
            caught.exception.code,
            ConnectorErrorCode.MALFORMED_RESPONSE,
        )

    def test_map_exception_types(self):
        self.assertEqual(
            map_exception(TimeoutError("t")).code,
            ConnectorErrorCode.TIMEOUT,
        )
        self.assertEqual(
            map_exception(ConnectionError("c")).code,
            ConnectorErrorCode.NETWORK_ERROR,
        )
        self.assertEqual(
            map_exception(PermissionError("p")).code,
            ConnectorErrorCode.PERMISSION_DENIED,
        )


class TestSessionRefresh(unittest.TestCase):
    def _calendar_service(self, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        registry = ConnectorRegistry()
        calendar = MockCalendarConnector(**kwargs)
        registry.register(calendar)
        service = ConnectorService(registry)
        service.provide_credentials(
            "calendar",
            {"client_id": "cid", "client_secret": CAL_SECRET},
        )
        service.grant_scopes(
            "calendar", ["calendar.read", "calendar.write"]
        )
        service.set_approver(lambda req: True)
        # Connect with a live token first; individual tests then
        # expire the stored session to exercise refresh paths.
        service.connect("calendar")
        return service, calendar, tmp

    def _expire_session(self, service):
        service._sessions["calendar"] = AuthSession(
            auth_type=AuthType.OAUTH2,
            secret_ref="calendar",
            expires_at=session_expiry_in(-5),
        )

    def test_expired_session_refreshes_before_execute(self):
        service, calendar, tmp = self._calendar_service()
        try:
            self._expire_session(service)
            result = service.execute(
                "calendar", "list_events", {}
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(calendar.refreshes, 1)
        finally:
            tmp.cleanup()

    def test_mid_call_expiry_refreshes_and_retries_once(self):
        service, calendar, tmp = self._calendar_service()
        try:
            calendar.expire_once = True
            result = service.execute(
                "calendar", "create_event", {"title": "Standup"}
            )
            self.assertEqual(result["status"], "ok")
            self.assertIn("Standup", result["summary"])
            self.assertEqual(calendar.refreshes, 1)
            # one failed attempt + one retry
            self.assertEqual(len(calendar.calls), 2)
        finally:
            tmp.cleanup()

    def test_failed_refresh_surfaces_expired(self):
        service, calendar, tmp = self._calendar_service()
        try:
            calendar.fail_refresh = True
            self._expire_session(service)
            with self.assertRaises(ConnectorException) as caught:
                service.execute("calendar", "list_events", {})
            self.assertEqual(
                caught.exception.code,
                ConnectorErrorCode.AUTHENTICATION_EXPIRED,
            )
        finally:
            tmp.cleanup()


class TestAuditTrail(unittest.TestCase):
    def test_audit_records_safe_fields(self):
        service, _, _, _, tmp = make_service()
        try:
            service.provide_credentials("email", {"api_key": SECRET})
            service.grant_scopes("email", ["email.read"])
            service.execute(
                "email", "search_emails", {"query": "x"}
            )
            entries = service.audit.recent(10)
            kinds = {
                (e["operation"], e["status"]) for e in entries
            }
            self.assertIn(("connect", "ok"), kinds)
            self.assertIn(("search_emails", "ok"), kinds)
            for entry in entries:
                for field in (
                    "connector_id", "operation", "risk",
                    "timestamp", "approval", "status",
                ):
                    self.assertIn(field, entry)
                self.assertNotIn(SECRET, json.dumps(entry))
        finally:
            tmp.cleanup()

    def test_audit_captures_approval_denial(self):
        service, _, _, _, tmp = make_service()
        try:
            service.provide_credentials("email", {"api_key": SECRET})
            service.grant_scopes("email", ["email.send"])
            service.set_approver(lambda req: False)
            with self.assertRaises(ConnectorException):
                service.execute(
                    "email", "send_email", {"to": "a@b.c"}
                )
            denied = [
                e for e in service.audit.recent(10)
                if e["operation"] == "send_email"
            ]
            self.assertEqual(denied[-1]["approval"], "denied")
            self.assertEqual(denied[-1]["status"], "failed")
            self.assertEqual(
                denied[-1]["error_code"], "approval_denied"
            )
        finally:
            tmp.cleanup()


class TestConnectorTools(unittest.TestCase):
    def test_six_tools_registered(self):
        service, _, _, _, tmp = make_service()
        try:
            names = sorted(
                t.name for t in create_connector_tools(service)
            )
            self.assertEqual(
                names,
                [
                    "connector_capabilities",
                    "connector_connect",
                    "connector_disconnect",
                    "connector_execute",
                    "connector_health_check",
                    "connector_list",
                ],
            )
        finally:
            tmp.cleanup()

    def test_tool_error_mapping(self):
        service, _, _, _, tmp = make_service()
        try:
            from afnan_ai.tools import ToolRegistry

            registry = ToolRegistry(
                create_connector_tools(service)
            )
            result = registry.execute(
                "connector_execute",
                {"connector_id": "nope", "operation": "x"},
            )
            self.assertFalse(result.success)
            details = result.error.details["connector_error"]
            self.assertEqual(
                details["code"], "invalid_connector"
            )
        finally:
            tmp.cleanup()


class TestAgentIntegration(unittest.TestCase):
    def _agent(self, replies, **kwargs):
        registry = ConnectorRegistry()
        email = MockEmailConnector()
        calendar = MockCalendarConnector()
        registry.register_many([email, calendar])
        agent = AfnanAgent(
            llm_provider=QueueLLM(replies),
            connector_registry=registry,
            enable_browser_tools=False,
            enable_screen_tools=False,
            enable_computer_tools=False,
            memory_dir=tempfile.mkdtemp(),
            **kwargs,
        )
        service = agent.get_connector_service()
        service.provide_credentials("email", {"api_key": SECRET})
        service.provide_credentials(
            "calendar",
            {"client_id": "cid", "client_secret": CAL_SECRET},
        )
        service.grant_scopes("email", ["email.read"])
        service.grant_scopes(
            "calendar", ["calendar.read", "calendar.write"]
        )
        return agent, service, email, calendar

    def test_end_to_end_email_to_calendar(self):
        goal = "Read invoice email, then create a calendar event"
        steps = [
            step(
                "e1", "connector_execute",
                {"connector_id": "email", "operation": "search_emails",
                 "parameters": {"query": "invoice"}},
                "Found 2 emails about 'invoice'",
            ),
            step(
                "e2", "connector_execute",
                {"connector_id": "calendar",
                 "operation": "create_event",
                 "parameters": {"title": "Pay invoice"}},
                "Created event 'Pay invoice'",
            ),
        ]
        replies = [
            plan_json(goal, steps),
            '{"goal": "done", "complete": true, "steps": []}',
        ]
        agent, service, email, calendar = self._agent(
            replies, connector_approver=lambda req: True
        )
        from afnan_ai.orchestrator import OrchestrationStatus

        result = agent.run_agent_loop(goal)
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        # The Planner chose Browser->Connector routing itself:
        # email was read, then a (approved) calendar write ran.
        self.assertEqual(
            [c[0] for c in email.calls], ["search_emails"]
        )
        self.assertEqual(
            [c[0] for c in calendar.calls], ["create_event"]
        )
        # AgentState carries safe metadata only — no secrets.
        state_blob = json.dumps(result.state.to_dict(), default=str)
        self.assertNotIn(SECRET, state_blob)
        self.assertNotIn(CAL_SECRET, state_blob)
        self.assertIn("search_emails", state_blob)
        # Audit trail recorded both operations with approval info.
        ops = {
            (e["connector_id"], e["operation"], e["approval"])
            for e in service.audit.recent(20)
        }
        self.assertIn(("email", "search_emails", "not_required"), ops)
        self.assertIn(
            ("calendar", "create_event", "granted"), ops
        )

    def test_sensitive_without_approver_parks_task(self):
        goal = "Create a calendar event"
        steps = [
            step(
                "e1", "connector_execute",
                {"connector_id": "calendar",
                 "operation": "create_event",
                 "parameters": {"title": "X"}},
                "Created event",
            ),
        ]
        replies = [plan_json(goal, steps)]
        agent, service, _, calendar = self._agent(replies)
        from afnan_ai.orchestrator import OrchestrationStatus

        result = agent.run_agent_loop(goal)
        # Approval refusal pauses the task resumably (existing
        # AgentLoop behavior): FAILED with an approval_required
        # error, checkpoint saved — it does not burn recovery.
        self.assertEqual(result.status, OrchestrationStatus.FAILED)
        self.assertEqual(
            (result.error or {}).get("code"), "approval_required"
        )
        self.assertEqual(calendar.calls, [])

    def test_loop_context_lists_connectors(self):
        agent, _, _, _ = self._agent(["{}"])
        loop = agent.get_agent_loop()
        context = loop._decision_context(
            AgentState(goal="g"), goal="g", cycle=1, observation=None,
            injection_flagged=False, note="",
        )
        self.assertIn("Available connectors", context)
        self.assertIn("email", context)
        self.assertIn("calendar", context)

    def test_loop_context_empty_without_connectors(self):
        agent = AfnanAgent(
            llm_provider=QueueLLM(["{}"]),
            enable_browser_tools=False,
            enable_screen_tools=False,
            enable_computer_tools=False,
            memory_dir=tempfile.mkdtemp(),
        )
        loop = agent.get_agent_loop()
        context = loop._decision_context(
            AgentState(goal="g"), goal="g", cycle=1, observation=None,
            injection_flagged=False, note="",
        )
        self.assertNotIn("Available connectors", context)

    def test_verifier_observation_is_safe(self):
        agent, service, _, _ = self._agent(["{}"])
        service.provide_credentials("email", {"api_key": SECRET})

        class Step:
            tool_name = "connector_execute"
            arguments = {
                "connector_id": "email", "operation": "search_emails"
            }

        obs = agent._verifier_observation_provider(Step())
        self.assertEqual(obs["kind"], "connector")
        self.assertNotIn(SECRET, json.dumps(obs, default=str))


if __name__ == "__main__":
    unittest.main()

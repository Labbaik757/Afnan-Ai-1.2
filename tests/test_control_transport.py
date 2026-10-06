"""Transport-level tests for the Remote & Multi-Device Control Plane.

Exercises the REAL network stack (``ControlTransport``: stdlib HTTP +
SSE + WebSocket) against the REAL ``ControlPlane`` wired to the REAL
``SecurityCenter`` / ``TaskManager`` / ``GoalManager`` /
``ActivityCenter`` / ``ApprovalCenter``.  Nothing in the control plane
itself is mocked.

Helpers (``make_plane``/``pair_device``/``grant``/``cmd``) are small
copies of the ones in ``tests/test_control_plane.py`` so this module
stays runnable under any test runner.

Ports 18766-18769 on 127.0.0.1 are used; nothing leaves loopback.

Note: an earlier draft of this file marked the HTTP 401 auth-mapping
tests as expected failures because the transport mapped
``AuthenticationError`` to 500.  The transport now maps them to 401,
so those tests assert the real (fixed) behavior directly.
"""

import base64
import http.client
import json
import os
import random
import socket
import struct
import threading
import time
import unittest
import urllib.error
import urllib.request
import tempfile
from pathlib import Path

from afnan_ai.activity.approval import ApprovalCenter
from afnan_ai.activity.center import ActivityCenter
from afnan_ai.control import (
    AuthenticationError,
    CommandType,
    ControlPlane,
    DeviceTrust,
    SessionState,
)
from afnan_ai.control.pairing import PairingError
from afnan_ai.control.transport import ControlTransport, _WSFrame
from afnan_ai.goal_manager import GoalManager
from afnan_ai.security.center import SecurityCenter
from afnan_ai.security.models import Actor, ActorKind
from afnan_ai.task_manager import TaskManager


# ---------------------------------------------------------------------------
# Helpers (mirrored from tests/test_control_plane.py)
# ---------------------------------------------------------------------------


def make_plane(**kwargs):
    tmp = tempfile.mkdtemp(prefix="afnan-ctl-transport-")
    security = SecurityCenter()
    task_manager = TaskManager(str(Path(tmp) / "tasks.json"))
    goal_manager = GoalManager(str(Path(tmp) / "goals.json"))
    activity = ActivityCenter()
    approvals = ApprovalCenter(center=activity)
    plane = ControlPlane(
        security_center=security,
        task_manager=task_manager,
        goal_manager=goal_manager,
        activity_center=activity,
        approval_center=approvals,
        device_path=str(Path(tmp) / "devices.json"),
        **kwargs,
    )
    return plane, tmp


def pair_device(plane, device_id="phone-1", **meta):
    pid, code = plane.pair_request(
        {"device_id": device_id, **meta}
    )
    plane.pair_approve(pid, approved_by="owner")
    result = plane.pair_redeem(
        {"pairing_id": pid, "code": code}
    )
    return result


def grant(plane, session_id, device_id, *capabilities):
    actor = Actor(
        kind=ActorKind.USER,
        actor_id=f"remote:{device_id}:{session_id}",
    )
    for capability in capabilities:
        plane.security.permissions.grant(actor, capability)
    session = plane.sessions.get(session_id)
    session.capabilities = tuple(
        set(session.capabilities) | set(capabilities)
    )


def cmd(command_type, payload=None, **kwargs):
    return {
        "command_type": command_type.value
        if isinstance(command_type, CommandType)
        else command_type,
        "payload": payload or {},
        **kwargs,
    }


# ---------------------------------------------------------------------------
# Raw WebSocket client framing (client side)
# ---------------------------------------------------------------------------


def _read_exact(sock: socket.socket, n: int) -> bytes | None:
    data = b""
    try:
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
    except (OSError, TimeoutError):
        return None
    return data


def _masked_frame(
    payload: bytes, opcode: int = _WSFrame.OPCODE_TEXT
) -> bytes:
    """Build a client->server frame (RFC 6455: clients MUST mask)."""
    mask = os.urandom(4)
    masked = bytes(
        b ^ mask[i % 4] for i, b in enumerate(payload)
    )
    n = len(payload)
    if n < 126:
        head = bytes([0x80 | opcode, 0x80 | n])
    elif n < 65536:
        head = bytes([0x80 | opcode, 0x80 | 126]) + struct.pack(
            ">H", n
        )
    else:
        head = bytes([0x80 | opcode, 0x80 | 127]) + struct.pack(
            ">Q", n
        )
    return head + mask + masked


def _ws_client_read(
    sock: socket.socket, timeout: float = 10.0
) -> tuple[int, bytes] | None:
    """Decode one server->client frame (servers never mask).

    ``_WSFrame.decode`` cannot be used here: it enforces the
    client-side masking rule, which does not apply to frames the
    server sends us.
    """
    sock.settimeout(timeout)
    head = _read_exact(sock, 2)
    if head is None:
        return None
    b1, b2 = head[0], head[1]
    opcode = b1 & 0x0F
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F
    if length == 126:
        ext = _read_exact(sock, 2)
        if ext is None:
            return None
        length = struct.unpack(">H", ext)[0]
    elif length == 127:
        ext = _read_exact(sock, 8)
        if ext is None:
            return None
        length = struct.unpack(">Q", ext)[0]
    mask = _read_exact(sock, 4) if masked else b""
    payload = _read_exact(sock, length) if length else b""
    if payload is None:
        return None
    if masked and mask:
        payload = bytes(
            b ^ mask[i % 4] for i, b in enumerate(payload)
        )
    return opcode, payload


def _close_code(payload: bytes) -> int | None:
    if len(payload) < 2:
        return None
    return struct.unpack(">H", payload[:2])[0]


# ---------------------------------------------------------------------------
# Base: per-test plane + transport on loopback
# ---------------------------------------------------------------------------


class _TransportTestBase(unittest.TestCase):
    PORT = 18766

    def setUp(self):
        super().setUp()
        self._sockets: list[socket.socket] = []
        self.plane, self._tmp = make_plane()
        self.transport = ControlTransport(
            self.plane,
            host="127.0.0.1",
            port=self.PORT,
            allow_insecure=True,
        )
        self.transport.start()

    def tearDown(self):
        for s in self._sockets:
            try:
                s.close()
            except OSError:
                pass
        self._sockets = []
        try:
            self.transport.stop()
        except Exception:
            pass
        try:
            self.plane.stop()
        except Exception:
            pass
        super().tearDown()

    # -- HTTP -----------------------------------------------------------

    def _http(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        raw: bytes | None = None,
        token: str | None = None,
    ) -> tuple[int, bytes]:
        data = (
            raw
            if raw is not None
            else (
                json.dumps(body).encode("utf-8")
                if body is not None
                else None
            )
        )
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.PORT}{path}",
            data=data,
            method=method,
        )
        if body is not None:
            req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("Authorization", "Bearer " + token)
        try:
            with urllib.request.urlopen(
                req, timeout=15
            ) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    # -- WebSocket ------------------------------------------------------

    def _ws_handshake(
        self, token: str = "", use_auth_header: bool = False
    ) -> tuple[socket.socket, str]:
        sock = socket.create_connection(
            ("127.0.0.1", self.PORT), timeout=10
        )
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        target = "/v1/stream"
        if token and not use_auth_header:
            target += "?token=" + token
        lines = [
            f"GET {target} HTTP/1.1",
            f"Host: 127.0.0.1:{self.PORT}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            "Sec-WebSocket-Key: " + key,
            "Sec-WebSocket-Version: 13",
        ]
        if use_auth_header and token:
            lines.append("Authorization: Bearer " + token)
        lines.extend(["", ""])
        sock.sendall("\r\n".join(lines).encode("ascii"))
        data = b""
        sock.settimeout(10)
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if len(data) > 65536:
                break
        status_line = data.decode("utf-8", "replace").split(
            "\r\n", 1
        )[0]
        self._sockets.append(sock)
        return sock, status_line


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class HttpTransportTests(_TransportTestBase):
    PORT = 18766

    def setUp(self):
        super().setUp()
        self.dev = pair_device(self.plane, device_id="http-phone")
        self.token = self.dev["token"]

    def test_health_returns_200_ok(self):
        status, body = self._http("GET", "/v1/health")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertTrue(data["ok"])
        self.assertEqual(data["protocol_version"], "v1")

    def test_commands_without_token_returns_401(self):
        # Missing credentials on the command path -> 401.
        status, _ = self._http(
            "POST",
            "/v1/commands",
            body=cmd(CommandType.HEALTH_CHECK),
        )
        self.assertEqual(status, 401)

    def test_commands_with_bad_token_returns_401(self):
        # Forged/malformed bearer token on the command path -> 401.
        status, _ = self._http(
            "POST",
            "/v1/commands",
            body=cmd(CommandType.HEALTH_CHECK),
            token="afn_invalid_token",
        )
        self.assertEqual(status, 401)

    def test_forbidden_command_denied_in_result(self):
        # Authenticated but lacking remote.tasks.control.
        status, body = self._http(
            "POST",
            "/v1/commands",
            body=cmd(
                CommandType.START_TASK, {"goal_text": "x"}
            ),
            token=self.token,
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "denied")

    def test_malformed_json_returns_400(self):
        status, _ = self._http(
            "POST",
            "/v1/commands",
            raw=b"{not valid json",
            token=self.token,
        )
        self.assertEqual(status, 400)

    def test_oversized_body_returns_400(self):
        status, _ = self._http(
            "POST",
            "/v1/commands",
            raw=b"x" * 1048577,
            token=self.token,
        )
        self.assertEqual(status, 400)

    def test_unknown_path_returns_404(self):
        status, _ = self._http("GET", "/v1/does-not-exist")
        self.assertEqual(status, 404)

    def test_events_without_token_returns_401(self):
        status, _ = self._http("GET", "/v1/events")
        self.assertEqual(status, 401)

    def test_pair_request_reachable_without_auth(self):
        status, body = self._http(
            "POST",
            "/v1/pair/request",
            body={
                "device_id": "http-new",
                "device_name": "New device",
            },
        )
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertIn("pairing_id", data)
        self.assertIn("code", data)
        self.assertEqual(len(data["code"]), 6)
        self.assertTrue(data["code"].isdigit())

    def test_concurrent_health_requests(self):
        results: list[bool] = []
        lock = threading.Lock()

        def hit():
            try:
                s, b = self._http("GET", "/v1/health")
                ok = s == 200 and json.loads(b)["ok"] is True
            except Exception:
                ok = False
            with lock:
                results.append(ok)

        threads = [
            threading.Thread(target=hit) for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(len(results), 10)
        self.assertTrue(
            all(results), "some concurrent health checks failed"
        )


# ---------------------------------------------------------------------------
# WebSocket (raw RFC 6455 client over a real socket)
# ---------------------------------------------------------------------------


class WebSocketTests(_TransportTestBase):
    PORT = 18767

    def setUp(self):
        super().setUp()
        self.dev = pair_device(self.plane, device_id="ws-phone")
        self.token = self.dev["token"]
        self.session_id = self.dev["session_id"]

    def test_valid_handshake_with_query_token(self):
        _, status = self._ws_handshake(token=self.token)
        self.assertIn("101", status, status)

    def test_valid_handshake_with_auth_header(self):
        # The Authorization header is the preferred mechanism;
        # ?token= is only a compatibility fallback.
        _, status = self._ws_handshake(
            token=self.token, use_auth_header=True
        )
        self.assertIn("101", status, status)

    def test_invalid_token_gets_no_101(self):
        _, status = self._ws_handshake(token="afn_invalid")
        self.assertNotIn("101", status, status)
        self.assertIn("401", status, status)

    def test_command_over_websocket(self):
        sock, status = self._ws_handshake(token=self.token)
        self.assertIn("101", status)
        sock.sendall(
            _masked_frame(
                json.dumps(cmd(CommandType.HEALTH_CHECK)).encode(
                    "utf-8"
                )
            )
        )
        result = None
        for _ in range(6):
            frame = _ws_client_read(sock, timeout=10)
            self.assertIsNotNone(
                frame, "server sent nothing after command"
            )
            opcode, payload = frame
            self.assertEqual(opcode, _WSFrame.OPCODE_TEXT)
            msg = json.loads(payload.decode("utf-8"))
            if msg.get("type") == "command_result":
                result = msg["result"]
                break
        self.assertIsNotNone(
            result, "no command_result frame received"
        )
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["result"]["ok"])

    def test_ping_gets_pong(self):
        # The server does not originate pings; it must answer ours.
        sock, status = self._ws_handshake(token=self.token)
        self.assertIn("101", status)
        sock.sendall(
            _masked_frame(b"pingdata", opcode=_WSFrame.OPCODE_PING)
        )
        frame = _ws_client_read(sock, timeout=10)
        self.assertIsNotNone(frame, "no pong from server")
        opcode, payload = frame
        self.assertEqual(opcode, _WSFrame.OPCODE_PONG)
        self.assertEqual(payload, b"pingdata")

    def test_unmasked_client_frame_closes_with_1002(self):
        # RFC 6455 s5.1: the server MUST close on unmasked frames.
        sock, status = self._ws_handshake(token=self.token)
        self.assertIn("101", status)
        sock.sendall(b"\x81\x05hello")  # FIN+text, NOT masked
        frame = _ws_client_read(sock, timeout=10)
        self.assertIsNotNone(
            frame, "server ignored an unmasked client frame"
        )
        opcode, payload = frame
        self.assertEqual(opcode, _WSFrame.OPCODE_CLOSE)
        self.assertEqual(_close_code(payload), 1002)

    def test_oversized_frame_closes_with_1009(self):
        # Declared length above the 1 MiB cap must be rejected
        # before any allocation happens.
        sock, status = self._ws_handshake(token=self.token)
        self.assertIn("101", status)
        sock.sendall(
            bytes([0x82, 0xFF])
            + struct.pack(">Q", 2 * 1024 * 1024)
        )
        frame = _ws_client_read(sock, timeout=10)
        self.assertIsNotNone(
            frame, "server ignored an oversized frame"
        )
        opcode, payload = frame
        self.assertEqual(opcode, _WSFrame.OPCODE_CLOSE)
        self.assertEqual(_close_code(payload), 1009)

    def test_close_frame_gets_clean_close(self):
        sock, status = self._ws_handshake(token=self.token)
        self.assertIn("101", status)
        sock.sendall(
            _masked_frame(b"", opcode=_WSFrame.OPCODE_CLOSE)
        )
        frame = _ws_client_read(sock, timeout=10)
        self.assertIsNotNone(frame, "no close echo from server")
        opcode, payload = frame
        self.assertEqual(opcode, _WSFrame.OPCODE_CLOSE)
        self.assertEqual(_close_code(payload), 1000)
        # The socket is closed after the close handshake.
        self.assertIsNone(_ws_client_read(sock, timeout=5))

    def test_revoked_session_gets_disconnected(self):
        sock, status = self._ws_handshake(token=self.token)
        self.assertIn("101", status)
        self.plane.sessions.revoke(self.session_id)
        # Nudge the server loop past its blocking read; the next
        # liveness check must terminate the connection with 1008.
        sock.sendall(
            _masked_frame(b"", opcode=_WSFrame.OPCODE_PING)
        )
        code = None
        for _ in range(4):
            frame = _ws_client_read(sock, timeout=10)
            if frame is None:
                break
            opcode, payload = frame
            if opcode == _WSFrame.OPCODE_CLOSE:
                code = _close_code(payload)
                break
        self.assertEqual(code, 1008)


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------


class SseTests(_TransportTestBase):
    PORT = 18768

    def setUp(self):
        super().setUp()
        self.dev = pair_device(self.plane, device_id="sse-phone")
        self.token = self.dev["token"]
        self.session_id = self.dev["session_id"]

    def _open_stream(self):
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.PORT, timeout=15
        )
        conn.putrequest("GET", "/v1/events")
        conn.putheader("Authorization", "Bearer " + self.token)
        conn.endheaders()
        resp = conn.getresponse()
        # The 15s connection timeout above also bounds reads; the
        # reader loop below enforces its own shorter deadline.
        return conn, resp

    def _read_until(self, resp, predicate, deadline_s=10.0):
        deadline = time.time() + deadline_s
        while time.time() < deadline:
            try:
                line = resp.readline()
            except (OSError, TimeoutError, socket.timeout):
                continue
            if not line:
                return None
            if predicate(line):
                return line
        return None

    def test_stream_opens_and_yields_published_event(self):
        conn, resp = self._open_stream()
        try:
            self.assertEqual(resp.status, 200)
            self.assertIn(
                "text/event-stream",
                resp.getheader("Content-Type") or "",
            )
            marker = "sse-marker-%d" % int(time.time() * 1000)
            published = self.plane.events.publish(
                "agent.status",
                {"marker": marker},
                device_id="sse-phone",
            )
            self.assertIsNotNone(published)
            line = self._read_until(
                resp,
                lambda ln: ln.startswith(b"data:")
                and marker.encode() in ln,
            )
            self.assertIsNotNone(
                line,
                "published event never arrived on the SSE stream",
            )
            payload = json.loads(
                line[len(b"data:"):].decode("utf-8")
            )
            self.assertEqual(payload["category"], "agent.status")
            self.assertEqual(payload["data"]["marker"], marker)
            # No secrets cross the boundary in the event payload.
            self.assertNotIn("token", json.dumps(payload).lower())
        finally:
            conn.close()

    def test_revoked_session_stops_stream(self):
        conn, resp = self._open_stream()
        try:
            self.assertEqual(resp.status, 200)
            self.plane.sessions.revoke(self.session_id)
            line = self._read_until(
                resp, lambda ln: ln.startswith(b":")
            )
            self.assertIsNotNone(
                line,
                "SSE stream did not terminate after revocation",
            )
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Security (real logic, real transport where it matters)
# ---------------------------------------------------------------------------


class TransportSecurityTests(_TransportTestBase):
    PORT = 18769

    def setUp(self):
        super().setUp()
        self.dev = pair_device(self.plane, device_id="sec-phone")
        self.token = self.dev["token"]
        self.device_id = self.dev["device_id"]
        self.session_id = self.dev["session_id"]

    def test_revoked_device_token_rejected_at_plane(self):
        self.plane.devices.revoke(self.device_id)
        with self.assertRaises(AuthenticationError):
            self.plane.execute_command(
                self.token, cmd(CommandType.HEALTH_CHECK)
            )
        device = self.plane.devices.get(self.device_id)
        self.assertEqual(device.trust, DeviceTrust.REVOKED)

    def test_http_command_auth_failures_return_401(self):
        # Every rejected authentication on the HTTP command path
        # must surface as 401 (never 500, never a silent success).
        cases = []
        # 1. missing token
        cases.append(
            self._http(
                "POST",
                "/v1/commands",
                body=cmd(CommandType.HEALTH_CHECK),
            )
        )
        # 2. forged token
        cases.append(
            self._http(
                "POST",
                "/v1/commands",
                body=cmd(CommandType.HEALTH_CHECK),
                token="afn_invalid",
            )
        )
        # 3. revoked device
        revoked = pair_device(self.plane, device_id="sec-revoked")
        self.plane.devices.revoke("sec-revoked")
        cases.append(
            self._http(
                "POST",
                "/v1/commands",
                body=cmd(CommandType.HEALTH_CHECK),
                token=revoked["token"],
            )
        )
        # 4. replayed (rotated-out) token
        rotated = pair_device(self.plane, device_id="sec-rotated")
        new_token = self.plane.rotate_token(
            rotated["token"], rotated["token"]
        )["token"]
        self.assertNotEqual(new_token, rotated["token"])
        cases.append(
            self._http(
                "POST",
                "/v1/commands",
                body=cmd(CommandType.HEALTH_CHECK),
                token=rotated["token"],
            )
        )
        for status, _ in cases:
            self.assertEqual(
                status,
                401,
                "authentication failure must be HTTP 401",
            )

    def test_token_replay_after_rotation_rejected(self):
        new_token = self.plane.rotate_token(
            self.token, self.token
        )["token"]
        self.assertNotEqual(new_token, self.token)
        # The old token is dead immediately.
        with self.assertRaises(AuthenticationError):
            self.plane.authenticate(self.token)
        # The new token works for the same session.
        session = self.plane.authenticate(new_token)
        self.assertEqual(session.session_id, self.session_id)

    def test_pairing_brute_force_locks_request(self):
        status, body = self._http(
            "POST",
            "/v1/pair/request",
            body={
                "device_id": "sec-brute",
                "device_name": "Brute",
            },
        )
        self.assertEqual(status, 200)
        info = json.loads(body)
        pid, real_code = info["pairing_id"], info["code"]
        self.plane.pair_approve(pid, approved_by="owner")
        for _ in range(6):
            status, body = self._http(
                "POST",
                "/v1/pair/redeem",
                body={"pairing_id": pid, "code": "000000"},
            )
            # PairingError currently surfaces as HTTP 500.
            self.assertEqual(status, 500)
            # The real code is never echoed back.
            self.assertNotIn(
                real_code, body.decode("utf-8", "replace")
            )
        # The request is now locked: even the correct code fails.
        with self.assertRaises(PairingError):
            self.plane.pair_redeem(
                {"pairing_id": pid, "code": real_code}
            )


# ---------------------------------------------------------------------------
# Concurrency (no transport; real plane + real managers)
# ---------------------------------------------------------------------------


class ConcurrencyTests(unittest.TestCase):
    def test_two_devices_pause_resume_same_task(self):
        plane, _ = make_plane()
        try:
            a = pair_device(plane, device_id="conc-a")
            b = pair_device(plane, device_id="conc-b")
            grant(
                plane,
                a["session_id"],
                "conc-a",
                "remote.tasks.control",
            )
            grant(
                plane,
                b["session_id"],
                "conc-b",
                "remote.tasks.control",
            )
            created = plane.execute_command(
                a["token"],
                cmd(
                    CommandType.START_TASK,
                    {"goal_text": "concurrency probe"},
                ),
            )
            self.assertEqual(created["status"], "completed")
            tid = created["result"]["task"]["task_id"]

            rng = random.Random(20261006)
            ops = [
                rng.choice(
                    [CommandType.PAUSE_TASK, CommandType.RESUME_TASK]
                )
                for _ in range(20)
            ]
            results: list[dict] = []
            lock = threading.Lock()

            def worker(i: int):
                tok = a["token"] if i % 2 == 0 else b["token"]
                try:
                    res = plane.execute_command(
                        tok,
                        cmd(
                            ops[i],
                            {"task_id": tid},
                            command_id="conc-%d" % i,
                            idempotency_key="conc-key-%d" % i,
                        ),
                    )
                except Exception as exc:  # must never escape
                    res = {
                        "status": "raised",
                        "error": "%s: %s"
                        % (type(exc).__name__, exc),
                    }
                with lock:
                    results.append(res)

            threads = [
                threading.Thread(target=worker, args=(i,))
                for i in range(20)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60)
            self.assertEqual(len(results), 20)
            for r in results:
                self.assertIn(
                    r["status"],
                    {"completed", "failed", "denied", "duplicate"},
                    r,
                )
            # TaskManager stays authoritative: the final state is
            # one of the two legal outcomes, never corrupted.
            final = plane.router.tasks.get(tid)
            self.assertIsNotNone(final)
            self.assertIn(final.status, {"pending", "paused"})
            # The remote view agrees with the authoritative manager.
            view = plane.execute_command(
                a["token"],
                cmd(CommandType.GET_TASK, {"task_id": tid}),
            )
            self.assertEqual(view["status"], "completed")
            self.assertEqual(
                view["result"]["task"]["task_id"], tid
            )
            self.assertEqual(
                view["result"]["task"]["status"], final.status
            )
        finally:
            plane.stop()

    def test_revoke_device_during_event_stream(self):
        plane, _ = make_plane()
        sid_a = None
        try:
            a = pair_device(plane, device_id="stream-a")
            b = pair_device(plane, device_id="stream-b")
            sid_a = a["session_id"]
            grant(
                plane,
                b["session_id"],
                "stream-b",
                "remote.devices.manage",
            )
            # Device A has a live event attachment (streaming).
            plane.events.attach(sid_a, from_seq=0)
            # Device B revokes device A through the real command path.
            res = plane.execute_command(
                b["token"],
                cmd(
                    CommandType.REVOKE_DEVICE,
                    {"device_id": "stream-a"},
                ),
            )
            self.assertEqual(res["status"], "completed")
            self.assertEqual(res["result"]["revoked_sessions"], 1)
            self.assertGreaterEqual(
                res["result"]["revoked_tokens"], 1
            )
            # Subsequent commands from A fail authentication.
            with self.assertRaises(AuthenticationError):
                plane.execute_command(
                    a["token"], cmd(CommandType.HEALTH_CHECK)
                )
            session = plane.sessions.get(sid_a)
            self.assertEqual(session.state, SessionState.REVOKED)
        finally:
            try:
                if sid_a is not None:
                    plane.events.detach(sid_a)
            except Exception:
                pass
            plane.stop()


# ---------------------------------------------------------------------------
# Long-horizon (bounded runtime, real state)
# ---------------------------------------------------------------------------


class LongHorizonTests(unittest.TestCase):
    def test_500_events_received_in_order(self):
        plane, _ = make_plane()
        try:
            d = pair_device(plane, device_id="lh-1")
            sid = d["session_id"]
            plane.events.attach(sid, from_seq=0)
            for i in range(500):
                plane.events.publish(
                    "command.completed",
                    {"i": i},
                    device_id="lh-1",
                )
            events = plane.events.poll(sid)
            self.assertEqual(len(events), 500)
            seqs = [e["seq"] for e in events]
            self.assertEqual(len(set(seqs)), 500)
            self.assertEqual(seqs, sorted(seqs))
            self.assertEqual(
                [e["data"]["i"] for e in events],
                list(range(500)),
            )
            plane.events.detach(sid)
        finally:
            plane.stop()

    def test_rapid_connect_disconnect_no_session_leak(self):
        plane, _ = make_plane()
        try:
            baseline_live = len(
                plane.sessions.list(live_only=True)
            )
            threads_before = len(threading.enumerate())
            for i in range(20):
                d = pair_device(
                    plane, device_id="lh-cycle-%d" % i
                )
                plane.sessions.revoke(d["session_id"])
            live = plane.sessions.list(live_only=True)
            self.assertEqual(len(live), baseline_live)
            for s in live:
                self.assertTrue(s.is_live())
            # No thread leak from the churn either.
            self.assertLessEqual(
                len(threading.enumerate()), threads_before + 2
            )
        finally:
            plane.stop()

    def test_token_rotation_chain(self):
        plane, _ = make_plane()
        try:
            d = pair_device(plane, device_id="lh-rot")
            tokens = [d["token"]]
            for _ in range(5):
                tokens.append(
                    plane.rotate_token(tokens[-1], tokens[-1])[
                        "token"
                    ]
                )
            # The latest token works.
            hb = plane.heartbeat(tokens[-1])
            self.assertEqual(hb["session_id"], d["session_id"])
            # Every older token is dead.
            for old in tokens[:-1]:
                with self.assertRaises(
                    AuthenticationError,
                    msg="rotated-out token still accepted",
                ):
                    plane.authenticate(old)
        finally:
            plane.stop()


if __name__ == "__main__":
    unittest.main()

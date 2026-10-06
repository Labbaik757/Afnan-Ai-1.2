"""Control-plane transport.

Stdlib-only HTTP server exposing the versioned remote API, plus
Server-Sent Events and WebSocket streams for real-time events.
No third-party dependencies — matching the repository's stdlib-only
convention (see the CDP client in ``browser/chromium_adapter.py``).

Security posture:

- TLS is used whenever a certificate/key pair is configured, with
  TLS 1.2 as the minimum version and a restricted cipher list.
- Plaintext is allowed only with ``allow_insecure=True`` (explicit
  development-only opt-in) and then only on loopback interfaces.
- Binding a non-loopback address without TLS is refused.
- Every request authenticates via ``Authorization: Bearer`` except
  the pairing bootstrap endpoints, which are rate-limited.
- Owner-only endpoints (``/v1/owner/pairings*``) authorize pairing
  approval from the PC itself: they require a loopback client
  address plus a high-entropy owner token minted on first start and
  kept in a mode-0600 file next to the device registry.  The token
  is compared in constant time and is never logged.
- WebSocket authentication prefers the ``Authorization: Bearer``
  header on the upgrade request.  A ``?token=`` query parameter is
  accepted only as a compatibility fallback for clients that cannot
  set headers; it is never logged, never persisted, and never
  included in audit/activity data.

WebSocket policy (RFC 6455):

- Client frames MUST be masked; unmasked frames are a protocol
  error and the connection is closed (1002).
- Fragmented data frames are not supported; FIN=0 is rejected
  with 1002.  Control frames must be unfragmented and <= 125 bytes.
- Reserved opcodes are rejected with 1002.
- Text frames must be valid UTF-8, otherwise 1007.
- Declared payloads above 1 MiB are rejected with 1009 before
  any allocation happens.
- Idle connections are closed after the configured timeout.
- Slow clients whose sends block past the socket timeout are
  disconnected; no unbounded buffering exists anywhere.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import socket
import ssl
import struct
import threading
import time
import collections
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from afnan_ai.control.auth import AuthenticationError
from afnan_ai.control.pairing import PairingError
from afnan_ai.control.plane import ControlPlaneError


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost")


#: Default location of the owner token file, next to the device
#: registry.  The token authorizes owner-only endpoints (pairing
#: approval) and is only ever honored on loopback connections.
DEFAULT_OWNER_TOKEN_PATH = os.path.expanduser(
    "~/.afnan-ai/control/owner_token"
)


def load_or_create_owner_token(path: str | None = None) -> str:
    """Load the owner token, creating it (mode 0600) if missing.

    The owner token is a high-entropy bearer secret minted on first
    server start.  It authorizes the owner-only HTTP endpoints
    (``/v1/owner/...``) used by the ``approve_pairing`` CLI on the
    PC itself.  The file must never be transmitted or logged.
    """
    raw = os.path.expanduser(path or DEFAULT_OWNER_TOKEN_PATH)
    parent = os.path.dirname(raw)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        fd = os.open(
            raw, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
        )
    except FileExistsError:
        with open(raw, "r", encoding="utf-8") as handle:
            token = handle.read().strip()
        if (
            not token
            or len(token) < 32
            or any(ch.isspace() for ch in token)
        ):
            raise TransportError(
                "owner token file is present but invalid; "
                "delete it and restart the server to mint a new one"
            )
        return token
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(secrets.token_urlsafe(32))
    except BaseException:
        try:
            os.unlink(raw)
        except OSError:
            pass
        raise
    with open(raw, "r", encoding="utf-8") as handle:
        return handle.read().strip()


class TransportError(Exception):
    """Transport misconfiguration."""


class WSProtocolError(Exception):
    """RFC 6455 protocol violation; carries the close code to send."""

    def __init__(self, code: int, reason: str) -> None:
        super().__init__(reason)
        self.code = code


class _WSFrame:
    """RFC 6455 framing (server side), production-hardened."""

    OPCODE_CONT = 0x0
    OPCODE_TEXT = 0x1
    OPCODE_BINARY = 0x2
    OPCODE_CLOSE = 0x8
    OPCODE_PING = 0x9
    OPCODE_PONG = 0xA

    #: Maximum accepted payload size (1 MiB).  Declared lengths above
    #: this are rejected before any memory is allocated.
    MAX_PAYLOAD = 1_048_576

    _RESERVED = frozenset((0x3, 0x4, 0x5, 0x6, 0x7, 0xB, 0xC, 0xD, 0xE, 0xF))

    @staticmethod
    def encode_text(payload: str) -> bytes:
        data = payload.encode("utf-8")
        header = bytes([0x81])
        length = len(data)
        if length < 126:
            header += bytes([length])
        elif length < 65536:
            header += bytes([126]) + struct.pack(">H", length)
        else:
            header += bytes([127]) + struct.pack(">Q", length)
        return header + data

    @staticmethod
    def encode_close(code: int = 1000, reason: str = "") -> bytes:
        payload = struct.pack(">H", code) + reason.encode(
            "utf-8"
        )[:123]
        return bytes([0x88, len(payload)]) + payload

    @staticmethod
    def encode_pong(payload: bytes = b"") -> bytes:
        return bytes([0x8A, len(payload)]) + payload

    @staticmethod
    def decode(sock: socket.socket) -> tuple[int, bytes] | None:
        """Read one frame.

        Returns ``(opcode, payload)`` or ``None`` on clean EOF.
        Raises :class:`WSProtocolError` on any protocol violation
        and ``OSError`` on transport failure.
        """
        head = _recv_exact(sock, 2)
        if not head:
            return None
        byte1, byte2 = head[0], head[1]
        fin = bool(byte1 & 0x80)
        opcode = byte1 & 0x0F
        masked = bool(byte2 & 0x80)
        length = byte2 & 0x7F

        if opcode in _WSFrame._RESERVED:
            raise WSProtocolError(1002, "reserved opcode")
        is_control = opcode >= 0x8
        if is_control and not fin:
            raise WSProtocolError(
                1002, "fragmented control frame"
            )
        if is_control and length > 125:
            raise WSProtocolError(
                1002, "control frame too large"
            )

        if length == 126:
            ext = _recv_exact(sock, 2)
            if not ext:
                return None
            length = struct.unpack(">H", ext)[0]
            if length < 126:
                raise WSProtocolError(
                    1002, "non-minimal length encoding"
                )
        elif length == 127:
            ext = _recv_exact(sock, 8)
            if not ext:
                return None
            length = struct.unpack(">Q", ext)[0]
            if length < 65536:
                raise WSProtocolError(
                    1002, "non-minimal length encoding"
                )
        if length > _WSFrame.MAX_PAYLOAD:
            raise WSProtocolError(1009, "message too big")

        # RFC 6455 section 5.1: a server MUST close the connection
        # when it receives an unmasked frame from a client.
        if not masked:
            raise WSProtocolError(
                1002, "client frames must be masked"
            )
        if not fin and not is_control:
            # Documented fragmentation policy: this server does not
            # support fragmented data messages.
            raise WSProtocolError(
                1002, "fragmented messages not supported"
            )

        mask = _recv_exact(sock, 4)
        if not mask:
            return None
        payload = _recv_exact(sock, length) if length else b""
        if payload is None:
            return None
        payload = bytes(
            b ^ mask[i % 4] for i, b in enumerate(payload)
        )
        if opcode == _WSFrame.OPCODE_TEXT:
            try:
                payload.decode("utf-8")
            except UnicodeDecodeError:
                raise WSProtocolError(
                    1007, "text frame is not valid UTF-8"
                )
        return opcode, payload


def _recv_exact(sock: socket.socket, count: int) -> bytes | None:
    data = b""
    try:
        while len(data) < count:
            chunk = sock.recv(count - len(data))
            if not chunk:
                return None
            data += chunk
    except (OSError, TimeoutError):
        return None
    return data


def _send_close(sock: socket.socket, code: int, reason: str = "") -> None:
    try:
        sock.sendall(_WSFrame.encode_close(code, reason))
    except OSError:
        pass


class _SlidingWindowLimiter:
    """Tiny per-key sliding-window rate limiter (in-memory)."""

    def __init__(self, max_hits: int, window_s: float) -> None:
        self._max = max_hits
        self._window = window_s
        self._hits: dict[str, collections.deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            dq = self._hits.setdefault(key, collections.deque())
            cutoff = now - self._window
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if len(dq) >= self._max:
                return False
            dq.append(now)
            return True


class ControlHTTPHandler(BaseHTTPRequestHandler):
    """Routes HTTP requests into the control plane."""

    server_version = "AfnanControl/1.0"
    protocol_version = "HTTP/1.1"

    # -- helpers ------------------------------------------------------------

    def log_message(self, *args):  # quiet; audit covers logging
        pass

    def _plane(self):
        return self.server.plane  # type: ignore[attr-defined]

    def _transport(self):
        return self.server.transport  # type: ignore[attr-defined]

    def _request_id(self) -> str:
        rid = self.headers.get("X-Request-Id", "").strip()
        if not rid or len(rid) > 64:
            rid = secrets.token_hex(8)
        return rid

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        if length > 1_048_576:
            # Drain a bounded prefix so the client receives the
            # 400 response instead of a connection reset.
            self._drain(min(length, 2_097_152))
            raise TransportError("request body too large")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            raise TransportError("malformed JSON body")
        if not isinstance(data, dict):
            raise TransportError("JSON body must be an object")
        return data

    def _drain(self, count: int) -> None:
        """Read and discard up to ``count`` request-body bytes."""
        try:
            remaining = count
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
        except OSError:
            pass

    def _cors_headers(self) -> dict[str, str]:
        transport = self._transport()
        origins = transport.cors_origins
        origin = self.headers.get("Origin", "")
        headers: dict[str, str] = {}
        # Deny-by-default: no CORS headers unless the origin is
        # explicitly allow-listed.
        if origins and origin and origin in origins:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Vary"] = "Origin"
            headers["Access-Control-Allow-Methods"] = (
                "GET, POST, OPTIONS"
            )
            headers["Access-Control-Allow-Headers"] = (
                "Authorization, Content-Type, X-Request-Id"
            )
            headers["Access-Control-Max-Age"] = "600"
        return headers

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        *,
        request_id: str = "",
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if request_id:
            self.send_header("X-Request-Id", request_id)
        for k, v in self._cors_headers().items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass

    def _bearer(self) -> str:
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
            # Reject obviously malformed tokens early.
            if token and len(token) <= 512 and " " not in token:
                return token
        return ""

    def _require_owner(self) -> None:
        """Authorize an owner-only endpoint.

        The owner approves pairings on the PC itself, so these
        endpoints additionally require a loopback client address —
        even over TLS from another host they are refused.  The
        owner token itself is compared in constant time.
        """
        if not _is_loopback(self._client_ip()):
            raise PermissionError(
                "owner endpoints are only available on loopback"
            )
        presented = self._bearer()
        transport = self._transport()
        if not presented or not hmac.compare_digest(
            presented, transport._owner_token
        ):
            raise AuthenticationError("owner authentication failed")

    @staticmethod
    def _owner_label(body: dict) -> str:
        """Audit label for who approved; strictly bounded."""
        label = body.get("approved_by", "owner")
        if not isinstance(label, str) or not label.strip():
            return "owner"
        return label.strip()[:64]

    def _client_ip(self) -> str:
        try:
            return self.client_address[0]
        except Exception:
            return "unknown"

    def _error(
        self, status: int, message: str, request_id: str
    ) -> None:
        self._send_json(
            status,
            {"error": message, "request_id": request_id},
            request_id=request_id,
        )

    # -- routing --------------------------------------------------------------

    def do_GET(self):  # noqa: N802
        request_id = self._request_id()
        if len(self.path) > 4096:
            self._error(414, "request URI too long", request_id)
            return
        parsed = urlparse(self.path)
        plane = self._plane()
        try:
            if parsed.path == "/v1/health":
                self._send_json(
                    200, plane.health(), request_id=request_id
                )
            elif parsed.path == "/v1/events":
                self._handle_sse(request_id)
            elif parsed.path == "/v1/stream":
                self._handle_websocket(request_id)
            elif parsed.path == "/v1/owner/pairings":
                self._require_owner()
                now = time.time()
                self._send_json(
                    200,
                    {
                        "pairings": [
                            {
                                "pairing_id": r.pairing_id,
                                "device_id": (
                                    r.requesting_device_id
                                ),
                                "device_name": (
                                    r.requesting_device_name
                                ),
                                "platform": r.platform,
                                "expires_in_s": max(
                                    0,
                                    int(r.expires_at - now),
                                ),
                            }
                            for r in plane.pairing.list_pending()
                        ]
                    },
                    request_id=request_id,
                )
            else:
                self._error(404, "not found", request_id)
        except TransportError as e:
            self._error(400, str(e), request_id)
        except AuthenticationError as e:
            self._error(401, str(e), request_id)
        except (PairingError, ControlPlaneError) as e:
            self._error(400, str(e), request_id)
        except PermissionError as e:
            self._error(403, str(e), request_id)
        except KeyError:
            self._error(404, "not found", request_id)
        except Exception:
            # Never leak tracebacks or internal detail.
            self._error(500, "internal error", request_id)

    def do_POST(self):  # noqa: N802
        request_id = self._request_id()
        if len(self.path) > 4096:
            self._error(414, "request URI too long", request_id)
            return
        parsed = urlparse(self.path)
        plane = self._plane()
        transport = self._transport()
        try:
            if parsed.path in (
                "/v1/pair/request",
                "/v1/pair/redeem",
                "/v1/owner/pairings/approve",
                "/v1/owner/pairings/deny",
            ):
                if not transport.pair_limiter.allow(
                    f"pair:{self._client_ip()}"
                ):
                    plane.metrics.record_rate_limit_hit()
                    self._error(
                        429,
                        "too many pairing attempts",
                        request_id,
                    )
                    return
            elif parsed.path == "/v1/commands":
                if not transport.command_limiter.allow(
                    f"cmd:{self._client_ip()}"
                ):
                    plane.metrics.record_rate_limit_hit()
                    self._error(
                        429, "command rate exceeded", request_id
                    )
                    return
            body = self._read_json()
            if parsed.path == "/v1/pair/request":
                pairing_id, code = plane.pair_request(body)
                self._send_json(
                    200,
                    {"pairing_id": pairing_id, "code": code},
                    request_id=request_id,
                )
            elif parsed.path == "/v1/pair/redeem":
                result = plane.pair_redeem(body)
                self._send_json(
                    200, result, request_id=request_id
                )
            elif parsed.path == "/v1/owner/pairings/approve":
                self._require_owner()
                pairing_id = str(body.get("pairing_id", ""))
                if not pairing_id:
                    raise ControlPlaneError(
                        "pairing_id is required"
                    )
                result = plane.pair_approve(
                    pairing_id,
                    approved_by=self._owner_label(body),
                )
                self._send_json(
                    200, result, request_id=request_id
                )
            elif parsed.path == "/v1/owner/pairings/deny":
                self._require_owner()
                pairing_id = str(body.get("pairing_id", ""))
                if not pairing_id:
                    raise ControlPlaneError(
                        "pairing_id is required"
                    )
                result = plane.pair_reject(
                    pairing_id,
                    approved_by=self._owner_label(body),
                )
                self._send_json(
                    200, result, request_id=request_id
                )
            elif parsed.path == "/v1/commands":
                result = plane.execute_command(
                    self._bearer(), body
                )
                self._send_json(
                    200, result, request_id=request_id
                )
            elif parsed.path == "/v1/session/heartbeat":
                result = plane.heartbeat(self._bearer())
                self._send_json(
                    200, result, request_id=request_id
                )
            elif parsed.path == "/v1/session/rotate":
                result = plane.rotate_token(
                    self._bearer(), body.get("token", "")
                )
                self._send_json(
                    200, result, request_id=request_id
                )
            else:
                self._error(404, "not found", request_id)
        except TransportError as e:
            self._error(400, str(e), request_id)
        except AuthenticationError as e:
            self._error(401, str(e), request_id)
        except (PairingError, ControlPlaneError) as e:
            # Client-side failures: bad metadata, wrong/expired/
            # replayed pairing codes, unknown commands.
            self._error(400, str(e), request_id)
        except PermissionError as e:
            self._error(403, str(e), request_id)
        except KeyError as e:
            self._error(404, "not found", request_id)
        except Exception:
            self._error(500, "internal error", request_id)

    def do_OPTIONS(self):  # noqa: N802
        # CORS preflight: only answered for allow-listed origins.
        request_id = self._request_id()
        headers = self._cors_headers()
        self.send_response(204 if headers else 403)
        self.send_header("Content-Length", "0")
        if request_id:
            self.send_header("X-Request-Id", request_id)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()

    # Methods we do not implement -> explicit 405 via __getattr__.

    def _method_not_allowed(self) -> None:
        request_id = self._request_id()
        self._error(405, "method not allowed", request_id)

    def __getattr__(self, name: str):  # pragma: no cover
        if name.startswith("do_"):
            return self._method_not_allowed
        raise AttributeError(name)

    # -- SSE --------------------------------------------------------------------

    def _handle_sse(self, request_id: str) -> None:
        plane = self._plane()
        token = self._bearer()
        try:
            session = plane.authenticate(token)
        except Exception as e:
            self._error(401, str(e), request_id)
            return
        # Optional explicit cursor for reconnect semantics.
        from_seq = None
        try:
            query = urlparse(self.path).query
            for part in query.split("&"):
                if part.startswith("cursor="):
                    from_seq = int(part[7:])
                    if from_seq < 0:
                        from_seq = None
                    break
        except (ValueError, TypeError):
            from_seq = None
        self.send_response(200)
        self.send_header(
            "Content-Type", "text/event-stream"
        )
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Request-Id", request_id)
        for k, v in self._cors_headers().items():
            self.send_header(k, v)
        self.end_headers()
        try:
            for chunk in plane.event_stream_sse(
                session, timeout_s=30.0, from_seq=from_seq
            ):
                self.wfile.write(chunk)
                self.wfile.flush()
        except OSError:
            pass

    # -- WebSocket ------------------------------------------------------------

    def _handle_websocket(self, request_id: str) -> None:
        # The request line and headers are already parsed by the
        # base handler; take over the raw socket from here.
        if (
            self.headers.get("Upgrade", "").lower()
            != "websocket"
        ):
            self._error(
                426, "use WebSocket upgrade", request_id
            )
            return
        connection = self.headers.get("Connection", "").lower()
        if "upgrade" not in connection:
            self._error(
                400, "missing Connection: Upgrade", request_id
            )
            return
        if self.headers.get("Sec-WebSocket-Version", "") != "13":
            self.send_response(426)
            self.send_header(
                "Sec-WebSocket-Version", "13"
            )
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        key = self.headers.get("Sec-WebSocket-Key", "")
        if not key:
            self._error(400, "missing WS key", request_id)
            return
        # Prefer the Authorization header.  The query parameter is
        # a compatibility fallback only; the raw URL (which may
        # contain the token) is never logged or persisted.
        token = self._bearer()
        if not token:
            query = urlparse(self.path).query
            for part in query.split("&"):
                if part.startswith("token="):
                    token = part[6:]
                    break
        plane = self._plane()
        transport = self._transport()
        try:
            session = plane.authenticate(token)
        except Exception as e:
            self._error(401, str(e), request_id)
            return
        if not transport.ws_acquire(session.device_id):
            self._error(
                429, "too many WebSocket connections",
                request_id,
            )
            plane.metrics.record_rate_limit_hit()
            return
        accept = base64.b64encode(
            hashlib.sha1(
                (
                    key
                    + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
                ).encode("utf-8")
            ).digest()
        ).decode("utf-8")
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        )
        try:
            self.connection.sendall(response.encode("utf-8"))
            plane.websocket_loop(session, self.connection)
        except OSError:
            pass
        finally:
            transport.ws_release(session.device_id)
            self.close_connection = True


class ControlTransport:
    """HTTP + SSE + WebSocket transport for the control plane."""

    #: Hard caps for WebSocket fan-out.
    MAX_WS_CONNECTIONS = 128
    MAX_WS_PER_DEVICE = 4

    def __init__(
        self,
        plane,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        tls_cert: str | None = None,
        tls_key: str | None = None,
        allow_insecure: bool = False,
        cors_origins: tuple[str, ...] = (),
        ws_idle_timeout_s: float = 120.0,
        owner_token_path: str | None = None,
    ) -> None:
        if not tls_cert or not tls_key:
            if not allow_insecure:
                raise TransportError(
                    "TLS certificate/key required unless "
                    "allow_insecure=True (development only)"
                )
            if not _is_loopback(host):
                raise TransportError(
                    "insecure transport refused on non-loopback "
                    "address"
                )
        elif not _tls_files_ok(tls_cert, tls_key):
            raise TransportError(
                "TLS certificate/key files are not readable"
            )
        self.plane = plane
        self.host = host
        self.port = port
        self._tls_cert = tls_cert
        self._tls_key = tls_key
        self.cors_origins = tuple(cors_origins)
        self.ws_idle_timeout_s = ws_idle_timeout_s
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        # Abuse protection: per-IP sliding windows.
        self.pair_limiter = _SlidingWindowLimiter(10, 60.0)
        self.command_limiter = _SlidingWindowLimiter(240, 60.0)
        # Owner token: bearer secret for the owner-only endpoints
        # (/v1/owner/...), minted once and kept in a 0600 file next
        # to the device registry.  Never logged, never transmitted
        # except by the owner's own CLI on this machine.
        self.owner_token_path = os.path.expanduser(
            owner_token_path or DEFAULT_OWNER_TOKEN_PATH
        )
        self._owner_token = load_or_create_owner_token(
            self.owner_token_path
        )
        # WebSocket connection accounting.
        self._ws_lock = threading.Lock()
        self._ws_total = 0
        self._ws_per_device: dict[str, int] = {}

    # -- WebSocket connection accounting ------------------------------------

    def ws_acquire(self, device_id: str) -> bool:
        with self._ws_lock:
            if self._ws_total >= self.MAX_WS_CONNECTIONS:
                return False
            per = self._ws_per_device.get(device_id, 0)
            if per >= self.MAX_WS_PER_DEVICE:
                return False
            self._ws_total += 1
            self._ws_per_device[device_id] = per + 1
            return True

    def ws_release(self, device_id: str) -> None:
        with self._ws_lock:
            self._ws_total = max(0, self._ws_total - 1)
            per = self._ws_per_device.get(device_id, 0)
            if per <= 1:
                self._ws_per_device.pop(device_id, None)
            else:
                self._ws_per_device[device_id] = per - 1

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._server is not None:
            return
        server = ThreadingHTTPServer(
            (self.host, self.port), ControlHTTPHandler
        )
        server.plane = self.plane  # type: ignore[attr-defined]
        server.transport = self  # type: ignore[attr-defined]
        server.daemon_threads = True
        if self._tls_cert and self._tls_key:
            server.socket = _wrap_tls(
                server.socket, self._tls_cert, self._tls_key
            )
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.2},
            daemon=True,
            name="afnan-control-transport",
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def scheme(self) -> str:
        return "https" if self._tls_cert else "http"

    @property
    def url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"


def _tls_files_ok(cert: str, key: str) -> bool:
    import os

    return os.path.isfile(cert) and os.access(
        cert, os.R_OK
    ) and os.path.isfile(key) and os.access(key, os.R_OK)


def _wrap_tls(
    sock: socket.socket, cert: str, key: str
) -> ssl.SSLSocket:
    """Wrap a socket in a hardened server-side TLS context."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    if hasattr(ssl, "TLSVersion"):
        context.minimum_version = ssl.TLSVersion.TLSv1_2
    # Disable obsolete protocol versions explicitly where the
    # flags exist on this platform.
    for flag_name in ("OP_NO_TLSv1", "OP_NO_TLSv1_1"):
        flag = getattr(ssl, flag_name, None)
        if flag is not None:
            context.options |= flag
    # Prefer strong ciphers; failures fall back to defaults.
    try:
        context.set_ciphers(
            "ECDHE-ECDSA-AES256-GCM-SHA384:"
            "ECDHE-RSA-AES256-GCM-SHA384:"
            "ECDHE-ECDSA-CHACHA20-POLY1305:"
            "ECDHE-RSA-CHACHA20-POLY1305:"
            "ECDHE-ECDSA-AES128-GCM-SHA256:"
            "ECDHE-RSA-AES128-GCM-SHA256"
        )
    except ssl.SSLError:
        pass
    try:
        context.load_cert_chain(cert, key)
    except (OSError, ssl.SSLError) as e:
        raise TransportError(
            "TLS certificate/key could not be loaded"
        ) from e
    # Mismatched cert/key surfaces at wrap/handshake time; fail
    # fast here by validating the pair loads cleanly.
    return context.wrap_socket(sock, server_side=True)

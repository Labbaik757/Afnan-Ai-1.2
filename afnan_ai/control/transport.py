"""Control-plane transport.

Stdlib-only HTTP server exposing the versioned remote API, plus
Server-Sent Events and WebSocket streams for real-time events.
No third-party dependencies — matching the repository's stdlib-only
convention (see the CDP client in ``browser/chromium_adapter.py``).

Security posture:

- TLS is used whenever a certificate/key pair is configured.
- Plaintext is allowed only with ``allow_insecure=True`` (explicit
  development-only opt-in) and then only on loopback interfaces.
- Binding a non-loopback address without TLS is refused.
- Every request authenticates via ``Authorization: Bearer`` except
  the pairing bootstrap endpoints, which are rate-limited.
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import ssl
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost")


class TransportError(Exception):
    """Transport misconfiguration."""


class _WSFrame:
    """Minimal RFC 6455 framing (server side)."""

    OPCODE_TEXT = 0x1
    OPCODE_CLOSE = 0x8
    OPCODE_PING = 0x9
    OPCODE_PONG = 0xA

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
    def decode(sock: socket.socket) -> tuple[int, bytes] | None:
        """Read one frame. Returns (opcode, payload) or None on EOF."""
        head = _recv_exact(sock, 2)
        if not head:
            return None
        byte1, byte2 = head[0], head[1]
        opcode = byte1 & 0x0F
        masked = bool(byte2 & 0x80)
        length = byte2 & 0x7F
        if length == 126:
            ext = _recv_exact(sock, 2)
            if not ext:
                return None
            length = struct.unpack(">H", ext)[0]
        elif length == 127:
            ext = _recv_exact(sock, 8)
            if not ext:
                return None
            length = struct.unpack(">Q", ext)[0]
        mask = _recv_exact(sock, 4) if masked else b""
        payload = _recv_exact(sock, length) if length else b""
        if payload is None:
            return None
        if masked and mask:
            payload = bytes(
                b ^ mask[i % 4] for i, b in enumerate(payload)
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
    except OSError:
        return None
    return data


class ControlHTTPHandler(BaseHTTPRequestHandler):
    """Routes HTTP requests into the control plane."""

    server_version = "AfnanControl/1.0"
    protocol_version = "HTTP/1.1"

    # -- helpers ------------------------------------------------------------

    def log_message(self, *args):  # quiet; audit covers logging
        pass

    def _plane(self):
        return self.server.plane  # type: ignore[attr-defined]

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        if length > 1_048_576:
            raise TransportError("request body too large")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            raise TransportError("malformed JSON body")
        if not isinstance(data, dict):
            raise TransportError("JSON body must be an object")
        return data

    def _send_json(
        self, status: int, payload: dict[str, Any]
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass

    def _bearer(self) -> str:
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip()
        return ""

    # -- routing --------------------------------------------------------------

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        plane = self._plane()
        try:
            if parsed.path == "/v1/health":
                self._send_json(200, plane.health())
            elif parsed.path == "/v1/events":
                self._handle_sse()
            elif parsed.path == "/v1/stream":
                self._handle_websocket()
            else:
                self._send_json(404, {"error": "not found"})
        except TransportError as e:
            self._send_json(400, {"error": str(e)})
        except Exception:
            self._send_json(500, {"error": "internal error"})

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        plane = self._plane()
        try:
            body = self._read_json()
            if parsed.path == "/v1/pair/request":
                pairing_id, code = plane.pair_request(body)
                self._send_json(
                    200,
                    {"pairing_id": pairing_id, "code": code},
                )
            elif parsed.path == "/v1/pair/redeem":
                result = plane.pair_redeem(body)
                self._send_json(200, result)
            elif parsed.path == "/v1/commands":
                result = plane.execute_command(
                    self._bearer(), body
                )
                self._send_json(200, result)
            elif parsed.path == "/v1/session/heartbeat":
                result = plane.heartbeat(self._bearer())
                self._send_json(200, result)
            elif parsed.path == "/v1/session/rotate":
                result = plane.rotate_token(
                    self._bearer(), body.get("token", "")
                )
                self._send_json(200, result)
            else:
                self._send_json(404, {"error": "not found"})
        except TransportError as e:
            self._send_json(400, {"error": str(e)})
        except PermissionError as e:
            self._send_json(403, {"error": str(e)})
        except KeyError as e:
            self._send_json(404, {"error": str(e)})
        except Exception:
            self._send_json(500, {"error": "internal error"})

    # -- SSE --------------------------------------------------------------------

    def _handle_sse(self) -> None:
        plane = self._plane()
        token = self._bearer()
        try:
            session = plane.authenticate(token)
        except Exception as e:
            self._send_json(401, {"error": str(e)})
            return
        self.send_response(200)
        self.send_header(
            "Content-Type", "text/event-stream"
        )
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for chunk in plane.event_stream_sse(
                session, timeout_s=30.0
            ):
                self.wfile.write(chunk)
                self.wfile.flush()
        except OSError:
            pass

    # -- WebSocket ------------------------------------------------------------

    def _handle_websocket(self) -> None:
        # The request line and headers are already parsed by the
        # base handler; take over the raw socket from here.
        if (
            self.headers.get("Upgrade", "").lower()
            != "websocket"
        ):
            self._send_json(
                426, {"error": "use WebSocket upgrade"}
            )
            return
        key = self.headers.get("Sec-WebSocket-Key", "")
        if not key:
            self._send_json(400, {"error": "missing WS key"})
            return
        token = ""
        query = urlparse(self.path).query
        for part in query.split("&"):
            if part.startswith("token="):
                token = part[6:]
        plane = self._plane()
        try:
            session = plane.authenticate(token)
        except Exception as e:
            self._send_json(401, {"error": str(e)})
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
            self.close_connection = True


class ControlTransport:
    """HTTP + SSE + WebSocket transport for the control plane."""

    def __init__(
        self,
        plane,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        tls_cert: str | None = None,
        tls_key: str | None = None,
        allow_insecure: bool = False,
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
        self.plane = plane
        self.host = host
        self.port = port
        self._tls_cert = tls_cert
        self._tls_key = tls_key
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            return
        server = ThreadingHTTPServer(
            (self.host, self.port), ControlHTTPHandler
        )
        server.plane = self.plane  # type: ignore[attr-defined]
        server.daemon_threads = True
        if self._tls_cert and self._tls_key:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(
                self._tls_cert, self._tls_key
            )
            server.socket = context.wrap_socket(
                server.socket, server_side=True
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

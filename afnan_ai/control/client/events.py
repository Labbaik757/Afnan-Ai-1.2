"""Event streaming client: SSE and WebSocket.

``EventClient`` consumes the two real-time transports the server
exposes:

- ``GET /v1/events`` (Server-Sent Events, bearer token in the
  ``Authorization`` header)
- ``GET /v1/stream?token=...`` (WebSocket upgrade; token in the query
  string, which is the documented compatibility behavior of the
  server)

WebSocket framing reuses :class:`_WSFrame` from
:mod:`afnan_ai.control.transport`; client-to-server frames are masked
per RFC 6455.  Ping/pong and the close handshake are handled, and
``disconnect()`` tears the connection down cleanly.
"""

from __future__ import annotations

import base64
import json
import os
import random
import socket
import ssl
import struct
import threading
from typing import Any, Callable
from urllib.parse import urlparse

from afnan_ai.control.client.base import (
    ControlClient,
    redact_token_text,
)
from afnan_ai.control.client.errors import (
    AuthenticationError,
    ConnectionError,
    ProtocolError,
)
from afnan_ai.control.transport import _WSFrame

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _encode_masked(opcode: int, payload: bytes) -> bytes:
    """Encode one client-to-server frame with a random mask."""
    mask = struct.pack(">I", random.getrandbits(32))
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    header = bytes([0x80 | (opcode & 0x0F)])
    length = len(masked)
    if length < 126:
        header += bytes([0x80 | length])
    elif length < 65536:
        header += bytes([0x80 | 126]) + struct.pack(">H", length)
    else:
        header += bytes([0x80 | 127]) + struct.pack(">Q", length)
    return header + mask + masked


def _encode_close() -> bytes:
    return _encode_masked(_WSFrame.OPCODE_CLOSE, b"")


def _recv_exact(
    sock: "socket.socket", count: int
) -> bytes | None:
    """Read exactly ``count`` bytes (works on TLS sockets too)."""
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


def _decode_server_frame(
    sock: "socket.socket",
) -> tuple[int, bytes] | None:
    """Decode one server -> client frame (client side).

    Per RFC 6455 section 5.1 a server MUST NOT mask its frames;
    a masked frame from the server is a protocol error.  Payloads
    above 1 MiB are rejected before allocation, text must be
    valid UTF-8, and fragmented data messages are not supported.
    Returns ``(opcode, payload)`` or ``None`` on clean EOF.
    """
    head = _recv_exact(sock, 2)
    if not head:
        return None
    byte1, byte2 = head[0], head[1]
    fin = bool(byte1 & 0x80)
    opcode = byte1 & 0x0F
    masked = bool(byte2 & 0x80)
    length = byte2 & 0x7F
    if masked:
        raise ProtocolError(
            "server sent a masked frame", code="ws_masked_server"
        )
    if opcode in (0x3, 0x4, 0x5, 0x6, 0x7, 0xB, 0xC, 0xD, 0xE, 0xF):
        raise ProtocolError(
            "reserved opcode", code="ws_bad_opcode"
        )
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
    if length > _WSFrame.MAX_PAYLOAD:
        raise ProtocolError(
            "frame too big", code="ws_frame_too_big"
        )
    if not fin and opcode < 0x8:
        raise ProtocolError(
            "fragmented messages not supported",
            code="ws_fragmented",
        )
    payload = _recv_exact(sock, length) if length else b""
    if payload is None:
        return None
    if opcode == _WSFrame.OPCODE_TEXT:
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError:
            raise ProtocolError(
                "invalid UTF-8", code="ws_bad_utf8"
            )
    return opcode, payload


class EventClient:
    """Subscribe to control-plane events over SSE or WebSocket."""

    def __init__(self, client: ControlClient) -> None:
        self._client = client
        self.last_cursor: int = 0
        self._sse_sock: Any = None
        self._sse_lock = threading.Lock()

    # -- SSE ---------------------------------------------------------------

    def stream_sse(
        self,
        token: str,
        *,
        cursor: int = 0,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        timeout_s: float = 30.0,
    ):
        """Yield event dicts from the SSE stream.

        The server replays missed events from the session's stored
        cursor and then streams live; ``cursor`` is forwarded as a
        query hint for forward compatibility and is tracked locally
        on :attr:`last_cursor` as events arrive.  The generator ends
        when ``timeout_s`` elapses or :meth:`disconnect` is called.
        """
        import urllib.request

        url = f"{self._client.base_url}/v1/events?cursor={int(cursor)}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "text/event-stream",
                "User-Agent": "AfnanControlClient/1.0",
            },
        )
        try:
            resp = urllib.request.urlopen(
                req,
                timeout=self._client.timeout_s,
                context=self._client._ssl_context,
            )
        except Exception as e:
            status = getattr(e, "code", None)
            if status == 401:
                raise AuthenticationError(
                    "SSE authentication failed", status=401
                )
            raise ConnectionError(
                "SSE connection failed: "
                + redact_token_text(str(e)),
                code="sse_connect_failed",
            )
        with self._sse_lock:
            self._sse_sock = resp
        import time as _time

        deadline = _time.time() + timeout_s
        data_lines: list[str] = []
        try:
            while _time.time() < deadline:
                with self._sse_lock:
                    if self._sse_sock is None:
                        break
                try:
                    line = resp.readline()
                except OSError:
                    break
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip(
                    "\r\n"
                )
                if text == "":
                    if data_lines:
                        raw = "\n".join(data_lines)
                        data_lines = []
                        event = self._parse_sse_data(raw)
                        if event is not None:
                            self._track_cursor(event)
                            if on_event is not None:
                                on_event(event)
                            yield event
                    continue
                if text.startswith(":"):
                    continue  # keep-alive comment
                if text.startswith("data:"):
                    data_lines.append(text[5:].lstrip(" "))
        finally:
            with self._sse_lock:
                self._sse_sock = None
            try:
                resp.close()
            except Exception:
                pass

    @staticmethod
    def _parse_sse_data(raw: str) -> dict[str, Any] | None:
        try:
            event = json.loads(raw)
        except ValueError:
            return None
        return event if isinstance(event, dict) else None

    def _track_cursor(self, event: dict[str, Any]) -> None:
        seq = event.get("seq")
        if isinstance(seq, int) and seq > self.last_cursor:
            self.last_cursor = seq

    # -- WebSocket -----------------------------------------------------------

    def connect_websocket(
        self,
        token: str,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        on_command_result: Callable[[dict[str, Any]], None]
        | None = None,
        on_error: Callable[[dict[str, Any]], None] | None = None,
        timeout_s: float = 30.0,
    ) -> "WebSocketConnection":
        """Open the WebSocket event stream and return the connection.

        The receive loop runs on a daemon thread and dispatches
        ``{"type": "event"}`` frames to ``on_event`` and
        ``{"type": "command_result"}`` frames to
        ``on_command_result``.  Use ``send_command`` to issue
        commands over the socket and ``disconnect`` to close it.
        """
        conn = WebSocketConnection(
            self._client,
            token,
            on_event=on_event,
            on_command_result=on_command_result,
            on_error=on_error,
            timeout_s=timeout_s,
        )
        conn.connect()
        return conn

    # -- shared ---------------------------------------------------------------

    def disconnect(self) -> None:
        """Close an in-progress SSE stream."""
        with self._sse_lock:
            sock = self._sse_sock
            self._sse_sock = None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass


class WebSocketConnection:
    """One authenticated WebSocket session to the control plane."""

    def __init__(
        self,
        client: ControlClient,
        token: str,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        on_command_result: Callable[[dict[str, Any]], None]
        | None = None,
        on_error: Callable[[dict[str, Any]], None] | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._client = client
        self._token = token
        self._on_event = on_event
        self._on_command_result = on_command_result
        self._on_error = on_error
        self._timeout_s = timeout_s
        self._sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False

    # -- handshake ---------------------------------------------------------

    def connect(self) -> None:
        """Perform the TCP/TLS connect and the WebSocket upgrade."""
        parsed = urlparse(self._client.base_url)
        secure = parsed.scheme in ("https", "wss")
        host = parsed.hostname or "127.0.0.1"
        default_port = 443 if secure else 80
        port = parsed.port or default_port
        try:
            sock = socket.create_connection(
                (host, port), timeout=self._client.timeout_s
            )
        except OSError as e:
            raise ConnectionError(
                f"WebSocket TCP connect to {host}:{port} failed: "
                f"{type(e).__name__}",
                code="ws_connect_failed",
            )
        if secure:
            ctx = (
                self._client._ssl_context
                or ssl.create_default_context()
            )
            try:
                sock = ctx.wrap_socket(
                    sock, server_hostname=host
                )
            except ssl.SSLError as e:
                sock.close()
                raise ConnectionError(
                    f"WebSocket TLS failed: {type(e).__name__}",
                    code="ws_tls_failed",
                )
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        # Authentication uses the Authorization header (preferred);
        # the token never goes in the URL.
        request = (
            "GET /v1/stream HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Authorization: Bearer {self._token}\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        try:
            sock.settimeout(self._timeout_s)
            sock.sendall(request.encode("utf-8"))
            response = self._read_http_response(sock)
        except OSError as e:
            sock.close()
            raise ConnectionError(
                f"WebSocket handshake failed: {type(e).__name__}",
                code="ws_handshake_failed",
            )
        status_line = response.split("\r\n", 1)[0]
        if "101" not in status_line:
            sock.close()
            if "401" in status_line:
                raise AuthenticationError(
                    "WebSocket authentication failed",
                    status=401,
                )
            raise ProtocolError(
                "WebSocket upgrade rejected: "
                + redact_token_text(status_line),
                code="ws_upgrade_rejected",
            )
        self._sock = sock
        self.connected = True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._receive_loop,
            daemon=True,
            name="afnan-control-ws",
        )
        self._thread.start()

    @staticmethod
    def _read_http_response(sock: socket.socket) -> str:
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if len(data) > 65536:
                raise ConnectionError(
                    "WebSocket handshake headers too large",
                    code="ws_handshake_failed",
                )
        return data.decode("utf-8", errors="replace")

    # -- sending -----------------------------------------------------------

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        with self._send_lock:
            sock = self._sock
            if sock is None or self._stop.is_set():
                raise ConnectionError(
                    "WebSocket is not connected",
                    code="ws_not_connected",
                )
            try:
                sock.sendall(_encode_masked(opcode, payload))
            except OSError as e:
                raise ConnectionError(
                    f"WebSocket send failed: {type(e).__name__}",
                    code="ws_send_failed",
                )

    def send_text(self, text: str) -> None:
        """Send one text frame."""
        self._send_frame(
            _WSFrame.OPCODE_TEXT, text.encode("utf-8")
        )

    def send_command(self, command_body: dict[str, Any]) -> None:
        """Send a RemoteCommand body over the socket.

        The server answers with a ``command_result`` frame that is
        dispatched to the ``on_command_result`` callback.
        """
        self.send_text(json.dumps(command_body))

    def send_heartbeat(self) -> None:
        """Send the protocol heartbeat shortcut."""
        self.send_text(json.dumps({"type": "heartbeat"}))

    def ping(self, payload: bytes = b"") -> None:
        """Send a WebSocket ping."""
        self._send_frame(_WSFrame.OPCODE_PING, payload)

    # -- receiving -----------------------------------------------------------

    def _receive_loop(self) -> None:
        assert self._sock is not None
        sock = self._sock
        try:
            while not self._stop.is_set():
                frame = _decode_server_frame(sock)
                if frame is None:
                    break
                opcode, payload = frame
                if opcode == _WSFrame.OPCODE_CLOSE:
                    self._send_close_reply()
                    break
                if opcode == _WSFrame.OPCODE_PING:
                    self._send_frame(
                        _WSFrame.OPCODE_PONG, payload
                    )
                    continue
                if opcode == _WSFrame.OPCODE_PONG:
                    continue
                if opcode != _WSFrame.OPCODE_TEXT:
                    continue
                self._dispatch_text(payload)
        except OSError:
            pass
        finally:
            self._teardown()

    def _send_close_reply(self) -> None:
        try:
            self._send_frame(_WSFrame.OPCODE_CLOSE, b"")
        except Exception:
            pass

    def _dispatch_text(self, payload: bytes) -> None:
        try:
            message = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(message, dict):
            return
        kind = message.get("type")
        if kind == "event":
            event = message.get("event")
            if isinstance(event, dict) and self._on_event:
                self._on_event(event)
        elif kind == "command_result":
            result = message.get("result")
            if isinstance(result, dict) and self._on_command_result:
                self._on_command_result(result)
        elif kind == "heartbeat_ack":
            return
        elif kind == "error":
            if self._on_error:
                self._on_error(message)

    # -- teardown --------------------------------------------------------------

    def _teardown(self) -> None:
        self.connected = False
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    def disconnect(self) -> None:
        """Send the close handshake and close the socket."""
        self._stop.set()
        self._send_close_reply()
        thread, self._thread = self._thread, None
        self._teardown()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def join(self, timeout: float | None = None) -> None:
        """Wait for the receive loop to finish."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

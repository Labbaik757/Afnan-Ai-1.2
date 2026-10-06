"""Control plane facade.

``ControlPlane`` wires the existing runtime components into a secure
remote control surface.  It never duplicates them: TaskManager,
GoalManager, SecurityCenter, ActivityCenter, ApprovalCenter,
ArtifactManager, BrowserController and ComputerRuntime are injected
and stay authoritative.

Typical wiring::

    plane = ControlPlane(
        security_center=agent.security_center,
        task_manager=agent.task_manager,
        goal_manager=agent.goal_manager,
        activity_center=agent.activity_center,
        ...
    )
    plane.start()   # background maintenance
"""

from __future__ import annotations

import threading
import time
from typing import Any

from afnan_ai.control import views
from afnan_ai.control.approvals import ApprovalGateway
from afnan_ai.control.audit import ControlPlaneAuditAdapter
from afnan_ai.control.auth import (
    AuthenticationError,
    AuthProvider,
)
from afnan_ai.control.capabilities import (
    DEFAULT_PAIRED_CAPABILITIES,
    define_remote_capabilities,
)
from afnan_ai.control.commands import RemoteCommandRouter
from afnan_ai.control.devices import DeviceRegistry
from afnan_ai.control.events import EventStream
from afnan_ai.control.idempotency import IdempotencyStore
from afnan_ai.control.metrics import MetricsCollector
from afnan_ai.control.models import (
    ClientSession,
    CommandStatus,
    CommandType,
    DeviceTrust,
    RemoteCommand,
    RemoteCommandResult,
    SessionState,
)
from afnan_ai.control.pairing import PairingManager, PairingError
from afnan_ai.control.sessions import ClientSessionManager
from afnan_ai.security.models import Actor, ActorKind


class ControlPlaneError(Exception):
    """Control plane operation failed."""


class ControlPlane:
    """Secure remote & multi-device control for the agent runtime."""

    def __init__(
        self,
        *,
        security_center,
        task_manager=None,
        goal_manager=None,
        scheduler=None,
        activity_center=None,
        approval_center=None,
        artifact_manager=None,
        browser_controller=None,
        computer_runtime=None,
        device_path: str | None = None,
        session_ttl_s: float = 24 * 3600,
        token_ttl_s: float = 12 * 3600,
        command_ttl_s: float = 120.0,
    ) -> None:
        self.security = security_center
        # Register the remote capability namespace once.
        define_remote_capabilities(
            self.security.capabilities
        )
        self.devices = DeviceRegistry(device_path)
        self.sessions = ClientSessionManager(
            session_ttl_s=session_ttl_s
        )
        self.auth = AuthProvider(
            self.security.vault,
            token_ttl_s=token_ttl_s,
            on_auth_event=self._on_auth_event,
        )
        self.pairing = PairingManager(
            rate_limiter=self.security.rate_limiter,
            on_pairing_event=self._on_pairing_event,
        )
        self.metrics = MetricsCollector()
        self.audit = ControlPlaneAuditAdapter(
            self.security.audit_center
        )
        self.events = EventStream(
            activity_center,
            on_event=self._on_stream_event,
        )
        self.approval_gateway = ApprovalGateway(
            approval_center,
            event_stream=self.events,
            on_remote_approval=self._on_remote_approval,
        )
        self.router = RemoteCommandRouter(
            security_center=security_center,
            task_manager=task_manager,
            goal_manager=goal_manager,
            scheduler=scheduler,
            activity_center=activity_center,
            approval_center=approval_center,
            artifact_manager=artifact_manager,
            browser_controller=browser_controller,
            computer_runtime=computer_runtime,
            device_registry=self.devices,
            session_manager=self.sessions,
            auth_provider=self.auth,
            idempotency=IdempotencyStore(),
            audit_adapter=self.audit,
            metrics=self.metrics,
            command_ttl_s=command_ttl_s,
            on_command=self._on_command,
        )
        self._lock = threading.RLock()
        self._maintenance_thread: threading.Thread | None = None
        self._stop_maintenance = threading.Event()
        self._started_at = time.time()

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Start background maintenance (reaping, heartbeats)."""
        if self._maintenance_thread is not None:
            return
        self._stop_maintenance.clear()
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True,
            name="afnan-control-maintenance",
        )
        self._maintenance_thread.start()

    def stop(self) -> None:
        self._stop_maintenance.set()
        if self._maintenance_thread is not None:
            self._maintenance_thread.join(timeout=5.0)
            self._maintenance_thread = None
        try:
            self.events.close()
        except Exception:
            pass

    def _maintenance_loop(self) -> None:
        while not self._stop_maintenance.wait(30.0):
            try:
                self.sessions.reap_expired()
                self.auth.reap_expired()
                stale = self.sessions.check_heartbeats()
                for session_id in stale:
                    try:
                        self.sessions.transition(
                            session_id,
                            SessionState.RECONNECTING,
                        )
                    except Exception:
                        pass
                try:
                    live = self.sessions.list(live_only=True)
                    self.metrics.set_presence(
                        devices=len(self.devices.list()),
                        sessions=len(live),
                    )
                except Exception:
                    pass
            except Exception:
                pass

    # -- pairing ---------------------------------------------------------------

    def pair_request(
        self, metadata: dict[str, Any]
    ) -> tuple[str, str]:
        """Start pairing for a new device. Returns (pairing_id, code).

        The code is shown once; only its hash is retained.
        """
        if not isinstance(metadata, dict):
            raise ControlPlaneError("metadata must be an object")
        device_id = str(metadata.get("device_id", "")).strip()
        if not device_id:
            raise ControlPlaneError("device_id is required")
        existing = self.devices.get(device_id)
        if (
            existing is not None
            and existing.trust == DeviceTrust.REVOKED
        ):
            # Re-pairing a revoked device is never automatic.
            # The owner must clear the revocation explicitly.
            raise PairingError(
                "device is revoked; revocation must be "
                "cleared by the owner before re-pairing"
            )
        self.devices.register(
            device_id,
            device_name=str(metadata.get("device_name", ""))[:120],
            platform=str(metadata.get("platform", ""))[:64],
            platform_version=str(
                metadata.get("platform_version", "")
            )[:64],
            client_version=str(
                metadata.get("client_version", "")
            )[:64],
            capabilities=tuple(
                metadata.get("capabilities", ())
            )[:32],
        )
        self.devices.set_trust(
            device_id, DeviceTrust.PENDING
        )
        request, code = self.pairing.create_request(
            requesting_device_id=device_id,
            requesting_device_name=str(
                metadata.get("device_name", "")
            )[:120],
            platform=str(metadata.get("platform", ""))[:64],
            rate_key=device_id,
        )
        return request.pairing_id, code

    def pair_approve(
        self, pairing_id: str, *, approved_by: str
    ) -> dict[str, Any]:
        """The user explicitly approves a pairing request."""
        request = self.pairing.approve(
            pairing_id, approved_by=approved_by
        )
        return request.to_dict()

    def pair_reject(
        self, pairing_id: str, *, approved_by: str = ""
    ) -> dict[str, Any]:
        request = self.pairing.reject(
            pairing_id, approved_by=approved_by
        )
        self.devices.set_trust(
            request.requesting_device_id, DeviceTrust.UNPAIRED
        )
        return request.to_dict()

    def pair_redeem(
        self, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Redeem an approved pairing code -> device trust + session."""
        pairing_id = str(body.get("pairing_id", ""))
        code = str(body.get("code", ""))
        request = self.pairing.redeem(pairing_id, code)
        device_id = request.requesting_device_id
        self.devices.set_trust(device_id, DeviceTrust.PAIRED)
        session = self.sessions.create(
            device_id,
            principal=f"device:{device_id}",
            capabilities=DEFAULT_PAIRED_CAPABILITIES,
            client_version="",
            transport="pairing",
        )
        # Grant the default least-privilege capability set to the
        # session actor id the router authorizes against.
        session_actor = Actor(
            kind=ActorKind.USER,
            actor_id=f"remote:{device_id}:{session.session_id}",
        )
        for capability in DEFAULT_PAIRED_CAPABILITIES:
            try:
                self.security.permissions.grant(
                    session_actor, capability
                )
            except Exception:
                pass
        session.transition(SessionState.AUTHENTICATING)
        token = self.auth.issue_token(session)
        session.transition(SessionState.AUTHORIZED)
        self.devices.mark_seen(device_id)
        self.audit.device_paired(
            device_id=device_id,
            approved_by=request.approved_by,
        )
        self.events.publish(
            "device.connected",
            {"device_id": device_id},
            device_id=device_id,
        )
        return {
            "device_id": device_id,
            "session_id": session.session_id,
            "token": token,
            "capabilities": list(DEFAULT_PAIRED_CAPABILITIES),
            "expires_at": session.expires_at,
        }

    def clear_device_revocation(
        self, device_id: str, *, cleared_by: str
    ) -> dict[str, Any]:
        """Owner-only: clear a device revocation so it may pair again.

        This is the only path back from REVOKED; re-pairing is never
        automatic.  The action is audited and published as an event.
        """
        info = self.devices.clear_revocation(
            device_id, cleared_by=cleared_by
        )
        self.audit.device_event(
            "revocation_cleared",
            device_id=device_id,
            actor=cleared_by,
        )
        self.events.publish(
            "device.revocation_cleared",
            {"device_id": device_id, "cleared_by": cleared_by},
            device_id=device_id,
        )
        return {
            "device_id": device_id,
            "trust": info.trust.value,
        }

    # -- authentication ------------------------------------------------------------

    def authenticate(self, token: str) -> ClientSession:
        """Authenticate a bearer token -> live session or raise."""
        record = self.auth.verify(token)
        session = self.sessions.get(record.session_id)
        if session is None:
            self.metrics.record_auth_failure()
            raise AuthenticationError("session not found")
        if not session.is_live():
            self.metrics.record_auth_failure()
            raise AuthenticationError(
                f"session is {session.state.value}"
            )
        if session.is_expired():
            try:
                self.sessions.transition(
                    session.session_id, SessionState.EXPIRED
                )
            except Exception:
                pass
            self.metrics.record_auth_failure()
            raise AuthenticationError("session expired")
        # Device trust is re-checked on every authentication.
        device = self.devices.get(session.device_id)
        if device is None or device.trust != DeviceTrust.PAIRED:
            self.metrics.record_auth_failure()
            raise AuthenticationError("device not paired")
        session.touch()
        return session

    def rotate_token(
        self, old_token: str, presented: str
    ) -> dict[str, Any]:
        session = self.authenticate(old_token)
        new_token = self.auth.rotate(session, presented)
        return {
            "token": new_token,
            "session_id": session.session_id,
        }

    def heartbeat(self, token: str) -> dict[str, Any]:
        session = self.authenticate(token)
        live = self.sessions.heartbeat(session.session_id)
        return {
            "session_id": live.session_id,
            "state": live.state.value,
            "at": time.time(),
        }

    # -- commands ----------------------------------------------------------------------

    def execute_command(
        self, token: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Authenticate + route one remote command."""
        session = self.authenticate(token)
        return self.execute_command_session(session, body)

    def execute_command_session(
        self, session: ClientSession, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Route one remote command for an already-authenticated
        session (e.g. a WebSocket connection authenticated at
        upgrade time).  Liveness and device trust are re-checked."""
        if not session.is_live():
            raise AuthenticationError(
                f"session is {session.state.value}"
            )
        device = self.devices.get(session.device_id)
        if device is None or device.trust != DeviceTrust.PAIRED:
            raise AuthenticationError("device not paired")
        try:
            command = RemoteCommand.from_dict(body or {})
        except Exception as e:
            raise ControlPlaneError(f"invalid command: {e}")
        if command.session_id and (
            command.session_id != session.session_id
        ):
            raise ControlPlaneError(
                "command session does not match auth session"
            )
        command.session_id = session.session_id
        command.device_id = session.device_id
        command.actor_id = (
            f"remote:{session.device_id}:{session.session_id}"
        )
        # Promote to ACTIVE while working.
        try:
            if session.state in (
                SessionState.AUTHORIZED,
                SessionState.IDLE,
            ):
                self.sessions.transition(
                    session.session_id, SessionState.ACTIVE
                )
        except Exception:
            pass
        result = self.router.route(
            command,
            session_capabilities=session.capabilities,
            live_session=True,
        )
        session.last_event_seq = self.events.cursor(
            session.session_id
        )
        return result.to_dict()

    # -- events --------------------------------------------------------------------------

    def event_stream_sse(
        self, session, timeout_s: float = 30.0,
        *, from_seq: int | None = None,
    ):
        """Yield SSE chunks for a session (used by the transport).

        ``from_seq`` overrides the session's stored cursor when the
        client explicitly supplies one (reconnect semantics).
        """
        try:
            attached = self.events.attach(
                session.session_id,
                from_seq=(
                    session.last_event_seq
                    if from_seq is None
                    else from_seq
                ),
            )
            for raw in attached["missed_events"]:
                yield (
                    f"data: {__import__('json').dumps(raw)}\n\n"
                ).encode("utf-8")
            end = time.time() + timeout_s
            while time.time() < end:
                # Revocation propagation: stop streaming the moment
                # the session is no longer live.
                live = self.sessions.get(session.session_id)
                if live is None or not live.is_live():
                    break
                events = self.events.poll(session.session_id)
                for raw in events:
                    if raw.get("category") == "stream.gap":
                        self.metrics.record_event_gap()
                    self.metrics.record_events_delivered()
                    yield (
                        f"data: {__import__('json').dumps(raw)}\n\n"
                    ).encode("utf-8")
                time.sleep(0.5)
            yield b": keep-alive\n\n"
        finally:
            self.metrics.record_sse_disconnect()

    def websocket_loop(self, session, sock) -> None:
        """Serve one WebSocket connection for a session.

        The connection is authenticated at upgrade time; session
        liveness (revocation/expiry) is re-validated on every
        iteration so a revoked device/session is disconnected
        immediately.  Malformed frames close the connection with
        the RFC 6455 close code and never affect other sessions
        or the AgentLoop.
        """
        from afnan_ai.control.transport import (
            _WSFrame,
            _send_close,
            WSProtocolError,
        )

        attached = self.events.attach(
            session.session_id,
            from_seq=session.last_event_seq,
        )
        try:
            for raw in attached["missed_events"]:
                sock.sendall(
                    _WSFrame.encode_text(
                        __import__("json").dumps(raw)
                    )
                )
            sock.settimeout(30.0)
            idle_deadline = __import__("time").time() + 120.0
            import select as _select

            while True:
                # Revocation propagation: stop serving a session
                # the moment it is no longer live.
                live = self.sessions.get(session.session_id)
                if live is None or not live.is_live():
                    _send_close(sock, 1008, "session revoked")
                    break
                if __import__("time").time() > idle_deadline:
                    _send_close(sock, 1000, "idle timeout")
                    break
                # Wait briefly for client input so server-side
                # events are pushed even when the client is
                # silent.  The socket timeout bounds slow sends.
                try:
                    readable, _, _ = _select.select(
                        [sock], [], [], 0.5
                    )
                except (OSError, ValueError):
                    break
                # TLS sockets may hold decrypted bytes that
                # select(2) cannot see; check pending() too.
                pending = False
                try:
                    pending = bool(
                        getattr(sock, "pending", lambda: 0)()
                    )
                except Exception:
                    pending = False
                if readable or pending:
                    # Client -> server: commands as JSON text frames.
                    try:
                        frame = _WSFrame.decode(sock)
                    except WSProtocolError as e:
                        _send_close(sock, e.code, str(e))
                        break
                    except OSError:
                        break
                    if frame is None:
                        break
                    idle_deadline = (
                        __import__("time").time() + 120.0
                    )
                    opcode, payload = frame
                    if opcode == _WSFrame.OPCODE_CLOSE:
                        _send_close(sock, 1000)
                        break
                    if opcode == _WSFrame.OPCODE_PING:
                        try:
                            sock.sendall(
                                _WSFrame.encode_pong(payload)
                            )
                        except OSError:
                            break
                        continue
                    if opcode != _WSFrame.OPCODE_TEXT:
                        # Binary frames are not part of the protocol.
                        continue
                    try:
                        body = __import__("json").loads(
                            payload.decode("utf-8")
                        )
                    except Exception:
                        continue
                    # Heartbeat shortcut.
                    if (
                        isinstance(body, dict)
                        and body.get("type") == "heartbeat"
                    ):
                        try:
                            self.sessions.heartbeat(
                                session.session_id
                            )
                            sock.sendall(
                                _WSFrame.encode_text(
                                    __import__("json").dumps(
                                        {"type": "heartbeat_ack"}
                                    )
                                )
                            )
                        except OSError:
                            break
                        continue
                    # Remote command over the socket.  The session was
                    # authenticated at upgrade time; re-validate
                    # liveness per command instead of re-verifying the
                    # bearer token.
                    try:
                        result = self.execute_command_session(
                            session, body
                        )
                        sock.sendall(
                            _WSFrame.encode_text(
                                __import__("json").dumps(
                                    {
                                        "type": "command_result",
                                        "result": result,
                                    }
                                )
                            )
                        )
                    except Exception as e:
                        # Never leak internal detail to the socket:
                        # only known-safe error types keep their message.
                        if isinstance(
                            e,
                            (
                                AuthenticationError,
                                ControlPlaneError,
                                PermissionError,
                                KeyError,
                            ),
                        ):
                            safe_error = str(e)[:300]
                        else:
                            safe_error = "command failed"
                        try:
                            sock.sendall(
                                _WSFrame.encode_text(
                                    __import__("json").dumps(
                                        {
                                            "type": "error",
                                            "error": safe_error,
                                        }
                                    )
                                )
                            )
                        except OSError:
                            break
                # Server -> client: pending events.
                try:
                    for raw in self.events.poll(
                        session.session_id
                    ):
                        if raw.get("category") == "stream.gap":
                            self.metrics.record_event_gap()
                        self.metrics.record_events_delivered()
                        sock.sendall(
                            _WSFrame.encode_text(
                                __import__("json").dumps(
                                    {
                                        "type": "event",
                                        "event": raw,
                                    }
                                )
                            )
                        )
                except OSError:
                    break
        finally:
            self.metrics.record_ws_disconnect()
            try:
                self.events.detach(session.session_id)
            except Exception:
                pass

    # -- health ------------------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "protocol_version": "v1",
            "uptime_s": round(time.time() - self._started_at, 1),
            "devices": len(self.devices.list()),
            "sessions_live": len(
                self.sessions.list(live_only=True)
            ),
            "emergency_tripped": bool(
                self.security.emergency.is_tripped()
            ),
        }

    # -- internal hooks --------------------------------------------------------------------------

    def _on_auth_event(self, event: dict[str, Any]) -> None:
        try:
            self.events.publish(
                "security.alert",
                {
                    "kind": event.get("event", ""),
                    "device_id": event.get("device_id", ""),
                },
                device_id=event.get("device_id", ""),
            )
        except Exception:
            pass

    def _on_pairing_event(self, event: dict[str, Any]) -> None:
        self._on_auth_event(event)

    def _on_stream_event(self, event: dict[str, Any]) -> None:
        pass

    def _on_command(self, info: dict[str, Any]) -> None:
        try:
            self.events.publish(
                "command.completed",
                {
                    "command_id": info.get("command_id", ""),
                    "command_type": info.get(
                        "command_type", ""
                    ),
                    "status": info.get("status", ""),
                },
                device_id=info.get("device_id", ""),
            )
        except Exception:
            pass

    def _on_remote_approval(
        self, info: dict[str, Any]
    ) -> None:
        self.audit.approval_decided(
            approval_id=info.get("approval_id", ""),
            granted=bool(info.get("granted", False)),
            device_id=info.get("device_id", ""),
            session_id=info.get("session_id", ""),
        )

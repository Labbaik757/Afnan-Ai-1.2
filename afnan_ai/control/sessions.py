"""Client session management.

Persistent sessions for remote clients with a strict state machine.
Invalid transitions are rejected; expired sessions are reaped;
revocation is immediate and terminal.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from afnan_ai.control.models import (
    ClientSession,
    SessionState,
    SessionTransitionError,
)


class ClientSessionManager:
    """Owns the lifecycle of remote client sessions."""

    def __init__(
        self,
        *,
        session_ttl_s: float = 24 * 3600,
        idle_timeout_s: float = 30 * 60,
        heartbeat_timeout_s: float = 120.0,
    ) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, ClientSession] = {}
        self._session_ttl_s = session_ttl_s
        self._idle_timeout_s = idle_timeout_s
        self._heartbeat_timeout_s = heartbeat_timeout_s

    # -- lifecycle --------------------------------------------------------

    def create(
        self,
        device_id: str,
        *,
        principal: str = "",
        capabilities: tuple[str, ...] | list[str] = (),
        client_version: str = "",
        transport: str = "",
    ) -> ClientSession:
        session = ClientSession(
            session_id=f"sess_{uuid.uuid4().hex[:16]}",
            device_id=device_id,
            principal=principal,
            capabilities=tuple(capabilities),
            state=SessionState.CONNECTING,
            expires_at=time.time() + self._session_ttl_s,
            correlation_id=uuid.uuid4().hex[:12],
            client_version=client_version,
            transport=transport,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> ClientSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def transition(
        self, session_id: str, target: SessionState
    ) -> ClientSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(
                    f"unknown session: {session_id}"
                )
            session.transition(target)
            return session

    def touch(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.touch()

    def heartbeat(self, session_id: str) -> ClientSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(
                    f"unknown session: {session_id}"
                )
            session.heartbeat_at = time.time()
            session.missed_heartbeats = 0
            session.touch()
            if session.state == SessionState.IDLE:
                session.transition(SessionState.ACTIVE)
            return session

    def list(
        self,
        *,
        device_id: str | None = None,
        live_only: bool = False,
    ) -> list[ClientSession]:
        with self._lock:
            sessions = list(self._sessions.values())
        if device_id is not None:
            sessions = [
                s for s in sessions if s.device_id == device_id
            ]
        if live_only:
            sessions = [s for s in sessions if s.is_live()]
        return sessions

    # -- revocation / expiry ----------------------------------------------

    def revoke(self, session_id: str) -> ClientSession:
        """Revoke a session immediately (terminal)."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(
                    f"unknown session: {session_id}"
                )
            # Revocation bypasses the normal transition table:
            # it is always allowed from any non-terminal state.
            if session.state not in (
                SessionState.REVOKED,
                SessionState.EXPIRED,
                SessionState.CLOSED,
            ):
                session.state = SessionState.REVOKED
                session.touch()
            return session

    def revoke_device_sessions(self, device_id: str) -> int:
        count = 0
        with self._lock:
            ids = [
                s.session_id
                for s in self._sessions.values()
                if s.device_id == device_id
            ]
        for session_id in ids:
            try:
                self.revoke(session_id)
                count += 1
            except KeyError:
                pass
        return count

    def reap_expired(self, now: float | None = None) -> int:
        """Mark expired sessions EXPIRED. Returns the count."""
        now = time.time() if now is None else now
        count = 0
        with self._lock:
            for session in self._sessions.values():
                if session.state in (
                    SessionState.REVOKED,
                    SessionState.EXPIRED,
                    SessionState.CLOSED,
                ):
                    continue
                expired = session.is_expired(now)
                idle = (
                    now - session.last_activity
                    > self._idle_timeout_s
                )
                if expired or idle:
                    session.state = SessionState.EXPIRED
                    session.touch()
                    count += 1
        return count

    def check_heartbeats(
        self, now: float | None = None
    ) -> list[str]:
        """Flag sessions that missed their heartbeat window."""
        now = time.time() if now is None else now
        stale: list[str] = []
        with self._lock:
            for session in self._sessions.values():
                if not session.is_live():
                    continue
                if not session.heartbeat_at:
                    continue
                if (
                    now - session.heartbeat_at
                    > self._heartbeat_timeout_s
                ):
                    session.missed_heartbeats += 1
                    stale.append(session.session_id)
        return stale

    def close(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            try:
                session.transition(SessionState.CLOSED)
            except SessionTransitionError:
                session.state = SessionState.CLOSED

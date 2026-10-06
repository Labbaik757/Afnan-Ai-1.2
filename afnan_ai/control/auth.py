"""Authentication for remote clients.

Token-based authentication backed by the existing
:class:`~afnan_ai.security.vault.CredentialVault`.  Only SHA-256
hashes of tokens are kept in the control plane; the token values
themselves live in the vault (or are shown once at issuance and
then dropped).  No long-lived static master token: every session
token expires and can be rotated or revoked.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any

from afnan_ai.control.models import ClientSession


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class TokenRecord:
    token_hash: str
    vault_owner: str
    vault_name: str
    device_id: str
    session_id: str
    issued_at: float
    expires_at: float
    rotated_from_hash: str = ""
    revoked: bool = False


class AuthenticationError(Exception):
    """Authentication failed (invalid, expired or revoked token)."""


class AuthProvider:
    """Issues, verifies, rotates and revokes session tokens."""

    def __init__(
        self,
        vault,
        *,
        token_ttl_s: float = 12 * 3600,
        on_auth_event=None,
    ) -> None:
        self._vault = vault
        self._token_ttl_s = token_ttl_s
        self._on_auth_event = on_auth_event
        self._lock = threading.RLock()
        self._tokens: dict[str, TokenRecord] = {}

    # -- issuance ---------------------------------------------------------

    def issue_token(
        self,
        session: ClientSession,
        *,
        ttl_s: float | None = None,
    ) -> str:
        """Issue a bearer token for a session. Returns the token
        value exactly once — the caller must deliver it to the
        client; it is never stored in plaintext."""
        token = f"afn_{secrets.token_urlsafe(32)}"
        token_hash = _hash(token)
        ttl = self._token_ttl_s if ttl_s is None else ttl_s
        now = time.time()
        vault_owner = f"device:{session.device_id}"
        vault_name = f"session:{session.session_id}"
        # The vault holds the value under an opaque ref so the
        # control plane can revoke without ever logging it.
        vault_ref = self._vault.put(
            owner=vault_owner,
            name=vault_name,
            value=token,
            kind="session_token",
        )
        record = TokenRecord(
            token_hash=token_hash,
            vault_owner=vault_owner,
            vault_name=vault_name,
            device_id=session.device_id,
            session_id=session.session_id,
            issued_at=now,
            expires_at=now + ttl,
        )
        with self._lock:
            self._tokens[token_hash] = record
        session.token_hash = token_hash
        session.token_ref = vault_ref
        self._emit(
            "token.issued",
            session.device_id,
            session.session_id,
        )
        return token

    # -- verification -------------------------------------------------------

    def verify(self, token: str) -> TokenRecord:
        """Verify a presented bearer token. Raises on any failure."""
        token_hash = _hash(token or "")
        with self._lock:
            record = self._tokens.get(token_hash)
        if record is None:
            raise AuthenticationError("invalid token")
        if record.revoked:
            raise AuthenticationError("token revoked")
        now = time.time()
        if now >= record.expires_at:
            raise AuthenticationError("token expired")
        # Constant-time compare against the stored hash.
        if not hmac.compare_digest(record.token_hash, token_hash):
            raise AuthenticationError("invalid token")
        return record

    # -- rotation -----------------------------------------------------------

    def rotate(
        self, session: ClientSession, old_token: str
    ) -> str:
        """Rotate a session token: the old token is revoked and a
        fresh one issued.  The old token stops working immediately."""
        old_hash = _hash(old_token or "")
        with self._lock:
            record = self._tokens.get(old_hash)
            if record is None or record.revoked:
                raise AuthenticationError(
                    "cannot rotate: invalid token"
                )
            if record.session_id != session.session_id:
                raise AuthenticationError(
                    "cannot rotate: token/session mismatch"
                )
            record.revoked = True
        try:
            self._vault.delete(record.vault_owner, record.vault_name)
        except Exception:
            pass
        new_token = self.issue_token(session)
        with self._lock:
            new_rec = self._tokens.get(_hash(new_token))
            if new_rec is not None:
                new_rec.rotated_from_hash = old_hash
        self._emit(
            "token.rotated",
            session.device_id,
            session.session_id,
        )
        return new_token

    # -- revocation ---------------------------------------------------------

    def revoke_token(self, token_hash: str) -> bool:
        with self._lock:
            record = self._tokens.get(token_hash)
            if record is None:
                return False
            record.revoked = True
        try:
            self._vault.delete(record.vault_owner, record.vault_name)
        except Exception:
            pass
        self._emit(
            "token.revoked", record.device_id, record.session_id
        )
        return True

    def revoke_session_tokens(self, session_id: str) -> int:
        count = 0
        with self._lock:
            hashes = [
                h
                for h, r in self._tokens.items()
                if r.session_id == session_id and not r.revoked
            ]
        for token_hash in hashes:
            if self.revoke_token(token_hash):
                count += 1
        return count

    def revoke_device_tokens(self, device_id: str) -> int:
        count = 0
        with self._lock:
            hashes = [
                h
                for h, r in self._tokens.items()
                if r.device_id == device_id and not r.revoked
            ]
        for token_hash in hashes:
            if self.revoke_token(token_hash):
                count += 1
        return count

    def reap_expired(self) -> int:
        now = time.time()
        count = 0
        with self._lock:
            for record in self._tokens.values():
                if not record.revoked and now >= record.expires_at:
                    record.revoked = True
                    count += 1
        return count

    def _emit(
        self, event: str, device_id: str, session_id: str
    ) -> None:
        if self._on_auth_event is not None:
            try:
                self._on_auth_event(
                    {
                        "event": event,
                        "device_id": device_id,
                        "session_id": session_id,
                        "at": time.time(),
                    }
                )
            except Exception:
                pass

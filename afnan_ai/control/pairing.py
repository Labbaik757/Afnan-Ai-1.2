"""Secure device pairing.

Pairing flow (conceptual):

    Device A:  generate a pairing request (short code)
    Device B:  user verifies the request out-of-band
    User:      explicitly approves (or rejects)
    System:    establishes a trust relationship, issues credentials

Pairing codes are short-lived, one-time, rate-limited and
non-replayable.  Only the code hash is stored; the code itself is
shown once at creation.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
import uuid
from typing import Any

from afnan_ai.control.models import PairingRequest, PairingState


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


class PairingError(Exception):
    """Pairing failed (invalid, expired, replayed or rate-limited)."""


class PairingManager:
    """Issues and redeems one-time pairing codes."""

    def __init__(
        self,
        *,
        code_ttl_s: float = 300.0,
        max_attempts: int = 5,
        rate_limiter=None,
        on_pairing_event=None,
    ) -> None:
        self._code_ttl_s = code_ttl_s
        self._max_attempts = max_attempts
        self._rate_limiter = rate_limiter
        self._on_pairing_event = on_pairing_event
        self._lock = threading.RLock()
        self._requests: dict[str, PairingRequest] = {}

    # -- issuance ---------------------------------------------------------

    def create_request(
        self,
        *,
        requesting_device_id: str,
        requesting_device_name: str = "",
        platform: str = "",
        requested_capabilities: tuple[str, ...] | list[str] = (),
        rate_key: str = "",
    ) -> tuple[PairingRequest, str]:
        """Create a pairing request. Returns (request, code) — the
        code is returned exactly once and must be shown to the user."""
        if self._rate_limiter is not None and rate_key:
            verdict = self._rate_limiter.check(
                f"control:pairing:{rate_key}"
            )
            if isinstance(verdict, dict) and not verdict.get(
                "ok", True
            ):
                raise PairingError(
                    "pairing rate limit exceeded"
                )
            self._rate_limiter.record_call(
                f"control:pairing:{rate_key}"
            )
        # 6-digit code: short enough to read aloud, 1M combinations,
        # single-use with a short TTL and attempt cap.
        code = f"{secrets.randbelow(1_000_000):06d}"
        now = time.time()
        request = PairingRequest(
            pairing_id=f"pair_{uuid.uuid4().hex[:16]}",
            code_hash=_hash_code(code),
            requesting_device_id=requesting_device_id,
            requesting_device_name=requesting_device_name,
            platform=platform,
            requested_capabilities=tuple(requested_capabilities),
            state=PairingState.PENDING,
            created_at=now,
            expires_at=now + self._code_ttl_s,
        )
        with self._lock:
            self._requests[request.pairing_id] = request
        self._emit("pairing.created", request)
        return request, code

    def get(self, pairing_id: str) -> PairingRequest | None:
        with self._lock:
            return self._requests.get(pairing_id)

    def list_pending(self) -> list[PairingRequest]:
        self._reap()
        with self._lock:
            return [
                r
                for r in self._requests.values()
                if r.state == PairingState.PENDING
            ]

    # -- user decision ----------------------------------------------------

    def approve(
        self, pairing_id: str, *, approved_by: str
    ) -> PairingRequest:
        with self._lock:
            request = self._requests.get(pairing_id)
            if request is None:
                raise PairingError("unknown pairing request")
            self._ensure_usable(request)
            request.state = PairingState.APPROVED
            request.approved_by = approved_by
        self._emit("pairing.approved", request)
        return request

    def reject(
        self, pairing_id: str, *, approved_by: str = ""
    ) -> PairingRequest:
        with self._lock:
            request = self._requests.get(pairing_id)
            if request is None:
                raise PairingError("unknown pairing request")
            if request.state != PairingState.PENDING:
                raise PairingError(
                    f"pairing is {request.state.value}"
                )
            request.state = PairingState.REJECTED
        self._emit("pairing.rejected", request)
        return request

    # -- redemption (device presents the code) ----------------------------

    def redeem(
        self, pairing_id: str, code: str
    ) -> PairingRequest:
        """Redeem a pairing code after user approval.  One-time:
        the first successful redeem consumes the request."""
        with self._lock:
            request = self._requests.get(pairing_id)
            if request is None:
                raise PairingError("unknown pairing request")
            self._ensure_usable(request)
            if request.state != PairingState.APPROVED:
                raise PairingError(
                    "pairing not approved by the user"
                )
            request.attempts += 1
            if request.attempts > self._max_attempts:
                request.state = PairingState.REJECTED
                raise PairingError(
                    "too many attempts; pairing rejected"
                )
            if not hmac.compare_digest(
                request.code_hash, _hash_code(code or "")
            ):
                raise PairingError("invalid pairing code")
            request.state = PairingState.CONSUMED
            request.consumed_at = time.time()
            # Drop the hash so the code can never be replayed.
            request.code_hash = ""
        self._emit("pairing.consumed", request)
        return request

    # -- internals --------------------------------------------------------

    def _ensure_usable(self, request: PairingRequest) -> None:
        if request.state in (
            PairingState.CONSUMED,
            PairingState.REJECTED,
        ):
            raise PairingError(
                f"pairing is {request.state.value}"
            )
        if request.is_expired():
            request.state = PairingState.EXPIRED
            raise PairingError("pairing code expired")

    def _reap(self) -> None:
        now = time.time()
        with self._lock:
            for request in self._requests.values():
                if (
                    request.state == PairingState.PENDING
                    and request.is_expired(now)
                ):
                    request.state = PairingState.EXPIRED

    def _emit(
        self, event: str, request: PairingRequest
    ) -> None:
        if self._on_pairing_event is not None:
            try:
                self._on_pairing_event(
                    {
                        "event": event,
                        "pairing_id": request.pairing_id,
                        "device_id": (
                            request.requesting_device_id
                        ),
                        "at": time.time(),
                    }
                )
            except Exception:
                pass

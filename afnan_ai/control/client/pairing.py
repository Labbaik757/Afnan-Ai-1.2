"""Pairing client for the control-plane bootstrap flow.

Implements the pairing side of the protocol documented in
``CONTROL_PLANE.md``:

1. ``request_pairing`` -> ``POST /v1/pair/request`` returns a
   ``(pairing_id, code)`` tuple.
2. The user verifies the request out-of-band and approves it on the
   owner side (server ``pair_approve``).
3. ``redeem`` -> ``POST /v1/pair/redeem`` returns the device id,
   session id, bearer token, capability list and token expiry.

The 6-digit code lives in memory only and is never logged.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.control.client.base import ControlClient
from afnan_ai.control.client.errors import ProtocolError


class PairingClient:
    """Drive device pairing against ``/v1/pair/*`` endpoints."""

    def __init__(self, client: ControlClient) -> None:
        self._client = client

    def request_pairing(
        self, device_metadata: dict[str, Any]
    ) -> tuple[str, str]:
        """Start pairing and return ``(pairing_id, code)``.

        ``device_metadata`` must include ``device_id``; the server
        also accepts ``device_name``, ``platform``,
        ``platform_version`` and ``client_version``.
        """
        if not isinstance(device_metadata, dict):
            raise ProtocolError(
                "device_metadata must be a dict",
                code="invalid_metadata",
            )
        device_id = str(device_metadata.get("device_id", "")).strip()
        if not device_id:
            raise ProtocolError(
                "device_metadata requires device_id",
                code="invalid_metadata",
            )
        result = self._client.post("/v1/pair/request", device_metadata)
        pairing_id = str(result.get("pairing_id", ""))
        code = str(result.get("code", ""))
        if not pairing_id or not code:
            raise ProtocolError(
                "pairing request returned no pairing_id/code",
                code="pairing_failed",
            )
        return pairing_id, code

    def redeem(self, pairing_id: str, code: str) -> dict[str, Any]:
        """Redeem an approved pairing code.

        Returns a dict with ``device_id``, ``session_id``, ``token``,
        ``capabilities`` and ``expires_at``.  The code is dropped
        from memory as soon as the request is built.
        """
        body = {"pairing_id": pairing_id, "code": code}
        # Never retain the code past this call.
        del code
        result = self._client.post("/v1/pair/redeem", body)
        token = str(result.get("token", ""))
        if not token:
            raise ProtocolError(
                "pairing redeem returned no token",
                code="pairing_failed",
            )
        return {
            "device_id": str(result.get("device_id", "")),
            "session_id": str(result.get("session_id", "")),
            "token": token,
            "capabilities": list(result.get("capabilities", [])),
            "expires_at": result.get("expires_at", 0.0),
        }

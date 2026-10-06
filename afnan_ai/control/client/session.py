"""Session client: heartbeat, token rotation, in-memory token custody.

The bearer token is kept in a single in-memory attribute.  ``clear()``
wipes it; nothing in this module writes the token to disk, logs or
error messages.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.control.client.base import ControlClient
from afnan_ai.control.client.errors import ProtocolError


class SessionClient:
    """Manage one control-plane session token."""

    def __init__(self, client: ControlClient) -> None:
        self._client = client
        self._token: str | None = None

    # -- token custody ---------------------------------------------------

    @property
    def token(self) -> str | None:
        """The current bearer token (in memory only)."""
        return self._token

    def set_token(self, token: str) -> None:
        """Store a bearer token received from pairing or rotation."""
        if not token:
            raise ProtocolError(
                "refusing to store an empty token",
                code="invalid_token",
            )
        self._token = token

    def clear(self) -> None:
        """Wipe the in-memory token."""
        self._token = None

    def _require_token(self, token: str | None) -> str:
        resolved = token if token is not None else self._token
        if not resolved:
            raise ProtocolError(
                "no session token available",
                code="no_token",
            )
        return resolved

    # -- session operations ------------------------------------------------

    def heartbeat(self, token: str | None = None) -> dict[str, Any]:
        """Keep the session warm.  Returns ``session_id``, ``state`` and ``at``."""
        bearer = self._require_token(token)
        return self._client.post("/v1/session/heartbeat", None, bearer)

    def rotate_token(
        self,
        old_token: str | None = None,
        presented_token: str | None = None,
    ) -> str:
        """Rotate the bearer token.

        The old token is revoked immediately by the server.  When
        ``presented_token`` is omitted it defaults to the old token,
        matching the server contract (``POST /v1/session/rotate``
        with ``{"token": <old token>}``).  The new token replaces the
        stored one and is returned.
        """
        bearer = self._require_token(old_token)
        presented = (
            presented_token if presented_token is not None else bearer
        )
        result = self._client.post(
            "/v1/session/rotate", {"token": presented}, bearer
        )
        new_token = str(result.get("token", ""))
        if not new_token:
            raise ProtocolError(
                "token rotation returned no token",
                code="rotation_failed",
            )
        self._token = new_token
        return new_token

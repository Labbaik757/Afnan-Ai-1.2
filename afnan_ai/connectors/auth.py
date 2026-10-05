"""Authentication building blocks for connectors.

OAuth2, API-key and session authentication live *inside* this
abstraction: a connector declares its
:class:`AuthType`, validates/exchanges credentials in
:meth:`~afnan_ai.connectors.base.Connector.authenticate`, and
renews them in
:meth:`~afnan_ai.connectors.base.Connector.refresh_session`.
The core agent never sees a token — it only sees
:class:`AuthSession` metadata (expiry, scopes) with the secret
material kept in the
:class:`~afnan_ai.connectors.credentials.CredentialStore`.

An interactive OAuth2 authorization-code exchange (opening the
provider's consent page and reading back the code) is a human
flow: connectors implement it through the browser tools and
the existing approval gate, then store the exchanged tokens
via :meth:`ConnectorService.provide_credentials`.  This module
provides the token/session bookkeeping around that flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


class AuthType(str, Enum):
    NONE = "none"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    SESSION = "session"


@dataclass
class AuthSession:
    """Authenticated session metadata.

    ``secret_ref`` points at the credential-store namespace
    holding the actual secret material (access token / api key /
    session cookie) — the session itself carries no secrets, so
    it is safe to keep in the service's connection bookkeeping.
    """

    auth_type: AuthType = AuthType.NONE
    secret_ref: str = ""
    token_type: str = "Bearer"
    expires_at: str | None = None
    scopes: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.auth_type, str):
            self.auth_type = AuthType(self.auth_type)
        self.scopes = tuple(self.scopes or ())

    @property
    def expired(self) -> bool:
        """True when the session has a known expiry in the past."""
        if not self.expires_at:
            return False
        try:
            expiry = datetime.fromisoformat(self.expires_at)
        except ValueError:
            return False
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) >= expiry

    def to_dict(self) -> dict[str, Any]:
        return {
            "auth_type": self.auth_type.value,
            "token_type": self.token_type,
            "expires_at": self.expires_at,
            "expired": self.expired,
            "scopes": list(self.scopes),
            # secret_ref deliberately omitted: bookkeeping only
        }


def session_expiry_in(seconds: float) -> str:
    """ISO expiry timestamp ``seconds`` from now (UTC)."""
    return (
        datetime.now(timezone.utc) + timedelta(seconds=seconds)
    ).isoformat()


def bearer_headers(access_token: str) -> dict[str, str]:
    """Authorization header for a bearer access token."""
    return {"Authorization": f"Bearer {access_token}"}


def api_key_headers(
    api_key: str,
    *,
    header_name: str = "X-API-Key",
) -> dict[str, str]:
    """Header carrying an API key (provider-specific name)."""
    return {header_name: api_key}


def require_fields(
    credentials: dict[str, Any],
    fields: tuple[str, ...],
    *,
    connector_id: str = "",
) -> None:
    """Raise for missing credential fields (values never echoed)."""
    from afnan_ai.connectors.errors import (
        AuthenticationError,
        ConnectorErrorCode,
    )

    missing = [name for name in fields if not credentials.get(name)]
    if missing:
        raise AuthenticationError(
            "Missing credential field(s): " + ", ".join(missing),
            code=ConnectorErrorCode.MISSING_CREDENTIALS,
            connector=connector_id or None,
            details={"missing_fields": missing},
        )

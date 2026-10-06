"""Shared HTTP plumbing for the control-plane client SDK.

``ControlClient`` is the transport core used by the pairing, session,
command and event clients.  It wraps :mod:`urllib` with JSON
encode/decode, HTTP status to exception mapping, optional TLS
verification control and strict token redaction in every error path.

Only this module touches the raw HTTP layer; the other clients build
on top of :meth:`ControlClient.request`.
"""

from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.request
from typing import Any

from afnan_ai.control.client.errors import (
    AuthenticationError,
    AuthorizationError,
    ConnectionError,
    ProtocolError,
)

_BEARER_RE = re.compile(r"Bearer\s+[^\s\"']+", re.IGNORECASE)
_QUERY_TOKEN_RE = re.compile(r"([?&]token=)[^&\s\"']+", re.IGNORECASE)


def redact_token_text(text: str) -> str:
    """Redact bearer tokens and token query params from a string."""
    text = _BEARER_RE.sub("Bearer <redacted>", text)
    text = _QUERY_TOKEN_RE.sub(r"\1<redacted>", text)
    return text


class ControlClient:
    """HTTP client for the control-plane ``/v1`` API.

    :param base_url: e.g. ``http://127.0.0.1:8765`` or
        ``https://afnan.example.com:8765``.  A trailing slash is
        tolerated.
    :param timeout_s: default per-request socket timeout.
    :param tls_verify: when False, TLS certificates are not
        verified.  Only meaningful for development against a
        self-signed server; never use in production.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 10.0,
        tls_verify: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.tls_verify = tls_verify
        self._closed = False
        self._ssl_context: ssl.SSLContext | None = None
        if self.base_url.startswith("https://") and not tls_verify:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            self._ssl_context = ctx

    # -- low level -------------------------------------------------------

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        token: str | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Send one JSON request and return the decoded JSON body.

        Raises :class:`AuthenticationError` on 401,
        :class:`AuthorizationError` on 403, :class:`ProtocolError`
        on 400/404/409/429 and other HTTP errors, and
        :class:`ConnectionError` on transport failures.  Token values
        never appear in raised messages.
        """
        if self._closed:
            raise ConnectionError(
                "client is closed", code="client_closed"
            )
        url = self.base_url + path
        data = None
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AfnanControlClient/1.0",
        }
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        if token:
            headers["Authorization"] = "Bearer " + token
        req = urllib.request.Request(
            url, data=data, headers=headers, method=method.upper()
        )
        timeout = self.timeout_s if timeout_s is None else timeout_s
        try:
            with urllib.request.urlopen(
                req, timeout=timeout, context=self._ssl_context
            ) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raise self._map_http_error(e, method, path)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            safe_url = redact_token_text(url)
            raise ConnectionError(
                f"{method.upper()} {safe_url}: "
                f"{type(e).__name__}",
                code="transport_failure",
            )
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            raise ProtocolError(
                "server returned non-JSON response",
                code="invalid_response",
            )
        if not isinstance(payload, dict):
            raise ProtocolError(
                "server returned a non-object response",
                code="invalid_response",
            )
        return payload

    def _map_http_error(
        self,
        error: urllib.error.HTTPError,
        method: str,
        path: str,
    ) -> ProtocolError:
        """Translate an HTTP error into a typed protocol error."""
        status = error.code
        code = f"http_{status}"
        message = f"{method.upper()} {path} failed"
        try:
            raw = error.read()
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            body = {}
        if isinstance(body, dict):
            if body.get("error_code"):
                code = str(body["error_code"])
            elif body.get("error"):
                message = str(body["error"])[:300]
        message = redact_token_text(message)
        if status == 401:
            return AuthenticationError(message, code=code, status=status)
        if status == 403:
            return AuthorizationError(message, code=code, status=status)
        if status == 400:
            return ProtocolError(message, code=code, status=status)
        if status == 404:
            return ProtocolError(message, code=code, status=status)
        if status == 409:
            return ProtocolError(message, code=code, status=status)
        if status == 429:
            return ProtocolError(
                message, code="rate_limited", status=status
            )
        return ProtocolError(message, code=code, status=status)

    def get(
        self,
        path: str,
        token: str | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Convenience GET."""
        return self.request(
            "GET", path, token=token, timeout_s=timeout_s
        )

    def post(
        self,
        path: str,
        body: dict[str, Any] | None = None,
        token: str | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Convenience POST."""
        return self.request(
            "POST",
            path,
            body=body,
            token=token,
            timeout_s=timeout_s,
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Mark the client closed; further requests are refused."""
        self._closed = True

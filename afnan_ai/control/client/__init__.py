"""Afnan AI remote control plane client SDK.

Transport-focused client for the ``/v1`` control-plane API.  It
covers pairing, session custody, command execution, SSE/WebSocket
event streaming, reconnect with backoff and a sanitized in-memory
state store.  It contains no agent reasoning logic.

Typical use::

    from afnan_ai.control.client import (
        ControlClient, PairingClient, SessionClient, CommandClient,
        EventClient,
    )

    http = ControlClient("http://127.0.0.1:8765")
    pairing = PairingClient(http)
    pairing_id, code = pairing.request_pairing({
        "device_id": "my-phone",
        "device_name": "My Phone",
        "platform": "android",
    })
    # show `code` to the user; they approve on the owner side
    creds = pairing.redeem(pairing_id, code)

    sessions = SessionClient(http)
    sessions.set_token(creds["token"])
    commands = CommandClient(http)
    result = commands.execute(
        creds["token"], "list_tasks", {}
    )

    events = EventClient(http)
    for event in events.stream_sse(creds["token"]):
        print(event["category"], event["seq"])
"""

from afnan_ai.control.client.base import ControlClient, redact_token_text
from afnan_ai.control.client.commands import CommandClient
from afnan_ai.control.client.errors import (
    AuthenticationError,
    AuthorizationError,
    ConnectionError,
    ProtocolError,
)
from afnan_ai.control.client.events import (
    EventClient,
    WebSocketConnection,
)
from afnan_ai.control.client.pairing import PairingClient
from afnan_ai.control.client.reconnect import ReconnectManager
from afnan_ai.control.client.session import SessionClient
from afnan_ai.control.client.state import (
    RemoteStateStore,
    sanitize,
)

__all__ = [
    "ControlClient",
    "PairingClient",
    "SessionClient",
    "CommandClient",
    "EventClient",
    "WebSocketConnection",
    "ReconnectManager",
    "RemoteStateStore",
    "ProtocolError",
    "AuthenticationError",
    "AuthorizationError",
    "ConnectionError",
    "redact_token_text",
    "sanitize",
]

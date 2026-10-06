"""Command client: validated, idempotent remote commands.

Builds :class:`RemoteCommand`-shaped request bodies and sends them to
``POST /v1/commands``.  The command type vocabulary is imported from
:mod:`afnan_ai.control.models` so the client can never drift from the
server's contract.

Cancellation is a server-side command (``CANCEL_TASK``); the client
does not kill anything locally and spawns no worker threads.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from afnan_ai.control.client.base import ControlClient
from afnan_ai.control.client.errors import ProtocolError
from afnan_ai.control.models import (
    PROTOCOL_VERSION,
    CommandType,
)

_MAX_PAYLOAD_CHARS = 65536


def _resolve_command_type(
    command_type: CommandType | str,
) -> CommandType:
    if isinstance(command_type, CommandType):
        return command_type
    try:
        return CommandType(str(command_type))
    except ValueError:
        known = sorted(t.value for t in CommandType)
        raise ProtocolError(
            f"unknown command_type {command_type!r}; "
            f"expected one of {', '.join(known)}",
            code="unknown_command_type",
        )


class CommandClient:
    """Execute remote commands against ``/v1/commands``."""

    def __init__(self, client: ControlClient) -> None:
        self._client = client

    def execute(
        self,
        token: str,
        command_type: CommandType | str,
        payload: dict[str, Any] | None = None,
        *,
        idempotency_key: str | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        """Execute one command and return the server result dict.

        The result carries ``status`` (completed/failed/denied/
        expired/duplicate), ``result``, ``error`` and
        ``verification``.  Transport problems raise
        :class:`ProtocolError` subclasses; a command-level failure
        is returned as data, not raised.
        """
        resolved = _resolve_command_type(command_type)
        body_payload = dict(payload) if payload else {}
        if len(str(body_payload)) > _MAX_PAYLOAD_CHARS:
            raise ProtocolError(
                "payload too large",
                code="payload_too_large",
            )
        created = time.time()
        body = {
            "command_id": f"cmd_{uuid.uuid4().hex[:16]}",
            "command_type": resolved.value,
            "payload": body_payload,
            "idempotency_key": idempotency_key
            or f"key_{uuid.uuid4().hex}",
            "created_at": created,
            "expires_at": created + max(timeout_s, 1.0),
            "protocol_version": PROTOCOL_VERSION,
        }
        result = self._client.post(
            "/v1/commands", body, token, timeout_s=timeout_s
        )
        if "status" not in result:
            raise ProtocolError(
                "server returned a malformed command result",
                code="invalid_response",
            )
        return result

    def cancel(
        self,
        token: str,
        task_id: str,
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Request server-side cancellation of a task.

        This sends the ``cancel_task`` command; the authoritative
        ``TaskManager`` performs the state transition.  The client
        itself stops nothing.
        """
        if not str(task_id).strip():
            raise ProtocolError(
                "task_id is required to cancel a task",
                code="invalid_target",
            )
        return self.execute(
            token,
            CommandType.CANCEL_TASK,
            {"task_id": str(task_id)},
            idempotency_key=idempotency_key,
        )

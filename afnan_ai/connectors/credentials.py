"""Secure credential storage abstraction.

Secrets (API keys, OAuth2 client secrets, refresh tokens,
session tokens, passwords) live *only* here — never in
``AgentState``, ``MemoryStore``, planner output, checkpoints
or ordinary logs.  The service hands them to a connector
transiently through the per-call
:class:`~afnan_ai.connectors.base.OperationContext` and never
stores them anywhere else.

Implementations:

* :class:`MemoryCredentialStore` — in-memory dict, the safe
  default for tests and short-lived sessions.  Nothing is
  written to disk.
* :class:`EnvironmentCredentialStore` — read-only view over
  environment variables (``<PREFIX>_<CONNECTOR_ID>_<FIELD>``),
  so secrets can be injected by the deployment environment
  without touching code.

Production deployments should plug a real vault (OS keychain,
cloud secret manager) behind :class:`CredentialStore`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from afnan_ai.log_config import get_logger

logger = get_logger(__name__)


class CredentialStore(ABC):
    """Namespace-keyed secret storage (namespace = connector id)."""

    @abstractmethod
    def put(self, namespace: str, values: dict[str, str]) -> None:
        """Store secret *values* under *namespace* (replaces)."""

    @abstractmethod
    def get(self, namespace: str) -> dict[str, str]:
        """Return the secrets for *namespace* ({} when absent)."""

    @abstractmethod
    def delete(self, namespace: str) -> bool:
        """Forget *namespace*; True when something was stored."""

    @abstractmethod
    def has(self, namespace: str) -> bool:
        """True when *namespace* holds any secret."""

    def namespaces(self) -> list[str]:
        """Stored namespaces (ids only — never values)."""
        return []


class MemoryCredentialStore(CredentialStore):
    """In-memory credential storage (nothing hits the disk)."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, str]] = {}

    def put(self, namespace: str, values: dict[str, str]) -> None:
        self._data[str(namespace)] = {
            str(key): str(value) for key, value in values.items()
        }
        logger.info(
            "credentials stored for connector %r (%d fields)",
            namespace, len(values),
        )

    def get(self, namespace: str) -> dict[str, str]:
        return dict(self._data.get(str(namespace), {}))

    def delete(self, namespace: str) -> bool:
        existed = str(namespace) in self._data
        self._data.pop(str(namespace), None)
        if existed:
            logger.info(
                "credentials revoked for connector %r", namespace
            )
        return existed

    def has(self, namespace: str) -> bool:
        return bool(self._data.get(str(namespace)))

    def namespaces(self) -> list[str]:
        return sorted(self._data)


class EnvironmentCredentialStore(CredentialStore):
    """Read-only credentials from environment variables.

    For connector ``email`` with prefix ``AFNAN`` the store reads
    ``AFNAN_EMAIL_API_KEY``, ``AFNAN_EMAIL_CLIENT_SECRET``, ...
    (``<PREFIX>_<CONNECTOR_ID>_<FIELD>``, upper-cased, with
    non-alphanumeric characters turned into underscores).
    """

    def __init__(self, prefix: str = "AFNAN") -> None:
        self.prefix = str(prefix).upper()

    @staticmethod
    def _field_name(connector_id: str, field: str) -> str:
        def clean(part: str) -> str:
            return "".join(
                ch if ch.isalnum() else "_" for ch in part.upper()
            )
        return f"{clean(connector_id)}_{clean(field)}"

    def _env_name(self, namespace: str, field: str) -> str:
        return f"{self.prefix}_{self._field_name(namespace, field)}"

    def put(self, namespace: str, values: dict[str, str]) -> None:
        raise NotImplementedError(
            "EnvironmentCredentialStore is read-only; set the "
            "environment variables instead"
        )

    def get(self, namespace: str) -> dict[str, str]:
        # Only the fields the deployment defined can be read; the
        # caller names the fields it needs (no blind dumping).
        found: dict[str, str] = {}
        marker = f"{self.prefix}_{self._field_name(namespace, '')}"
        for env_key, env_value in os.environ.items():
            if env_key.startswith(marker) and env_value:
                field = env_key[len(marker):].lower()
                found[field] = env_value
        return found

    def delete(self, namespace: str) -> bool:
        raise NotImplementedError(
            "EnvironmentCredentialStore is read-only"
        )

    def has(self, namespace: str) -> bool:
        return bool(self.get(namespace))

    def namespaces(self) -> list[str]:  # pragma: no cover - env-driven
        marker = f"{self.prefix}_"
        ids = set()
        for env_key in os.environ:
            if env_key.startswith(marker):
                ids.add(env_key[len(marker):].split("_")[0].lower())
        return sorted(ids)


def redact_namespace_log(namespace: str, values: dict[str, Any]) -> str:
    """Log line for a credential write — fields only, no values."""
    fields = ", ".join(sorted(str(k) for k in values))
    return f"credentials for {namespace!r}: fields [{fields}]"

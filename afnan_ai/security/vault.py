"""CredentialVault — one abstraction for all secrets.

API keys, OAuth tokens, refresh tokens, passwords,
cookies and session credentials live here and *only*
here.  They never appear in AgentState, MemoryStore,
Planner output, LLM context, screenshots, normal logs
or audit records — every read is redacted at the
boundary and every access is logged as an event
(without the value).
"""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable


class CredentialVault(ABC):
    """Secrets in, references out."""

    @abstractmethod
    def put(
        self, owner: str, name: str, value: str, *,
        kind: str = "generic",
    ) -> str:
        """Store a secret; returns an opaque reference."""

    @abstractmethod
    def get(self, ref: str) -> str | None:
        """Resolve a reference (transient use only)."""

    @abstractmethod
    def has(self, owner: str, name: str) -> bool: ...

    @abstractmethod
    def delete(self, owner: str, name: str) -> bool: ...

    @abstractmethod
    def owners(self) -> list[str]: ...


class MemoryCredentialVault(CredentialVault):
    """In-memory vault (default).  Nothing is persisted."""

    def __init__(
        self,
        on_access: Callable[[str, str, str], None]
        | None = None,
    ) -> None:
        self._secrets: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._on_access = on_access
        self._counter = 0

    def put(
        self, owner: str, name: str, value: str, *,
        kind: str = "generic",
    ) -> str:
        if not value:
            raise ValueError("refusing to store an empty secret")
        with self._lock:
            self._counter += 1
            ref = (
                f"vault://{owner}/{name}/{self._counter}"
            )
            self._secrets[ref] = {
                "owner": str(owner),
                "name": str(name),
                "kind": str(kind),
                "value": str(value),
            }
            return ref

    def get(self, ref: str) -> str | None:
        with self._lock:
            entry = self._secrets.get(ref)
        if entry is None:
            return None
        if self._on_access:
            try:
                self._on_access(
                    entry["owner"], entry["name"],
                    entry["kind"],
                )
            except Exception:
                pass  # access logging never breaks reads
        return entry["value"]

    def has(self, owner: str, name: str) -> bool:
        with self._lock:
            return any(
                e["owner"] == owner and e["name"] == name
                for e in self._secrets.values()
            )

    def delete(self, owner: str, name: str) -> bool:
        with self._lock:
            doomed = [
                ref for ref, e in self._secrets.items()
                if e["owner"] == owner
                and e["name"] == name
            ]
            for ref in doomed:
                del self._secrets[ref]
            return bool(doomed)

    def owners(self) -> list[str]:
        with self._lock:
            return sorted(
                {e["owner"] for e in self._secrets.values()}
            )

    def clear(self) -> None:
        with self._lock:
            self._secrets.clear()


class EnvCredentialVault(CredentialVault):
    """Read-only vault backed by environment variables.

    ``put``/``delete`` are refused — the environment is
    managed outside the agent.  References resolve
    ``<PREFIX>_<OWNER>_<NAME>`` (upper-cased).
    """

    def __init__(self, prefix: str = "AFNAN_SECRET") -> None:
        self._prefix = str(prefix).upper()

    def _var(self, owner: str, name: str) -> str:
        clean = lambda s: "".join(
            c if (c.isalnum()) else "_"
            for c in str(s).upper()
        )
        return f"{self._prefix}_{clean(owner)}_{clean(name)}"

    def put(self, owner, name, value, *, kind="generic"):
        raise NotImplementedError(
            "env vault is read-only"
        )

    def get(self, ref: str) -> str | None:
        # ref format: vault://owner/name[/n]
        try:
            _, rest = ref.split("://", 1)
            owner, name = rest.split("/")[:2]
        except ValueError:
            return None
        return os.environ.get(self._var(owner, name))

    def has(self, owner: str, name: str) -> bool:
        return self._var(owner, name) in os.environ

    def delete(self, owner: str, name: str) -> bool:
        raise NotImplementedError(
            "env vault is read-only"
        )

    def owners(self) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# Operation leases: the agent never receives raw credential values.
# Instead it holds a single-use lease token describing an approved
# operation; only the vault (or a trusted connector redeeming through
# the vault) can resolve it — and every redemption is audited.


class CredentialLease:
    def __init__(
        self, token: str, owner: str, name: str,
        operation: str, expires_at: float,
    ) -> None:
        self.token = token
        self.owner = owner
        self.name = name
        self.operation = operation
        self.expires_at = expires_at
        self.redeemed = False


class LeasedCredentialVault(MemoryCredentialVault):
    """Vault with operation leases on top of storage."""

    def __init__(
        self,
        on_access=None,
    ) -> None:
        super().__init__(on_access=on_access)
        self._leases: dict[str, CredentialLease] = {}
        self._lease_counter = 0

    def lease_credential(
        self, owner: str, name: str, operation: str,
        *, ttl_s: float = 300.0,
    ) -> str:
        """Issue a single-use lease token for an approved
        operation.  The token reveals nothing; redemption
        resolves the value once, then burns the token."""
        import time as _time
        import uuid as _uuid

        if not self.has(owner, name):
            raise KeyError(
                f"no credential {owner}/{name} in vault"
            )
        with self._lock:
            self._lease_counter += 1
            token = f"lease-{_uuid.uuid4().hex[:16]}"
            self._leases[token] = CredentialLease(
                token=token,
                owner=str(owner),
                name=str(name),
                operation=str(operation),
                expires_at=_time.time() + float(ttl_s),
            )
            return token

    def redeem(self, token: str) -> str | None:
        """Resolve a lease token once.  Returns None for
        unknown, expired or already-redeemed tokens."""
        import time as _time

        with self._lock:
            lease = self._leases.get(token)
            if lease is None:
                return None
            if lease.redeemed:
                return None
            if _time.time() >= lease.expires_at:
                del self._leases[token]
                return None
            lease.redeemed = True
            ref = self._ref_for(lease.owner, lease.name)
            value = (
                self._secrets.get(ref, {}).get("value")
                if ref
                else None
            )
            del self._leases[token]
        if (
            value is not None
            and self._on_access is not None
        ):
            try:
                self._on_access(
                    lease.owner,
                    lease.name,
                    f"lease:{lease.operation}",
                )
            except Exception:
                pass
        return value

    def _ref_for(
        self, owner: str, name: str
    ) -> str | None:
        for ref, entry in self._secrets.items():
            if (
                entry["owner"] == owner
                and entry["name"] == name
            ):
                return ref
        return None

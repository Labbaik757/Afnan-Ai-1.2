"""Workspace network policy enforcement.

Default-deny unless the workspace policy allows the
domain/protocol.  Also tracks request and transfer
budgets against the workspace resource limits.
"""

from __future__ import annotations

import threading
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

from .models import WorkspaceNetworkPolicy


@dataclass
class NetworkDecision:
    allowed: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reason": self.reason}


class WorkspaceNetworkGuard:
    """Enforces a workspace's network policy at runtime."""

    def __init__(
        self,
        policy: WorkspaceNetworkPolicy,
        *,
        max_requests: int = 1000,
        max_upload_mb: float = 64.0,
        max_download_mb: float = 512.0,
    ) -> None:
        self.policy = policy
        self.max_requests = max_requests
        self.max_upload_mb = max_upload_mb
        self.max_download_mb = max_download_mb
        self._lock = threading.Lock()
        self._requests = 0
        self._upload_mb = 0.0
        self._download_mb = 0.0
        self._blocked = 0

    def check_url(
        self, url: str, *, connector: str = ""
    ) -> NetworkDecision:
        """Decide whether *url* may be fetched."""
        try:
            parsed = urllib.parse.urlparse(url)
        except Exception:
            return NetworkDecision(False, "unparseable url")
        scheme = (parsed.scheme or "").lower()
        if not self.policy.allows_protocol(scheme):
            with self._lock:
                self._blocked += 1
            return NetworkDecision(
                False, f"protocol denied: {scheme!r}"
            )
        host = (parsed.hostname or "").lower()
        if connector:
            allowed_hosts = self.policy.connector_permissions.get(
                connector, ()
            )
            if allowed_hosts and not any(
                host == h.lower() or host.endswith("." + h.lower())
                for h in allowed_hosts
            ):
                with self._lock:
                    self._blocked += 1
                return NetworkDecision(
                    False,
                    f"connector {connector!r} denied host {host!r}",
                )
        elif not self.policy.allows_domain(host):
            with self._lock:
                self._blocked += 1
            return NetworkDecision(
                False, f"domain denied by policy: {host!r}"
            )
        with self._lock:
            if self._requests >= self.max_requests:
                self._blocked += 1
                return NetworkDecision(
                    False, "network request budget exhausted"
                )
            self._requests += 1
        return NetworkDecision(True, "allowed")

    def record_transfer(
        self, *, upload_mb: float = 0.0,
        download_mb: float = 0.0,
    ) -> NetworkDecision:
        """Account a transfer; deny if over budget."""
        with self._lock:
            if (
                self._upload_mb + upload_mb
                > self.max_upload_mb
            ):
                self._blocked += 1
                return NetworkDecision(
                    False, "upload budget exhausted"
                )
            if (
                self._download_mb + download_mb
                > self.max_download_mb
            ):
                self._blocked += 1
                return NetworkDecision(
                    False, "download budget exhausted"
                )
            self._upload_mb += upload_mb
            self._download_mb += download_mb
        return NetworkDecision(True, "within budget")

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": self._requests,
                "blocked": self._blocked,
                "upload_mb": round(self._upload_mb, 3),
                "download_mb": round(self._download_mb, 3),
                "max_requests": self.max_requests,
                "max_upload_mb": self.max_upload_mb,
                "max_download_mb": self.max_download_mb,
            }

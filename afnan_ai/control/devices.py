"""Device identity registry.

Stable, persistent identity for every connected device.  Trust is
explicit (unpaired/pending/paired/revoked) and revocable; device
secrets are never stored — only their SHA-256 hashes.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from afnan_ai.control.models import (
    ConnectionState,
    DeviceInfo,
    DeviceTrust,
)


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class DeviceRegistry:
    """Persistent registry of known devices."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._lock = threading.RLock()
        self._devices: dict[str, DeviceInfo] = {}
        if self._path and self._path.exists():
            self._load()

    # -- persistence ----------------------------------------------------

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return
        for item in raw.get("devices", []):
            try:
                info = DeviceInfo(
                    device_id=str(item["device_id"]),
                    device_name=str(item.get("device_name", "")),
                    platform=str(item.get("platform", "")),
                    platform_version=str(
                        item.get("platform_version", "")
                    ),
                    client_version=str(
                        item.get("client_version", "")
                    ),
                    agent_runtime_version=str(
                        item.get("agent_runtime_version", "")
                    ),
                    capabilities=tuple(
                        item.get("capabilities", ())
                    ),
                    connection_state=ConnectionState(
                        item.get("connection_state", "offline")
                    ),
                    trust=DeviceTrust(
                        item.get("trust", "unpaired")
                    ),
                    last_seen=float(item.get("last_seen", 0.0)),
                    registered_at=float(
                        item.get("registered_at", time.time())
                    ),
                    secret_hash=str(item.get("secret_hash", "")),
                )
            except (KeyError, ValueError):
                continue
            self._devices[info.device_id] = info

    def _save(self) -> None:
        if not self._path:
            return
        try:
            self._path.parent.mkdir(
                parents=True, exist_ok=True
            )
            payload = {
                "devices": [
                    {
                        **d.to_dict(),
                        "secret_hash": d.secret_hash,
                    }
                    for d in self._devices.values()
                ]
            }
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            tmp.replace(self._path)
        except Exception:
            pass

    # -- identity -------------------------------------------------------

    def register(
        self,
        device_id: str,
        *,
        device_name: str = "",
        platform: str = "",
        platform_version: str = "",
        client_version: str = "",
        agent_runtime_version: str = "",
        capabilities: tuple[str, ...] | list[str] = (),
    ) -> DeviceInfo:
        """Register (or update) a device's identity metadata."""
        device_id = device_id.strip()
        if not device_id or len(device_id) > 128:
            raise ValueError("invalid device_id")
        with self._lock:
            existing = self._devices.get(device_id)
            if existing is not None:
                existing.device_name = device_name or (
                    existing.device_name
                )
                existing.platform = platform or existing.platform
                existing.platform_version = (
                    platform_version or existing.platform_version
                )
                existing.client_version = (
                    client_version or existing.client_version
                )
                existing.agent_runtime_version = (
                    agent_runtime_version
                    or existing.agent_runtime_version
                )
                if capabilities:
                    existing.capabilities = tuple(capabilities)
                self._save()
                return existing
            info = DeviceInfo(
                device_id=device_id,
                device_name=device_name,
                platform=platform,
                platform_version=platform_version,
                client_version=client_version,
                agent_runtime_version=agent_runtime_version,
                capabilities=tuple(capabilities),
            )
            self._devices[device_id] = info
            self._save()
            return info

    def get(self, device_id: str) -> DeviceInfo | None:
        with self._lock:
            return self._devices.get(device_id)

    def list(
        self, *, trust: DeviceTrust | None = None
    ) -> list[DeviceInfo]:
        with self._lock:
            devices = list(self._devices.values())
        if trust is not None:
            devices = [d for d in devices if d.trust == trust]
        return devices

    # -- trust ----------------------------------------------------------

    def set_trust(
        self, device_id: str, trust: DeviceTrust
    ) -> DeviceInfo:
        with self._lock:
            info = self._devices.get(device_id)
            if info is None:
                raise KeyError(
                    f"unknown device: {device_id}"
                )
            info.trust = trust
            self._save()
            return info

    def set_secret_hash(
        self, device_id: str, secret_hash: str
    ) -> None:
        with self._lock:
            info = self._devices.get(device_id)
            if info is None:
                raise KeyError(
                    f"unknown device: {device_id}"
                )
            info.secret_hash = secret_hash
            self._save()

    def verify_secret(
        self, device_id: str, secret: str
    ) -> bool:
        """Constant-time comparison against the stored hash."""
        import hmac

        with self._lock:
            info = self._devices.get(device_id)
            if info is None or not info.secret_hash:
                return False
            expected = info.secret_hash
        presented = hash_secret(secret)
        return hmac.compare_digest(expected, presented)

    def revoke(self, device_id: str) -> DeviceInfo:
        """Revoke a device's trust immediately."""
        with self._lock:
            info = self.set_trust(device_id, DeviceTrust.REVOKED)
            info.connection_state = ConnectionState.OFFLINE
            self._save()
            return info

    # -- presence -------------------------------------------------------

    def mark_seen(
        self,
        device_id: str,
        state: ConnectionState = ConnectionState.ONLINE,
    ) -> None:
        with self._lock:
            info = self._devices.get(device_id)
            if info is None:
                return
            info.last_seen = time.time()
            info.connection_state = state
            self._save()

    def mark_offline(self, device_id: str) -> None:
        self.mark_seen(device_id, ConnectionState.OFFLINE)

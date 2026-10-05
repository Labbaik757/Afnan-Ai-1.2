"""Sandbox policy — explicit allow/deny for generated code.

Generated skills/code only ever run inside the existing
sandbox architecture.  This module adds the policy side:
what the sandbox may touch.

* filesystem: allowed/denied path prefixes
* network: allowed/denied hosts (default: no network)
* process: allowed/denied executables
* credentials: never reachable (always denied)
* environment: allowed/denied variable names

``SandboxPolicy.check_*`` validates one proposed access;
``validate_plan`` checks a whole action plan up front.
Violations raise ``SandboxViolation`` with a structured
reason — and the policy engine turns that into a
sandbox_violation security event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SandboxViolation(Exception):
    def __init__(self, message: str, code: str = "sandbox_violation",
                 details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass
class SandboxPolicy:
    name: str = "default"
    # Filesystem: paths the sandbox may read/write.
    allowed_paths: tuple[str, ...] = ()
    denied_paths: tuple[str, ...] = (
        "/etc", "/root", "/sys", "/proc",
        "C:\\Windows", "C:\\Program Files",
    )
    # Network: default deny; explicit allow-list only.
    allowed_hosts: tuple[str, ...] = ()
    # Processes that may be spawned (by executable name).
    allowed_executables: tuple[str, ...] = ()
    # Environment variables visible inside the sandbox.
    allowed_env: tuple[str, ...] = ("PATH", "HOME", "TMPDIR", "TEMP")
    # Credentials are NEVER reachable from the sandbox.
    allow_credentials: bool = False

    def _under(self, path: str, prefixes: tuple[str, ...]) -> bool:
        try:
            resolved = str(Path(str(path)).resolve())
        except Exception:
            resolved = str(path)
        return any(
            resolved == p or resolved.startswith(p.rstrip("/") + "/")
            or resolved.startswith(p + "\\")
            for p in prefixes
        )

    def check_filesystem(
        self, path: str, *, write: bool = False
    ) -> None:
        if self._under(path, self.denied_paths):
            raise SandboxViolation(
                f"filesystem access denied: {path}",
                details={"path": str(path), "write": write},
            )
        if self.allowed_paths and not self._under(
            path, self.allowed_paths
        ):
            raise SandboxViolation(
                f"filesystem access outside allow-list: {path}",
                details={"path": str(path), "write": write},
            )

    def check_network(self, host: str) -> None:
        host = str(host or "").strip().lower()
        if host in self.allowed_hosts:
            return
        raise SandboxViolation(
            f"network access denied: {host or '(empty)'}",
            details={"host": host},
        )

    def check_process(self, executable: str) -> None:
        name = str(executable or "").strip().lower()
        allowed = {
            e.lower() for e in self.allowed_executables
        }
        if name not in allowed:
            raise SandboxViolation(
                f"process spawn denied: {executable}",
                details={"executable": str(executable)},
            )

    def check_env(self, var: str) -> None:
        if str(var).upper() not in {
            v.upper() for v in self.allowed_env
        }:
            raise SandboxViolation(
                f"env access denied: {var}",
                details={"var": str(var)},
            )

    def check_credentials(self) -> None:
        if not self.allow_credentials:
            raise SandboxViolation(
                "credential access denied inside sandbox",
                details={},
            )

    def validate_plan(
        self, plan: dict[str, Any]
    ) -> None:
        """Check a generated-code action plan up front."""
        for path in plan.get("read_paths", []) or []:
            self.check_filesystem(path)
        for path in plan.get("write_paths", []) or []:
            self.check_filesystem(path, write=True)
        for host in plan.get("hosts", []) or []:
            self.check_network(host)
        for exe in plan.get("executables", []) or []:
            self.check_process(exe)
        for var in plan.get("env_vars", []) or []:
            self.check_env(var)
        if plan.get("needs_credentials"):
            self.check_credentials()


# A conservative default: temp-dir only, no network, no
# processes, no credentials, minimal env.
DEFAULT_SANDBOX_POLICY = SandboxPolicy(
    name="default",
    allowed_paths=("/tmp",),
    allowed_hosts=(),
    allowed_executables=(),
)

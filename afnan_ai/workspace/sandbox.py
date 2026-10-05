"""Workspace execution sandbox.

Wraps the central :class:`ExecutionSandbox` with the
workspace validation pipeline:

capability → permission → resource → workspace
→ execute → capture → verify

Arbitrary model-generated host code is never executed.
Only registered tool operations and declared scripts
with explicit capabilities run, inside the sandbox.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..security.sandbox_exec import (
    ExecutionSandbox,
    SandboxLimits,
    SandboxResult,
)


@dataclass
class SandboxedOperation:
    """A declared executable operation."""

    op_id: str
    kind: str  # "script" | "command" | "tool"
    capability: str
    argv: list[str] = field(default_factory=list)
    script: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    actor: str = "agent:main"
    workspace_id: str = ""


@dataclass
class SandboxVerdict:
    allowed: bool
    reason: str = ""
    result: SandboxResult | None = None
    tool_result: Any = None
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "duration_s": round(self.duration_s, 3),
            "tool_result": (
                str(self.tool_result)[:500]
                if self.tool_result is not None
                else None
            ),
            "sandbox_result": (
                self.result.to_dict()
                if self.result is not None
                else None
            ),
        }


class WorkspaceExecutionSandbox:
    """Policy-gated execution inside a workspace."""

    def __init__(
        self,
        *,
        check_capability: (
            Callable[[str, str], bool] | None
        ) = None,
        check_permission: (
            Callable[[str, str], bool] | None
        ) = None,
        check_resources: (
            Callable[[SandboxedOperation], bool] | None
        ) = None,
        check_workspace: (
            Callable[[SandboxedOperation], bool] | None
        ) = None,
        tool_runner: (
            Callable[[str, dict[str, Any]], Any] | None
        ) = None,
        sandbox: ExecutionSandbox | None = None,
        default_limits: SandboxLimits | None = None,
    ) -> None:
        self.check_capability = check_capability
        self.check_permission = check_permission
        self.check_resources = check_resources
        self.check_workspace = check_workspace
        self.tool_runner = tool_runner
        self.sandbox = sandbox or ExecutionSandbox(
            default_limits or SandboxLimits()
        )

    def execute(
        self, op: SandboxedOperation
    ) -> SandboxVerdict:
        started = time.time()

        # 1. capability validation
        if self.check_capability is not None and not (
            self.check_capability(op.actor, op.capability)
        ):
            return SandboxVerdict(
                False,
                f"capability denied: {op.capability!r}",
                duration_s=time.time() - started,
            )
        # 2. permission validation
        if self.check_permission is not None and not (
            self.check_permission(op.actor, op.op_id)
        ):
            return SandboxVerdict(
                False,
                f"permission denied for op {op.op_id!r}",
                duration_s=time.time() - started,
            )
        # 3. resource validation
        if self.check_resources is not None and not (
            self.check_resources(op)
        ):
            return SandboxVerdict(
                False,
                "resource validation failed",
                duration_s=time.time() - started,
            )
        # 4. workspace validation
        if self.check_workspace is not None and not (
            self.check_workspace(op)
        ):
            return SandboxVerdict(
                False,
                "workspace validation failed",
                duration_s=time.time() - started,
            )
        # 5-6. execution + result capture
        if op.kind == "script":
            if not op.script.strip():
                return SandboxVerdict(
                    False, "empty script",
                    duration_s=time.time() - started,
                )
            result = self.sandbox.run_python(op.script)
            verdict = SandboxVerdict(
                result.ok,
                "executed" if result.ok else result.error,
                result=result,
                duration_s=time.time() - started,
            )
        elif op.kind == "command":
            if not op.argv:
                return SandboxVerdict(
                    False, "empty argv",
                    duration_s=time.time() - started,
                )
            result = self.sandbox.run_command(op.argv)
            verdict = SandboxVerdict(
                result.ok,
                "executed" if result.ok else result.error,
                result=result,
                duration_s=time.time() - started,
            )
        elif op.kind == "tool":
            if self.tool_runner is None:
                return SandboxVerdict(
                    False, "no tool runner bound",
                    duration_s=time.time() - started,
                )
            try:
                tool_result = self.tool_runner(
                    op.tool_name, dict(op.tool_args)
                )
            except Exception as exc:
                return SandboxVerdict(
                    False, f"tool failed: {exc}",
                    duration_s=time.time() - started,
                )
            verdict = SandboxVerdict(
                True, "tool executed",
                tool_result=tool_result,
                duration_s=time.time() - started,
            )
        else:
            return SandboxVerdict(
                False, f"unknown op kind: {op.kind!r}",
                duration_s=time.time() - started,
            )
        # 7. verification hook: fail closed on sandbox error
        if verdict.result is not None and not verdict.allowed:
            verdict.reason = (
                f"verification failed: {verdict.reason}"
            )
        return verdict

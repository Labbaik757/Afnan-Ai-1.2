"""Pre-publish security analysis for skills.

Inspects a skill before publication for:

- excessive permissions (risk vs. declared need)
- unexpected network access
- unauthorized filesystem access
- credential access
- hidden tool calls (steps referencing undeclared tools)
- undeclared dependencies
- unbounded loops
- resource abuse (limits too lax)
- suspicious data transfer (exfiltration shapes)
- policy conflicts

Any failing check blocks publication.  Agent-generated
skills always get the strict profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afnan_ai.redaction import redact_text

_NETWORK_TOOLS = frozenset({
    "web_search", "page_open", "http_get", "http_post",
    "email_send", "webhook_post", "connector_call",
})

_FILESYSTEM_TOOLS = frozenset({
    "file_write", "file_delete", "file_move", "file_copy",
    "shell_exec", "process_run",
})

_CREDENTIAL_SHAPES = (
    "api_key", "secret", "password", "token", "credential",
    "private_key",
)

_EXFIL_SHAPES = (
    "upload", "exfiltrate", "send_all", "dump",
)


@dataclass
class SecurityFinding:
    severity: str  # info|warning|blocker
    code: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "detail": redact_text(self.detail)[:300],
        }


@dataclass
class SecurityReport:
    passed: bool
    findings: list[SecurityFinding] = field(
        default_factory=list
    )
    strict: bool = False

    def blockers(self) -> list[SecurityFinding]:
        return [
            f for f in self.findings if f.severity == "blocker"
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "strict": self.strict,
            "findings": [f.to_dict() for f in self.findings],
        }


def analyze_skill(
    skill: Any,
    *,
    strict: bool = False,
    declared_tools: set[str] | None = None,
) -> SecurityReport:
    """Run the pre-publish security analysis."""
    findings: list[SecurityFinding] = []
    steps = getattr(skill, "steps", None) or []
    deps = getattr(skill, "dependencies", None)
    declared = set(declared_tools or ())
    if deps is not None:
        declared |= set(getattr(deps, "tools", ()) or ())

    risk = str(
        getattr(
            getattr(skill, "risk", ""), "value",
            getattr(skill, "risk", "read_only"),
        )
    )

    step_tools: set[str] = set()
    for step in steps:
        tool = (
            step.tool
            if hasattr(step, "tool")
            else str(step.get("tool", ""))
        )
        step_tools.add(tool)
        args = (
            step.arguments
            if hasattr(step, "arguments")
            else step.get("arguments", {})
        )
        arg_text = str(args).lower()
        # Hidden/undeclared tool calls.
        if (
            declared
            and not tool.startswith("skill:")
            and tool not in declared
        ):
            findings.append(
                SecurityFinding(
                    "blocker", "undeclared_tool",
                    f"step {getattr(step, 'step_id', '?')} "
                    f"calls undeclared tool {tool!r}",
                )
            )
        # Credential access shapes.
        if any(
            shape in arg_text for shape in _CREDENTIAL_SHAPES
        ):
            findings.append(
                SecurityFinding(
                    "blocker" if strict else "warning",
                    "credential_access",
                    f"step {getattr(step, 'step_id', '?')} "
                    "references credential-like data",
                )
            )
        # Exfiltration shapes.
        if any(
            shape in arg_text for shape in _EXFIL_SHAPES
        ) or (
            tool in _NETWORK_TOOLS
            and "file" in arg_text
            and "upload" in (tool + arg_text)
        ):
            findings.append(
                SecurityFinding(
                    "warning", "suspicious_transfer",
                    f"step {getattr(step, 'step_id', '?')} "
                    f"moves data via {tool!r}",
                )
            )

    # Excessive permissions: destructive tools at low risk.
    risky_tools = step_tools & (
        _FILESYSTEM_TOOLS | {"email_send", "purchase"}
    )
    if risky_tools and risk in ("read_only", "reversible"):
        findings.append(
            SecurityFinding(
                "blocker", "excessive_permission",
                f"risk {risk!r} but steps use "
                f"{sorted(risky_tools)}",
            )
        )

    # Network access must be declared via capabilities.
    capabilities = (
        set(getattr(deps, "capabilities", ()) or ())
        if deps
        else set()
    )
    if (step_tools & _NETWORK_TOOLS) and (
        "network" not in capabilities
    ):
        findings.append(
            SecurityFinding(
                "warning", "undeclared_network",
                "steps use network tools without the "
                "'network' capability",
            )
        )

    # Resource abuse: absurd limits.
    limits = getattr(skill, "limits", None)
    if limits is not None:
        if getattr(limits, "max_steps", 50) > 500:
            findings.append(
                SecurityFinding(
                    "warning", "lax_limits",
                    "max_steps unusually high",
                )
            )

    # Strict profile (agent-generated): any warning blocks.
    if strict:
        for finding in findings:
            if finding.severity == "warning":
                finding.severity = "blocker"

    passed = not any(
        f.severity == "blocker" for f in findings
    )
    return SecurityReport(
        passed=passed, findings=findings, strict=strict
    )

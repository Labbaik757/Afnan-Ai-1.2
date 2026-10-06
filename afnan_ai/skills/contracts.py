"""Preconditions and postconditions for skill execution.

Before execution: required tools available, connectors
connected, permissions granted, workspace ready, input
valid, dependencies available, resource budget sufficient.
A failed precondition stops execution *before* it starts.

After execution: required state change, expected output
schema, artifacts, external operation, completion
criteria.  A failed postcondition marks the skill
UNCERTAIN or FAILED — never silently successful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afnan_ai.redaction import redact_text


@dataclass
class ContractCheck:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": redact_text(self.detail)[:300],
        }


@dataclass
class ContractReport:
    passed: bool
    checks: list[ContractCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
        }

    def failures(self) -> list[str]:
        return [
            c.name for c in self.checks if not c.passed
        ]


def check_preconditions(
    skill: Any,
    arguments: dict[str, Any],
    *,
    tool_registry: Any = None,
    connector_status: dict[str, bool] | None = None,
    workspace_ready: bool = True,
    resource_budget_ok: bool = True,
) -> ContractReport:
    """Verify everything a skill needs *before* it runs."""
    checks: list[ContractCheck] = []

    # Input schema validation.
    schema = getattr(skill, "input_schema", None) or {}
    required = schema.get("required", [])
    missing = [
        key
        for key in required
        if key not in (arguments or {})
    ]
    checks.append(
        ContractCheck(
            "input_valid",
            not missing,
            (
                "missing required inputs: "
                + ", ".join(missing)
                if missing
                else "inputs valid"
            ),
        )
    )

    # Required tools exist in the registry.
    deps = getattr(skill, "dependencies", None)
    needed_tools = (
        list(getattr(deps, "tools", ()) or ())
        if deps
        else []
    )
    if tool_registry is not None and needed_tools:
        available = set()
        try:
            available = set(
                tool_registry.list_tools()
                if hasattr(tool_registry, "list_tools")
                else []
            )
        except Exception:
            available = set()
        absent = [t for t in needed_tools if t not in available]
        checks.append(
            ContractCheck(
                "tools_available",
                not absent,
                (
                    "missing tools: " + ", ".join(absent)
                    if absent
                    else "all tools available"
                ),
            )
        )
    else:
        checks.append(
            ContractCheck("tools_available", True, "skipped")
        )

    # Required connectors connected.
    needed_connectors = (
        list(getattr(deps, "connectors", ()) or ())
        if deps
        else []
    )
    if needed_connectors:
        status = connector_status or {}
        down = [
            c for c in needed_connectors if not status.get(c)
        ]
        checks.append(
            ContractCheck(
                "connectors_connected",
                not down,
                (
                    "connectors not connected: "
                    + ", ".join(down)
                    if down
                    else "connectors ready"
                ),
            )
        )

    checks.append(
        ContractCheck(
            "workspace_ready",
            bool(workspace_ready),
            "workspace ready"
            if workspace_ready
            else "workspace not ready",
        )
    )
    checks.append(
        ContractCheck(
            "resource_budget",
            bool(resource_budget_ok),
            "budget sufficient"
            if resource_budget_ok
            else "resource budget exhausted",
        )
    )

    passed = all(c.passed for c in checks)
    return ContractReport(passed=passed, checks=checks)


def check_postconditions(
    skill: Any,
    result: dict[str, Any],
    *,
    expected_artifacts: list[str] | None = None,
) -> ContractReport:
    """Verify the skill actually achieved its contract."""
    checks: list[ContractCheck] = []

    # Output schema: required keys present.
    schema = getattr(skill, "output_schema", None) or {}
    required = schema.get("required", [])
    output = (result or {}).get("output", result or {})
    missing = [
        key
        for key in required
        if not isinstance(output, dict) or key not in output
    ]
    checks.append(
        ContractCheck(
            "output_schema",
            not missing,
            (
                "missing output keys: " + ", ".join(missing)
                if missing
                else "output schema satisfied"
            ),
        )
    )

    # Declared verification criteria met.
    criteria = getattr(
        skill, "verification_criteria", None
    ) or []
    evidence = str(result or {}).lower()
    unmet = [
        c
        for c in criteria
        if c[:40].lower() not in evidence
    ]
    checks.append(
        ContractCheck(
            "verification_criteria",
            not unmet,
            (
                "unmet criteria: " + "; ".join(unmet[:3])
                if unmet
                else "criteria evidenced"
            ),
        )
    )

    # Expected artifacts produced.
    if expected_artifacts:
        produced = set((result or {}).get("artifacts", []))
        absent = [
            a for a in expected_artifacts if a not in produced
        ]
        checks.append(
            ContractCheck(
                "artifacts",
                not absent,
                (
                    "missing artifacts: " + ", ".join(absent)
                    if absent
                    else "artifacts present"
                ),
            )
        )

    passed = all(c.passed for c in checks)
    return ContractReport(passed=passed, checks=checks)

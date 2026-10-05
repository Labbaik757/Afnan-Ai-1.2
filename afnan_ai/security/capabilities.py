"""Capability-Based Authorization.

Every executable capability is explicitly defined —
never implied, never inherited silently:

* capability_id, name, description, category
* risk_level, required_permissions
* allowed_resources, allowed_operations
* approval_requirement, expiration, owner/policy scope

The agent starts with NOTHING.  Capabilities are
granted explicitly (profile, temporary grant, or owner
approval) and every grant is audited.  There is no
unrestricted default, no master switch, no bypass flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from afnan_ai.security.models import RiskLevel


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# Legacy capability names (from the pre-capability permission
# system) mapped to their canonical capability ids.  Grants
# and checks normalize through this so "file.*" and
# "filesystem.*" mean the same thing — there is exactly one
# namespace, not two parallel ones.
_LEGACY_PREFIXES: dict[str, str] = {
    "file": "filesystem",
}


def canonical_capability_id(capability_id: str) -> str:
    """Normalize a legacy capability id to its canonical form."""
    text = str(capability_id or "").strip().lower()
    if not text:
        return ""
    if "." in text:
        prefix, _, rest = text.partition(".")
        canonical = _LEGACY_PREFIXES.get(prefix)
        if canonical:
            return f"{canonical}.{rest}"
        return text
    canonical = _LEGACY_PREFIXES.get(text)
    return canonical or text


@dataclass(frozen=True)
class Capability:
    capability_id: str
    name: str
    description: str = ""
    category: str = "general"
    risk_level: RiskLevel = RiskLevel.SENSITIVE
    required_permissions: tuple[str, ...] = ()
    allowed_resources: tuple[str, ...] = ()
    allowed_operations: tuple[str, ...] = ()
    approval_requirement: str = "none"  # none|human|owner
    expiration_s: int | None = None
    owner_scope: str = "agent"  # agent|owner|system

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "risk_level": self.risk_level.value,
            "required_permissions": list(
                self.required_permissions
            ),
            "allowed_resources": list(self.allowed_resources),
            "allowed_operations": list(
                self.allowed_operations
            ),
            "approval_requirement": self.approval_requirement,
            "expiration_s": self.expiration_s,
            "owner_scope": self.owner_scope,
        }


# -- built-in catalog -------------------------------------------------------
# Explicit, reviewable, versioned alongside the code.
BUILTIN_CAPABILITIES: tuple[Capability, ...] = (
    Capability("browser.read", "Browser read",
               "Read page content, observe tabs",
               category="browser", risk_level=RiskLevel.READ_ONLY,
               allowed_operations=("read", "observe", "list")),
    Capability("browser.navigate", "Browser navigate",
               "Navigate to URLs within allowed domains",
               category="browser", risk_level=RiskLevel.LOW_RISK_WRITE,
               allowed_operations=("navigate",),
               approval_requirement="none"),
    Capability("browser.click", "Browser click",
               "Click/type/submit on pages",
               category="browser", risk_level=RiskLevel.SENSITIVE,
               allowed_operations=("click", "type", "submit"),
               approval_requirement="human"),
    Capability("browser.download", "Browser download",
               "Download files to the approved directory",
               category="browser", risk_level=RiskLevel.LOW_RISK_WRITE,
               allowed_operations=("download",),
               allowed_resources=("download_dir",)),
    Capability("computer.input", "Computer input",
               "Keyboard/mouse input to allowed applications",
               category="computer", risk_level=RiskLevel.SENSITIVE,
               allowed_operations=("click", "type", "key"),
               approval_requirement="human"),
    Capability("computer.observe", "Computer observe",
               "Read screen state, locate elements",
               category="computer", risk_level=RiskLevel.READ_ONLY,
               allowed_operations=("observe", "locate")),
    Capability("filesystem.read", "Filesystem read",
               "Read files inside the allowed roots",
               category="filesystem", risk_level=RiskLevel.READ_ONLY,
               allowed_operations=("read", "list")),
    Capability("filesystem.write", "Filesystem write",
               "Create/modify files inside the allowed roots",
               category="filesystem",
               risk_level=RiskLevel.LOW_RISK_WRITE,
               allowed_operations=("write", "create", "mkdir")),
    Capability("filesystem.delete", "Filesystem delete",
               "Delete files inside the allowed roots",
               category="filesystem",
               risk_level=RiskLevel.IRREVERSIBLE,
               allowed_operations=("delete",),
               approval_requirement="human"),
    Capability("connector.email.read", "Email read",
               "Read mailbox content",
               category="connector", risk_level=RiskLevel.READ_ONLY,
               allowed_operations=("read", "list", "search")),
    Capability("connector.email.send", "Email send",
               "Send email messages",
               category="connector", risk_level=RiskLevel.SENSITIVE,
               allowed_operations=("send",),
               approval_requirement="human"),
    Capability("connector.calendar.write", "Calendar write",
               "Create/modify calendar events",
               category="connector",
               risk_level=RiskLevel.LOW_RISK_WRITE,
               allowed_operations=("create", "update")),
    Capability("connector.bulk_delete", "Bulk delete",
               "Delete many records at once",
               category="connector",
               risk_level=RiskLevel.IRREVERSIBLE,
               allowed_operations=("bulk_delete",),
               approval_requirement="human"),
    Capability("skill.execute", "Skill execute",
               "Execute a registered skill",
               category="skill", risk_level=RiskLevel.SENSITIVE,
               allowed_operations=("execute",),
               approval_requirement="human"),
    Capability("code.sandbox.execute", "Sandboxed code",
               "Run generated code inside the execution sandbox",
               category="code",
               risk_level=RiskLevel.LOW_RISK_WRITE,
               allowed_operations=("execute",),
               allowed_resources=("sandbox_fs",)),
    Capability("artifact.create", "Artifact create",
               "Create versioned deliverables",
               category="artifact",
               risk_level=RiskLevel.LOW_RISK_WRITE,
               allowed_operations=("create", "update")),
    Capability("artifact.delete", "Artifact delete",
               "Delete versioned deliverables",
               category="artifact",
               risk_level=RiskLevel.IRREVERSIBLE,
               allowed_operations=("delete",),
               approval_requirement="human"),
    Capability("memory.write", "Memory write",
               "Write long-term memories",
               category="memory",
               risk_level=RiskLevel.LOW_RISK_WRITE,
               allowed_operations=("write",)),
    Capability("subagent.spawn", "Spawn subagent",
               "Create isolated subagents",
               category="subagent",
               risk_level=RiskLevel.LOW_RISK_WRITE,
               allowed_operations=("spawn",)),
)


class CapabilityManager:
    """Explicit capability registry.  Starts empty — the
    agent has no capabilities until they are granted."""

    def __init__(
        self,
        definitions: tuple[Capability, ...]
        | None = None,
    ) -> None:
        self._definitions: dict[str, Capability] = {}
        for cap in definitions or BUILTIN_CAPABILITIES:
            self._definitions[cap.capability_id] = cap

    def define(self, capability: Capability) -> None:
        if capability.capability_id in self._definitions:
            raise ValueError(
                f"capability {capability.capability_id!r} "
                "already defined"
            )
        self._definitions[capability.capability_id] = capability

    def get(self, capability_id: str) -> Capability | None:
        return self._definitions.get(
            str(capability_id or "").strip().lower()
        )

    def require(self, capability_id: str) -> Capability:
        cap = self.get(capability_id)
        if cap is None:
            raise KeyError(
                f"unknown capability {capability_id!r}: "
                "capabilities must be explicitly defined, "
                "never invented"
            )
        return cap

    def all_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))

    def by_category(
        self, category: str
    ) -> tuple[Capability, ...]:
        return tuple(
            c for c in self._definitions.values()
            if c.category == category
        )

    def by_risk(
        self, risk: RiskLevel
    ) -> tuple[Capability, ...]:
        return tuple(
            c for c in self._definitions.values()
            if c.risk_level is risk
        )

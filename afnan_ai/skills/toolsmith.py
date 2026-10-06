"""Dynamic tool generation pipeline (architecture).

When a required capability has no registered tool, the
ToolBuilder proposes one:

    capability specification → tool schema →
    implementation strategy → sandbox → tests →
    security validation → registration

Rules: a generated tool never gets privileged capability
automatically; it starts at the lowest risk that fits,
needs the same approval gates as any tool, and the model
never executes arbitrary host code — implementations are
compositions of existing primitives or sandboxed snippets
that pass the full pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from afnan_ai.redaction import redact_text


class ToolProposalStatus(str, Enum):
    DRAFT = "draft"
    SANDBOX_TESTED = "sandbox_tested"
    SECURITY_REVIEWED = "security_reviewed"
    REGISTERED = "registered"
    REJECTED = "rejected"


@dataclass
class ToolProposal:
    """A proposed new atomic capability."""

    tool_id: str
    description: str
    schema: dict[str, Any] = field(default_factory=dict)
    risk: str = "read_only"  # starts low; never privileged
    composed_of: list[str] = field(
        default_factory=list
    )  # existing tools it wraps
    status: ToolProposalStatus = ToolProposalStatus.DRAFT
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = ToolProposalStatus(self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "description": redact_text(self.description)[:300],
            "schema": dict(self.schema),
            "risk": self.risk,
            "composed_of": list(self.composed_of),
            "status": self.status.value,
            "rejection_reason": redact_text(
                self.rejection_reason
            )[:300],
        }


class ToolBuilder:
    """Propose tools as compositions of existing primitives."""

    # Tools a proposal may never wrap (privilege floor).
    _FORBIDDEN_BASE = frozenset({
        "shell_exec", "process_run", "eval", "exec_code",
    })

    def propose(
        self,
        tool_id: str,
        description: str,
        *,
        schema: dict[str, Any] | None = None,
        composed_of: list[str] | None = None,
        available_tools: set[str] | None = None,
    ) -> ToolProposal:
        composed = list(composed_of or [])
        proposal = ToolProposal(
            tool_id=tool_id,
            description=description,
            schema=dict(schema or {}),
            composed_of=composed,
        )
        forbidden = set(composed) & self._FORBIDDEN_BASE
        if forbidden:
            proposal.status = ToolProposalStatus.REJECTED
            proposal.rejection_reason = (
                "cannot wrap privileged tools: "
                + ", ".join(sorted(forbidden))
            )
            return proposal
        if available_tools is not None:
            missing = [
                t for t in composed if t not in available_tools
            ]
            if missing:
                proposal.status = (
                    ToolProposalStatus.REJECTED
                )
                proposal.rejection_reason = (
                    "unknown base tools: "
                    + ", ".join(missing)
                )
                return proposal
        return proposal

    def mark_sandbox_tested(
        self, proposal: ToolProposal, *, passed: bool
    ) -> ToolProposal:
        proposal.status = (
            ToolProposalStatus.SANDBOX_TESTED
            if passed
            else ToolProposalStatus.REJECTED
        )
        if not passed:
            proposal.rejection_reason = "sandbox tests failed"
        return proposal

    def mark_security_reviewed(
        self, proposal: ToolProposal, *, passed: bool
    ) -> ToolProposal:
        if proposal.status != ToolProposalStatus.SANDBOX_TESTED:
            proposal.status = ToolProposalStatus.REJECTED
            proposal.rejection_reason = (
                "security review requires sandbox tests first"
            )
            return proposal
        proposal.status = (
            ToolProposalStatus.SECURITY_REVIEWED
            if passed
            else ToolProposalStatus.REJECTED
        )
        if not passed:
            proposal.rejection_reason = (
                "security review failed"
            )
        return proposal

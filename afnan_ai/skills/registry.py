"""SkillRegistry — dynamic skill lifecycle management.

Responsibilities (and only these):

* register / discover / update / version / disable skills,
* give the Agent structured skill descriptions,
* validate definitions (schemas, steps, dependencies, risk,
  injection and secret scans) before anything goes live,
* keep an auditable version history with rollback,
* bridge skills into the ToolRegistry via the SkillTool
  *adapter* — a Skill is never a Tool itself.

Tool vs Skill, clearly separated:

* **Tool** — one atomic capability (``Tool`` subclass,
  ``ToolRegistry``).  Knows nothing about skills.
* **Skill** — a named, versioned composition of tools
  (``Skill`` dataclass, ``SkillRegistry``).  Knows nothing
  about execution; the ``SkillExecutor`` runs it through
  the normal ``ToolRegistry.execute`` path, so Planner /
  Executor / Verifier / Recovery / AgentLoop all keep
  working unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from afnan_ai.log_config import get_logger
from afnan_ai.persistence import JsonFileStore
from afnan_ai.redaction import redact_text
from afnan_ai.skills.models import (
    Skill,
    SkillRisk,
    SkillStatus,
    SkillStep,
)
from afnan_ai.skills.risk import (
    effective_risk,
    needs_approval,
    resolve_sub_skill_risks,
    risk_report,
)
from afnan_ai.skills.sandbox import (
    SkillValidation,
    ValidationIssue,
)

logger = get_logger(__name__)

_TEMPLATE_RE = re.compile(r"\{\{\s*steps\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_.]+)\s*\}\}")


class SkillError(Exception):
    """Structured skill-registry failure."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "skill_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": dict(self.details),
        }


def _scan_injection() -> Callable[[str], list]:
    from afnan_ai.agent_loop import scan_for_injection

    return scan_for_injection


def _secret_like(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "password", "passwd", "api_key", "apikey", "secret",
        "bearer ", "token=", "aws_secret",
    )
    return any(m in lowered for m in markers)


class SkillRegistry:
    """Versioned store of reusable skills."""

    def __init__(
        self,
        tool_registry: Any | None = None,
        *,
        audit_path: str | Path | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        # skill_id -> {version: Skill}
        self._versions: dict[str, dict[str, Skill]] = {}
        # skill_id -> active version
        self._active: dict[str, str] = {}
        self._audit_path = (
            Path(str(audit_path)).expanduser()
            if audit_path else None
        )

    # -- lifecycle ------------------------------------------------------
    def register(
        self, skill: Skill, *, validate: bool = True
    ) -> Skill:
        """Register a skill (new id, or a new version of an
        existing id).  Drafts cannot be registered."""
        if not isinstance(skill, Skill):
            raise SkillError(
                "only Skill objects can be registered",
                code="invalid_skill",
            )
        if skill.status is SkillStatus.DRAFT:
            raise SkillError(
                f"skill {skill.skill_id!r} is a draft: validate "
                "and finalize it before registering",
                code="draft_not_allowed",
            )
        if validate:
            validation = self.validate(skill)
            if not validation.ok:
                raise SkillError(
                    "skill failed validation: "
                    + "; ".join(
                        i.message for i in validation.issues[:3]
                    ),
                    code="validation_failed",
                    details=validation.to_dict(),
                )
        versions = self._versions.setdefault(skill.skill_id, {})
        if skill.version in versions:
            raise SkillError(
                f"skill {skill.skill_id!r} version "
                f"{skill.version} already exists",
                code="duplicate_version",
            )
        versions[skill.version] = skill
        # Newest registered version becomes active.
        self._active[skill.skill_id] = skill.version
        self._audit("registered", skill)
        logger.info(
            "skill registered: %s v%s", skill.skill_id,
            skill.version,
        )
        return skill

    def update_skill(self, skill: Skill) -> Skill:
        """Validate a new version in the sandbox *before* it
        replaces the active one.  On validation failure the
        previous stable version stays active."""
        if skill.skill_id not in self._versions:
            raise SkillError(
                f"unknown skill {skill.skill_id!r}",
                code="unknown_skill",
            )
        validation = self.validate(skill)
        if not validation.ok:
            self._audit("update_rejected", skill)
            raise SkillError(
                "new version failed validation; previous "
                "stable version kept: "
                + "; ".join(
                    i.message for i in validation.issues[:3]
                ),
                code="validation_failed",
                details=validation.to_dict(),
            )
        return self.register(skill, validate=False)

    def rollback(
        self, skill_id: str, version: str
    ) -> Skill:
        """Restore a previous stable version as active."""
        versions = self._versions.get(skill_id)
        if not versions or version not in versions:
            raise SkillError(
                f"no such version {version!r} of skill "
                f"{skill_id!r}",
                code="unknown_version",
            )
        self._active[skill_id] = version
        skill = versions[version]
        self._audit("rolled_back", skill)
        logger.info(
            "skill rolled back: %s -> v%s", skill_id, version
        )
        return skill

    def disable(self, skill_id: str) -> None:
        skill = self.get(skill_id)
        skill.status = SkillStatus.DISABLED
        self._audit("disabled", skill)

    def enable(self, skill_id: str) -> None:
        skill = self.get(skill_id)
        skill.status = SkillStatus.ACTIVE
        self._audit("enabled", skill)

    # -- lookup -----------------------------------------------------------
    def get(
        self, skill_id: str, version: str | None = None
    ) -> Skill:
        versions = self._versions.get(skill_id)
        if not versions:
            raise SkillError(
                f"unknown skill {skill_id!r}",
                code="unknown_skill",
            )
        if version is None:
            version = self._active.get(skill_id)
        skill = versions.get(version or "")
        if skill is None:
            raise SkillError(
                f"no such version {version!r} of skill "
                f"{skill_id!r}",
                code="unknown_version",
            )
        return skill

    def get_or_none(
        self, skill_id: str, version: str | None = None
    ) -> Skill | None:
        try:
            return self.get(skill_id, version)
        except SkillError:
            return None

    def versions(self, skill_id: str) -> list[str]:
        return sorted(
            self._versions.get(skill_id, {}).keys()
        )

    def list_skills(
        self, *, status: SkillStatus | None = None,
        include_disabled: bool = False,
    ) -> list[Skill]:
        out = []
        for skill_id, version in self._active.items():
            skill = self._versions[skill_id][version]
            if not include_disabled and (
                skill.status is SkillStatus.DISABLED
            ):
                continue
            if status is not None and skill.status is not status:
                continue
            out.append(skill)
        return sorted(out, key=lambda s: s.skill_id)

    def skill_ids(self) -> list[str]:
        return sorted(self._versions.keys())

    # -- agent-facing descriptions ------------------------------------------
    def describe_for_agent(self) -> list[dict[str, Any]]:
        """Structured skill descriptions for the Planner."""
        return [
            {
                "skill_id": skill.skill_id,
                "tool_name": skill.tool_name,
                "name": skill.name,
                "description": skill.description,
                "version": skill.version,
                "risk": effective_risk(skill).value,
                "input_schema": dict(skill.input_schema),
                "dependencies": (
                    skill.dependencies.to_dict()
                ),
            }
            for skill in self.list_skills()
        ]

    def context_section(self, *, max_chars: int = 800) -> str:
        """Compact skill list for the loop's decision context."""
        skills = self.list_skills()
        if not skills:
            return ""
        lines = ["Available skills (use as skill_<id> tools):"]
        for skill in skills:
            lines.append(
                f"- {skill.tool_name}: {skill.description[:100]} "
                f"[risk: {effective_risk(skill).value}]"
            )
        text = "\n".join(lines)
        return text[:max_chars]

    # -- dependencies ----------------------------------------------------------
    def check_dependencies(
        self, skill: Skill
    ) -> list[dict[str, str]]:
        """Missing dependencies; empty means executable."""
        missing: list[dict[str, str]] = []
        tools = self._tool_registry
        for name in skill.dependencies.tools:
            if tools is None or not tools.has(name):
                missing.append(
                    {"kind": "tool", "name": name}
                )
        for name in skill.dependencies.skills:
            sub = self.get_or_none(name)
            if sub is None or (
                sub.status is SkillStatus.DISABLED
            ):
                missing.append(
                    {"kind": "skill", "name": name}
                )
        # connectors / capabilities are informational here;
        # the executor re-checks against live registries.
        return missing

    # -- validation ---------------------------------------------------------------
    def validate(self, skill: Skill) -> SkillValidation:
        """Full sandbox validation (static — no code runs)."""
        issues: list[ValidationIssue] = []
        scan = _scan_injection()

        def fail(code: str, message: str,
                 step_id: str | None = None) -> None:
            issues.append(
                ValidationIssue(code, message, step_id)
            )

        if not skill.name.strip():
            fail("invalid_metadata", "skill name is required")
        if not skill.description.strip():
            fail(
                "invalid_metadata",
                "skill description is required",
            )
        for label, text in (
            ("description", skill.description),
            ("name", skill.name),
        ):
            if scan(text):
                fail(
                    "prompt_injection",
                    f"instruction-like content in skill {label}; "
                    "external instructions are never promoted "
                    "into skill definitions",
                )
        blob = " ".join([
            skill.description,
            *[s.description for s in skill.steps],
            *[str(v) for s in skill.steps
              for v in s.arguments.values()],
        ])
        if _secret_like(blob):
            fail(
                "secret_content",
                "secret-like material in skill definition; "
                "credentials never belong in a skill",
            )
        if not skill.steps:
            fail("no_steps", "skill has no steps")
        tools = self._tool_registry
        seen: set[str] = set()
        for step in skill.steps:
            if not step.step_id or not step.tool:
                fail(
                    "invalid_step",
                    "step needs step_id and tool",
                    step.step_id or None,
                )
                continue
            if step.tool.startswith("skill:"):
                sub_id = step.tool[len("skill:"):]
                if sub_id == skill.skill_id:
                    fail(
                        "circular_skill",
                        "skill cannot invoke itself",
                        step.step_id,
                    )
                elif self.get_or_none(sub_id) is None:
                    fail(
                        "unknown_step_target",
                        f"unknown skill {sub_id!r}",
                        step.step_id,
                    )
            elif tools is None:
                fail(
                    "no_tool_registry",
                    "cannot resolve step tools without a "
                    "tool registry",
                    step.step_id,
                )
            elif not tools.has(step.tool):
                fail(
                    "unknown_step_target",
                    f"unknown tool {step.tool!r}",
                    step.step_id,
                )
            else:
                tool = tools.get(step.tool)
                try:
                    tool.validate_arguments(
                        dict(step.arguments)
                    )
                except Exception as e:  # noqa: BLE001
                    fail(
                        "invalid_step_arguments",
                        f"step {step.step_id}: {e}"[:200],
                        step.step_id,
                    )
            # Template references must point at earlier steps.
            for match in _TEMPLATE_RE.finditer(
                str(step.arguments)
            ):
                ref = match.group(1)
                if ref not in seen:
                    fail(
                        "bad_template_ref",
                        f"step {step.step_id} references unknown "
                        f"or later step {ref!r}",
                        step.step_id,
                    )
            seen.add(step.step_id)
        for missing in self.check_dependencies(skill):
            fail(
                "missing_dependency",
                f"missing {missing['kind']}: {missing['name']}",
            )
        report = risk_report(
            skill,
            resolve_sub_skill_risks(skill, self.get_or_none),
        )
        if report["escalated"]:
            # Not a failure — the effective risk is raised
            # automatically; recorded for audit.
            logger.info(
                "skill %s risk escalated %s -> %s",
                skill.skill_id, report["declared_risk"],
                report["effective_risk"],
            )
        return SkillValidation(
            ok=not issues, issues=issues
        )

    # -- ToolRegistry bridge -----------------------------------------------------
    def register_skill_tools(
        self, tool_registry: Any, executor: Any | None = None
    ) -> int:
        """Expose active skills as ``skill_<id>`` tools.

        The adapter is one-way: skills stay skills; the tool
        is just how the Planner invokes them through the
        normal Executor path.  Pass the shared executor so
        the human approver is preserved.
        """
        from afnan_ai.skills.executor import SkillTool

        count = 0
        for skill in self.list_skills():
            adapter = SkillTool(skill, self, executor)
            try:
                tool_registry.register(adapter, replace=True)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "could not expose skill %s as tool: %s",
                    skill.skill_id, e,
                )
                continue
            count += 1
        return count

    # -- audit ----------------------------------------------------------------------
    def _audit(self, event: str, skill: Skill) -> None:
        if self._audit_path is None:
            return
        try:
            store = JsonFileStore(str(self._audit_path))
            data = store.read({"events": []})
            events = data.get("events", [])
            events.append({
                "event": event,
                "skill_id": skill.skill_id,
                "version": skill.version,
                "risk": effective_risk(skill).value,
                "at": skill.created_at,
            })
            store.write({"events": events[-500:]})
        except Exception as e:  # noqa: BLE001 - never breaks
            logger.warning("skill audit write failed: %s", e)

    def _redacted(self, text: str) -> str:
        return redact_text(text)[:200]

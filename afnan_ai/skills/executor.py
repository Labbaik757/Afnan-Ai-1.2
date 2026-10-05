"""Skill execution — through the normal tool path.

A skill never gets its own orchestration loop.  Instead:

* ``SkillExecutor.execute`` runs a skill's steps one by one
  through ``ToolRegistry.execute`` — the same validation,
  redaction and result recording every tool gets.
* ``SkillTool`` is a thin ``Tool`` *adapter* that exposes an
  active skill as a ``skill_<id>`` tool, so the Planner,
  Executor, Verifier, RecoveryManager and AgentLoop handle
  skills with zero special cases.

``SkillExecutionError`` subclasses ``ToolExecutionError``,
so approval codes (``approval_required`` / ``approval_denied``)
pause the AgentLoop resumably exactly like any sensitive
tool, and every other failure becomes a structured
``ToolResult`` — a failed skill is isolated, never fatal.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from afnan_ai.log_config import get_logger
from afnan_ai.redaction import redact_text, redact_value
from afnan_ai.skills.models import Skill, SkillStatus
from afnan_ai.skills.registry import SkillRegistry
from afnan_ai.skills.risk import (
    effective_risk,
    needs_approval,
    resolve_sub_skill_risks,
)
from afnan_ai.tools.base import (
    Tool,
    ToolErrorCode,
    ToolExecutionError,
)

logger = get_logger(__name__)

_TEMPLATE_RE = re.compile(
    r"\{\{\s*(?:steps\.([A-Za-z0-9_-]+)((?:\.[A-Za-z0-9_-]+)*)"
    r"|input\.([A-Za-z0-9_-]+))\s*\}\}"
)

_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float))
    and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int)
    and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
}


class SkillExecutionError(ToolExecutionError):
    """A skill failed — structured, isolated, never fatal."""


def _validate_schema(
    arguments: dict[str, Any],
    schema: dict[str, Any],
    *,
    what: str,
) -> dict[str, Any]:
    args = dict(arguments or {})
    if not schema:
        return args
    properties = schema.get("properties", {})
    for key in schema.get("required", []):
        if args.get(key) is None or (
            isinstance(args.get(key), str)
            and not args[key].strip()
        ):
            raise SkillExecutionError(
                f"{what} is missing required argument {key!r}",
                code=ToolErrorCode.MISSING_ARGUMENTS,
                tool=what,
                details={"missing": [key]},
            )
    for key, spec in properties.items():
        if key not in args or args[key] is None:
            continue
        expected = (spec or {}).get("type")
        check = _TYPE_CHECKS.get(expected or "")
        if check is not None and not check(args[key]):
            raise SkillExecutionError(
                f"{what} argument {key!r} must be {expected}",
                code=ToolErrorCode.INVALID_ARGUMENTS,
                tool=what,
                details={"argument": key},
            )
    return args


def _resolve_templates(
    value: Any, outputs: dict[str, Any],
    skill_input: dict[str, Any] | None = None,
) -> Any:
    """Fill ``{{steps.<id>.<key>}}`` and ``{{input.<key>}}``
    from earlier outputs and the skill's input arguments."""
    skill_input = skill_input or {}

    def _lookup(match: re.Match) -> Any:
        step_id, dotted, input_key = (
            match.group(1), match.group(2), match.group(3)
        )
        if input_key:
            return skill_input.get(input_key)
        current: Any = outputs.get(step_id)
        parts = [
            p for p in (dotted or "").strip(".").split(".")
            if p
        ]
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    if isinstance(value, str):
        full = _TEMPLATE_RE.fullmatch(value)
        if full:
            # Whole-string template keeps the value's type.
            return _lookup(full)
        return _TEMPLATE_RE.sub(
            lambda m: (
                "" if _lookup(m) is None else str(_lookup(m))
            ),
            value,
        )
    if isinstance(value, dict):
        return {
            key: _resolve_templates(item, outputs, skill_input)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_templates(item, outputs, skill_input)
            for item in value
        ]
    return value


class SkillExecutor:
    """Run skills through the ToolRegistry (no own loop)."""

    def __init__(
        self,
        skill_registry: SkillRegistry,
        tool_registry: Any,
        *,
        approver: Callable[[str], bool] | None = None,
        audit_path: str | Path | None = None,
        max_depth: int = 3,
    ) -> None:
        self.skills = skill_registry
        self.tools = tool_registry
        self.approver = approver
        self.max_depth = max_depth
        self._audit_path = (
            Path(str(audit_path)).expanduser()
            if audit_path else None
        )

    def set_approver(
        self, approver: Callable[[str], bool] | None
    ) -> None:
        self.approver = approver

    def execute(
        self,
        skill_id: str,
        arguments: dict[str, Any] | None = None,
        *,
        _depth: int = 0,
        _stack: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Run a skill; returns the (output-schema-validated)
        result dict.  Raises SkillExecutionError on failure."""
        if _depth > self.max_depth:
            raise SkillExecutionError(
                f"skill {skill_id!r} exceeds max nesting depth "
                f"({self.max_depth})",
                code="max_depth_exceeded",
                tool=f"skill_{skill_id}",
            )
        if skill_id in _stack:
            raise SkillExecutionError(
                f"circular skill invocation: "
                f"{' -> '.join([*_stack, skill_id])}",
                code="circular_skill",
                tool=f"skill_{skill_id}",
            )
        skill = self.skills.get_or_none(skill_id)
        if skill is None:
            raise SkillExecutionError(
                f"unknown skill {skill_id!r}",
                code="unknown_skill",
                tool=f"skill_{skill_id}",
            )
        if skill.status is SkillStatus.DISABLED:
            raise SkillExecutionError(
                f"skill {skill_id!r} is disabled",
                code="skill_disabled",
                tool=skill.tool_name,
            )
        args = _validate_schema(
            arguments or {}, skill.input_schema,
            what=skill.tool_name,
        )
        missing = self.skills.check_dependencies(skill)
        if missing:
            raise SkillExecutionError(
                "skill "
                f"{skill_id!r} has missing dependencies: "
                + ", ".join(
                    f"{m['kind']}:{m['name']}" for m in missing
                ),
                code="missing_dependency",
                tool=skill.tool_name,
                details={"missing": missing},
            )
        risk = effective_risk(
            skill,
            resolve_sub_skill_risks(
                skill, self.skills.get_or_none
            ),
        )
        if needs_approval(risk):
            self._require_approval(skill, risk.value, args)
        outputs: dict[str, Any] = {}
        step_records: list[dict[str, Any]] = []
        try:
            for step in skill.steps:
                step_args = _resolve_templates(
                    step.arguments, outputs, args
                )
                if step.tool.startswith("skill:"):
                    sub_id = step.tool[len("skill:"):]
                    sub_result = self.execute(
                        sub_id, step_args if isinstance(
                            step_args, dict) else {},
                        _depth=_depth + 1,
                        _stack=(*_stack, skill_id),
                    )
                    output = sub_result.get("result")
                else:
                    result = self.tools.execute(
                        step.tool, step_args
                    )
                    output = result.output
                    if not result.success:
                        error = result.error
                        message = (
                            error.get("message", "failed")
                            if isinstance(error, dict)
                            else str(error)
                        )
                        raise SkillExecutionError(
                            f"skill {skill_id!r} step "
                            f"{step.step_id} ({step.tool}) "
                            f"failed: {message}"[:300],
                            code="skill_step_failed",
                            tool=skill.tool_name,
                            details={
                                "step_id": step.step_id,
                                "tool": step.tool,
                                "error": message[:200],
                            },
                        )
                outputs[step.step_id] = output
                step_records.append({
                    "step_id": step.step_id,
                    "tool": step.tool,
                    "ok": True,
                })
        except SkillExecutionError as e:
            self._audit_execution(
                skill, args, "failed", step_records,
                error=str(e)[:200],
            )
            raise
        final = {"result": outputs, "skill_id": skill_id}
        if skill.output_schema:
            _validate_schema(
                {"result": outputs}, skill.output_schema,
                what=skill.tool_name,
            )
        self._audit_execution(
            skill, args, "completed", step_records
        )
        return final

    def _require_approval(
        self, skill: Skill, risk: str, args: dict
    ) -> None:
        request = (
            f"Skill '{skill.skill_id}' v{skill.version} "
            f"(risk: {risk}) wants to run "
            f"{len(skill.steps)} step(s)."
        )
        if self.approver is None:
            raise SkillExecutionError(
                f"sensitive skill {skill.skill_id!r} needs "
                "human approval; no approver configured",
                code="approval_required",
                tool=skill.tool_name,
                details={"risk": risk, "resumable": True},
            )
        try:
            allowed = self.approver(request)
        except Exception:
            allowed = False
        if not allowed:
            raise SkillExecutionError(
                f"human denied skill {skill.skill_id!r}",
                code="approval_denied",
                tool=skill.tool_name,
                details={"risk": risk},
            )

    def _audit_execution(
        self, skill: Skill, args: dict, status: str,
        steps: list[dict], error: str | None = None,
    ) -> None:
        if self._audit_path is None:
            return
        try:
            from afnan_ai.persistence import JsonFileStore

            store = JsonFileStore(str(self._audit_path))
            data = store.read({"executions": []})
            executions = data.get("executions", [])
            executions.append({
                "skill_id": skill.skill_id,
                "version": skill.version,
                "risk": effective_risk(skill).value,
                "status": status,
                "steps": steps,
                "arguments": redact_value(args),
                "error": redact_text(error or "")[:200],
                "at": datetime.now(timezone.utc).isoformat(),
            })
            store.write({"executions": executions[-500:]})
        except Exception as e:  # noqa: BLE001 - never breaks
            logger.warning(
                "skill execution audit failed: %s", e
            )


class SkillTool(Tool):
    """Adapter: exposes one Skill as a ``skill_<id>`` Tool.

    One-way bridge only — the Skill itself never becomes a
    Tool subclass, and all execution still flows through the
    SkillExecutor (which itself goes through the normal
    ToolRegistry path).
    """

    def __init__(
        self, skill: Skill, registry: SkillRegistry,
        executor: SkillExecutor | None = None,
    ) -> None:
        self._skill = skill
        self._registry = registry
        self._executor = executor
        self.name = skill.tool_name
        self.description = (
            f"[Skill v{skill.version}] {skill.description} "
            f"(risk: {effective_risk(skill).value})"
        )
        self.input_schema = dict(skill.input_schema) or {
            "type": "object", "properties": {}, "required": [],
        }

    def run(self, arguments: dict[str, Any]) -> Any:
        # Always resolve the *current* active version: a
        # rollback or update between planning and execution
        # must take effect immediately.
        skill = self._registry.get_or_none(
            self._skill.skill_id
        )
        if skill is None:
            raise SkillExecutionError(
                f"skill {self._skill.skill_id!r} no longer "
                "registered",
                code="unknown_skill",
                tool=self.name,
            )
        executor = self._executor or SkillExecutor(
            self._registry,
            self._registry._tool_registry,
        )
        return executor.execute(skill.skill_id, arguments)

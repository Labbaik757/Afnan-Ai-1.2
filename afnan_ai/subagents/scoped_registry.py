"""Scoped tool registry — least-privilege tool access.

A subagent never sees the parent's full ToolRegistry.  It
gets a read-only *view* exposing only its allowed tools:

* ``allowed`` entries are exact names or ``prefix_*`` globs,
* registration is refused (subagents cannot add tools),
* every execution additionally checks the subagent's risk
  permissions (a researcher cannot run a destructive tool
  even if it is somehow listed),
* ``connector_execute`` is filtered by ``allowed_connectors``
  at the argument level,
* tool calls are counted against the subagent's budget.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.log_config import get_logger
from afnan_ai.skills.models import SkillRisk
from afnan_ai.skills.risk import _RANK, classify_tool
from afnan_ai.tools.base import (
    Tool,
    ToolErrorCode,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolResult,
)
from afnan_ai.tools.registry import ToolRegistry

logger = get_logger(__name__)


def _matches(pattern: str, name: str) -> bool:
    if pattern.endswith("*"):
        return name.startswith(pattern[:-1])
    return name == pattern


class ScopedToolRegistry(ToolRegistry):
    """Read-only least-privilege view over a parent registry."""

    def __init__(
        self,
        parent: ToolRegistry,
        *,
        allowed: list[str],
        risk_permissions: tuple[SkillRisk, ...],
        allowed_connectors: list[str] | None = None,
        max_tool_calls: int | None = None,
        owner_id: str = "",
    ) -> None:
        # Do NOT call super().__init__ with tools: this
        # registry is a view, not a store.
        self._parent = parent
        self._allowed = list(allowed or [])
        self._risk_permissions = tuple(risk_permissions or ())
        self._allowed_connectors = list(
            allowed_connectors or []
        )
        self._max_tool_calls = max_tool_calls
        self._tool_calls = 0
        self._owner_id = owner_id

    # -- visibility -------------------------------------------------------
    def _is_allowed(self, name: str) -> bool:
        return any(
            _matches(pattern, name)
            for pattern in self._allowed
        )

    def _check_risk(self, name: str) -> None:
        risk = classify_tool(name)
        if risk not in self._risk_permissions:
            raise ToolExecutionError(
                f"tool {name!r} (risk {risk.value}) is outside "
                f"this subagent's permissions",
                code=ToolErrorCode.EXECUTION_FAILED,
                tool=name,
                details={
                    "permission": "risk",
                    "risk": risk.value,
                    "subagent": self._owner_id,
                },
            )

    def _check_connector(self, arguments: dict[str, Any]) -> None:
        connector_id = str(
            (arguments or {}).get("connector_id", "")
        )
        if (
            connector_id
            and connector_id not in self._allowed_connectors
        ):
            raise ToolExecutionError(
                f"connector {connector_id!r} is not allowed "
                "for this subagent",
                code=ToolErrorCode.EXECUTION_FAILED,
                tool="connector_execute",
                details={
                    "permission": "connector",
                    "connector_id": connector_id,
                    "subagent": self._owner_id,
                },
            )

    # -- read-only view ------------------------------------------------------
    def register(
        self, tool: Tool, *, replace: bool = False
    ) -> Tool:
        raise ToolRegistrationError(
            "subagents cannot register tools",
            tool=getattr(tool, "name", "?"),
        )

    def register_many(self, tools: Any) -> None:
        raise ToolRegistrationError(
            "subagents cannot register tools"
        )

    def unregister(self, name: str) -> Tool:
        raise ToolRegistrationError(
            "subagents cannot unregister tools"
        )

    def get(self, name: str) -> Tool:
        if not self._is_allowed(name):
            raise ToolNotFoundError(
                f"tool {name!r} is not available to this "
                "subagent",
                tool=name,
            )
        return self._parent.get(name)

    def get_or_none(self, name: str) -> Tool | None:
        if not self._is_allowed(name):
            return None
        return self._parent.get_or_none(name)

    def has(self, name: str) -> bool:
        return self._is_allowed(name) and self._parent.has(name)

    def __contains__(self, name: object) -> bool:
        return (
            isinstance(name, str) and self.has(name)
        )

    def names(self) -> list[str]:
        return sorted(
            name for name in self._parent.names()
            if self._is_allowed(name)
        )

    def list_tools(self) -> list[Tool]:
        return [
            tool for tool in self._parent.list_tools()
            if self._is_allowed(tool.name)
        ]

    def definitions(self) -> list[dict[str, Any]]:
        return [tool.definition() for tool in self.list_tools()]

    def __len__(self) -> int:
        return len(self.names())

    # -- guarded execution ------------------------------------------------------
    def _before_call(
        self, name: str, arguments: dict[str, Any] | None
    ) -> None:
        if not self._is_allowed(name):
            raise ToolNotFoundError(
                f"tool {name!r} is not available to this "
                "subagent",
                tool=name,
            )
        self._check_risk(name)
        if name == "connector_execute":
            self._check_connector(arguments or {})
        if (
            self._max_tool_calls is not None
            and self._tool_calls >= self._max_tool_calls
        ):
            raise ToolExecutionError(
                f"subagent {self._owner_id!r} exceeded its "
                f"tool-call budget ({self._max_tool_calls})",
                code=ToolErrorCode.EXECUTION_FAILED,
                tool=name,
                details={
                    "permission": "budget",
                    "subagent": self._owner_id,
                },
            )
        self._tool_calls += 1

    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            self._before_call(name, arguments)
        except (ToolNotFoundError, ToolExecutionError) as e:
            return ToolResult.fail(
                name,
                e.error if hasattr(e, "error") else None
                or {
                    "code": "permission_denied",
                    "message": str(e),
                },
            )
        # Subagent calls are attributed to the subagent
        # actor so the central policy sees who is acting.
        # The parent registry's SecurityCenter (if any)
        # authorizes with least-privilege subagent caps.
        kwargs.setdefault(
            "security_actor",
            f"subagent:{self._owner_id or 'unknown'}",
        )
        return self._parent.execute(name, arguments, **kwargs)

    def execute_or_raise(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self._before_call(name, arguments)
        return self._parent.execute_or_raise(
            name, arguments, **kwargs
        )

    @property
    def tool_calls(self) -> int:
        return self._tool_calls

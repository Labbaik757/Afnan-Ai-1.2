"""ToolRegistry — central place to register, find and run tools.

The agent (and an LLM, a CLI or a UI) talks to capabilities only
through the registry:

* ``register(tool)`` / ``register_many([...])`` — add tools at
  runtime; duplicate or malformed tools are rejected with a
  structured :class:`ToolRegistrationError`
* ``get(name)`` — return the tool, raising a structured
  :class:`ToolNotFoundError` for an unknown name
  (``get_or_none`` / ``has`` for soft lookups)
* ``execute(name, arguments)`` — run a tool and always get a
  :class:`ToolResult` back.  Unknown tools, missing/invalid
  arguments and execution failures come back as
  ``ToolResult(success=False, error=ToolError(code=...))`` instead
  of raising or crashing the caller.
* ``definitions()`` / ``list_tools()`` — serialisable descriptions
  of everything registered, ready to hand to an LLM.
"""

from __future__ import annotations

from typing import Any, Iterable

from afnan_ai.tools.base import (
    Tool,
    ToolError,
    ToolErrorCode,
    ToolException,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolResult,
)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        # Central security hook (None = legacy behavior).
        # When a SecurityCenter is attached, EVERY tool call
        # is authorized through it first: validate →
        # classify → permission check → approval → audit.
        self.security_center = None
        if tools:
            self.register_many(tools)

    def set_security_center(self, center: Any) -> None:
        """Attach the central SecurityCenter (late binding)."""
        self.security_center = center

    # -- registration -----------------------------------------------------
    def register(self, tool: Tool, *, replace: bool = False) -> Tool:
        if not isinstance(tool, Tool):
            raise ToolRegistrationError(
                f"Cannot register {tool!r}: not a Tool instance",
                code=ToolErrorCode.INVALID_TOOL,
                details={"received": type(tool).__name__},
            )
        name = (tool.name or "").strip()
        if not name:
            raise ToolRegistrationError(
                "Cannot register a tool with an empty name",
                code=ToolErrorCode.INVALID_TOOL,
            )
        if not replace and name in self._tools:
            raise ToolRegistrationError(
                f"Tool {name!r} is already registered",
                code=ToolErrorCode.TOOL_ALREADY_REGISTERED,
                tool=name,
            )
        self._tools[name] = tool
        # Registration is a developer trust action: the
        # agent actor gains the tool's domain capability and,
        # where the central system knows one, its capability
        # id — centrally-authorized calls keep working as
        # tools are added dynamically.
        center = self.security_center
        if center is not None:
            domain, _, _ = name.partition("_")
            if domain:
                center.grant_agent_capabilities(
                    f"{domain}.*"
                )
            try:
                from afnan_ai.security.policy import (
                    TOOL_CAPABILITY_MAP,
                )
                capability_id = TOOL_CAPABILITY_MAP.get(
                    name
                )
            except Exception:
                capability_id = None
            if capability_id:
                try:
                    center.grant_capability(
                        center.agent_actor, capability_id
                    )
                except KeyError:
                    pass
        return tool

    def register_many(self, tools: Iterable[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str) -> Tool:
        tool = self.get(name)  # raises ToolNotFoundError if absent
        del self._tools[name]
        return tool

    # -- lookup ---------------------------------------------------------------
    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(
                f"Unknown tool {name!r}. "
                f"Available tools: {', '.join(self.names()) or '(none)'}",
                tool=name,
                details={"available": self.names()},
            ) from None

    def get_or_none(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    __contains__ = has

    def names(self) -> list[str]:
        return sorted(self._tools)

    def list_tools(self) -> list[Tool]:
        return [self._tools[name] for name in self.names()]

    def definitions(self) -> list[dict[str, Any]]:
        """Serialisable definitions for every registered tool."""
        return [tool.definition() for tool in self.list_tools()]

    def __len__(self) -> int:
        return len(self._tools)

    # -- execution ---------------------------------------------------------------
    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Run tool *name* and return a structured ToolResult.

        Expected failures (unknown tool, missing/invalid arguments,
        the tool itself failing) are returned, not raised.

        The reserved ``security_actor`` kwarg selects the actor
        for central authorization; it never reaches the tool.
        """
        actor = kwargs.pop("security_actor", None)
        merged: dict[str, Any] = dict(arguments or {})
        merged.update(kwargs)

        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.fail(
                name,
                ToolError(
                    code=ToolErrorCode.TOOL_NOT_FOUND,
                    message=(
                        f"Unknown tool {name!r}. "
                        f"Available tools: {', '.join(self.names()) or '(none)'}"
                    ),
                    tool=name,
                    details={"available": self.names()},
                ),
            )

        center = self.security_center
        if center is not None:
            decision = center.authorize(
                tool_name=name,
                arguments=merged,
                actor=actor or center.agent_actor.label,
            )
            if decision.action == "deny":
                return ToolResult.fail(
                    name,
                    ToolError(
                        code=ToolErrorCode.PERMISSION_DENIED,
                        message=(
                            f"Security policy denied {name!r}: "
                            f"{decision.reason}"
                        ),
                        tool=name,
                        details={
                            "security": True,
                            "risk": decision.risk_level.value,
                            "reason": decision.reason,
                        },
                    ),
                )
            if decision.action == "approval_required":
                # The loop already pauses on approval codes —
                # resumable, like every other sensitive tool.
                return ToolResult.fail(
                    name,
                    ToolError(
                        code=ToolErrorCode.APPROVAL_REQUIRED,
                        message=(
                            f"Security policy requires human "
                            f"approval for {name!r} "
                            f"(risk={decision.risk_level.value})"
                        ),
                        tool=name,
                        details={
                            "security": True,
                            "risk": decision.risk_level.value,
                            "resumable": True,
                        },
                    ),
                )

        tool = self._tools.get(name)

        actor_label = ""
        if center is not None:
            actor_label = (
                actor or center.agent_actor.label
            )

        try:
            output = tool.execute(merged)
        except ToolException as e:
            # Already structured — preserve its code/message/details
            if e.error.tool is None:
                e.error.tool = name
            if center is not None:
                center.record_execution_result(
                    tool_name=name,
                    actor=actor_label,
                    task_id="",
                    success=False,
                    failure_reason=e.error.message,
                )
            return ToolResult.fail(name, e.error)
        except Exception as e:
            # A tool crash must never crash the caller
            if center is not None:
                center.record_execution_result(
                    tool_name=name,
                    actor=actor_label,
                    task_id="",
                    success=False,
                    failure_reason=(
                        f"{type(e).__name__}: {e}"
                    ),
                )
            return ToolResult.fail(
                name,
                ToolError(
                    code=ToolErrorCode.EXECUTION_FAILED,
                    message=f"Tool {name!r} failed: {e}",
                    tool=name,
                    details={"exception": type(e).__name__},
                ),
            )

        if isinstance(output, ToolResult):
            result = output
        else:
            result = ToolResult.ok(name, output)
        if center is not None:
            center.record_execution_result(
                tool_name=name,
                actor=actor_label,
                task_id="",
                success=bool(result.success),
                failure_reason=(
                    "" if result.success
                    else str(result.error)
                ),
            )
        return result

    def execute_or_raise(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Like :meth:`execute` but raises the structured exception
        form and returns only the output on success."""
        result = self.execute(name, arguments, **kwargs)
        if result.success:
            return result.output
        error = result.error
        assert error is not None  # fail() always sets an error
        if error.code == ToolErrorCode.TOOL_NOT_FOUND:
            raise ToolNotFoundError(
                error.message, tool=name, details=error.details
            )
        if error.code == ToolErrorCode.EXECUTION_FAILED:
            raise ToolExecutionError(
                error.message, tool=name, details=error.details
            )
        raise ToolException(
            error.message, code=error.code, tool=name, details=error.details
        )

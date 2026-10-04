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
        if tools:
            self.register_many(tools)

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
        """
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

        try:
            output = tool.execute(merged)
        except ToolException as e:
            # Already structured — preserve its code/message/details
            if e.error.tool is None:
                e.error.tool = name
            return ToolResult.fail(name, e.error)
        except Exception as e:
            # A tool crash must never crash the caller
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
            return output
        return ToolResult.ok(name, output)

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

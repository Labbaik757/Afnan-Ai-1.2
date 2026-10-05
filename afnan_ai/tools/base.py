"""Generic Tool interface and structured tool errors.

Every capability Afnan exposes to the agent (opening a URL,
launching an app, searching Google, taking a screenshot, ...) is a
:class:`Tool` with:

* ``name`` — unique machine name, e.g. ``"open_url"``
* ``description`` — human/LLM readable summary of what it does
* ``input_schema`` — JSON-Schema-style description of its arguments
  (``properties`` + ``required``), usable directly for LLM
  function-calling or validation
* ``execute(arguments)`` — validate the arguments and do the work

Failure model
-------------
Tools and the registry never leak bare, provider-style exceptions
for the expected failure modes.  They use structured errors:

* :class:`ToolError` — a serializable record
  ``{code, message, tool, details}``
* :class:`ToolResult` — the registry's return value:
  ``{tool, success, output, error}`` where ``error`` is a
  :class:`ToolError` on failure and ``None`` on success
* Exception forms (:class:`ToolNotFoundError`,
  :class:`ToolValidationError`, :class:`ToolExecutionError`) for
  callers who prefer ``try/except`` — each carries the same
  structured :class:`ToolError` on ``.error``

Error codes:

* ``tool_not_found`` — no tool registered under that name
* ``missing_arguments`` — a required argument was not supplied
* ``invalid_arguments`` — an argument has the wrong type/shape or
  an unknown argument was supplied
* ``execution_failed`` — the tool ran but could not do the work
* ``tool_already_registered`` / ``invalid_tool`` — registry misuse
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolErrorCode(str, Enum):
    TOOL_NOT_FOUND = "tool_not_found"
    MISSING_ARGUMENTS = "missing_arguments"
    INVALID_ARGUMENTS = "invalid_arguments"
    EXECUTION_FAILED = "execution_failed"
    TOOL_ALREADY_REGISTERED = "tool_already_registered"
    INVALID_TOOL = "invalid_tool"
    PERMISSION_DENIED = "permission_denied"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_DENIED = "approval_denied"


@dataclass
class ToolError:
    """Structured, serializable description of a tool failure."""

    code: ToolErrorCode
    message: str
    tool: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "tool": self.tool,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolError":
        return cls(
            code=ToolErrorCode(data.get("code", ToolErrorCode.EXECUTION_FAILED.value)),
            message=data.get("message", ""),
            tool=data.get("tool"),
            details=dict(data.get("details") or {}),
        )


class ToolException(Exception):
    """Exception carrying a structured :class:`ToolError`."""

    default_code = ToolErrorCode.EXECUTION_FAILED

    def __init__(
        self,
        message: str,
        *,
        code: ToolErrorCode | None = None,
        tool: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.error = ToolError(
            code=code or self.default_code,
            message=message,
            tool=tool,
            details=dict(details or {}),
        )
        super().__init__(message)

    @property
    def code(self) -> ToolErrorCode:
        return self.error.code


class ToolNotFoundError(ToolException):
    default_code = ToolErrorCode.TOOL_NOT_FOUND


class ToolValidationError(ToolException):
    """Missing or invalid arguments."""

    default_code = ToolErrorCode.INVALID_ARGUMENTS


class ToolExecutionError(ToolException):
    default_code = ToolErrorCode.EXECUTION_FAILED


class ToolRegistrationError(ToolException):
    default_code = ToolErrorCode.INVALID_TOOL


@dataclass
class ToolResult:
    """Result returned by ``ToolRegistry.execute`` — never raises
    for not-found / missing-argument / execution failures."""

    tool: str
    success: bool
    output: Any = None
    error: ToolError | None = None

    @classmethod
    def ok(cls, tool: str, output: Any = None) -> "ToolResult":
        return cls(tool=tool, success=True, output=output)

    @classmethod
    def fail(cls, tool: str, error: ToolError) -> "ToolResult":
        return cls(tool=tool, success=False, error=error)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "success": bool(self.success),
            "output": self.output,
            "error": self.error.to_dict() if self.error else None,
        }


_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: (isinstance(v, (int, float)) and not isinstance(v, bool)),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


class Tool(ABC):
    """Interface every tool implements."""

    #: unique machine name, e.g. "open_url"
    name: str = ""
    #: what the tool does, for humans and LLMs
    description: str = ""
    #: JSON-Schema-style: {"type": "object", "properties": {...},
    #:                      "required": [...]}
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    # -- description for registries / LLMs --------------------------------
    def definition(self) -> dict[str, Any]:
        """Serialisable tool definition (name, description, schema)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": self.input_schema.get("type", "object"),
                "properties": dict(self.input_schema.get("properties", {})),
                "required": list(self.input_schema.get("required", [])),
            },
        }

    # -- argument handling ---------------------------------------------------
    def validate_arguments(
        self, arguments: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Return cleaned arguments or raise ToolValidationError."""
        args = dict(arguments or {})
        schema = self.input_schema or {}
        properties: dict[str, Any] = schema.get("properties", {})
        required: list[str] = list(schema.get("required", []))

        missing = [key for key in required if args.get(key) is None]
        if missing:
            raise ToolValidationError(
                f"Tool {self.name!r} is missing required argument(s): "
                + ", ".join(missing),
                code=ToolErrorCode.MISSING_ARGUMENTS,
                tool=self.name,
                details={"missing": missing},
            )

        if schema.get("additionalProperties") is False:
            unknown = [key for key in args if key not in properties]
            if unknown:
                raise ToolValidationError(
                    f"Tool {self.name!r} got unknown argument(s): "
                    + ", ".join(sorted(unknown)),
                    tool=self.name,
                    details={"unknown": sorted(unknown)},
                )

        for key, spec in properties.items():
            if key not in args or args[key] is None:
                continue
            expected = (spec or {}).get("type")
            check = _TYPE_CHECKS.get(expected or "")
            if check is not None and not check(args[key]):
                raise ToolValidationError(
                    f"Argument {key!r} for tool {self.name!r} must be "
                    f"of type {expected}, got {type(args[key]).__name__}",
                    tool=self.name,
                    details={"argument": key, "expected": expected},
                )
            if expected == "string" and key in required and not str(args[key]).strip():
                raise ToolValidationError(
                    f"Tool {self.name!r} is missing required argument(s): {key}",
                    code=ToolErrorCode.MISSING_ARGUMENTS,
                    tool=self.name,
                    details={"missing": [key]},
                )
        return args

    # -- execution --------------------------------------------------------------
    def execute(self, arguments: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        """Validate *arguments* and run the tool.

        Raises ToolValidationError for missing/invalid arguments and
        ToolExecutionError (or a subclass) when the work fails.
        ``ToolRegistry.execute`` converts those into a ToolResult.
        """
        merged: dict[str, Any] = dict(arguments or {})
        merged.update(kwargs)
        cleaned = self.validate_arguments(merged)
        return self.run(cleaned)

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> Any:
        """Do the work with validated arguments and return the output."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r})"


class FunctionTool(Tool):
    """A Tool built from a plain callable — handy for registration
    of small/custom tools without writing a subclass."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        func,
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema or {
            "type": "object",
            "properties": {},
            "required": [],
        }
        self._func = func

    def run(self, arguments: dict[str, Any]) -> Any:
        return self._func(**arguments)

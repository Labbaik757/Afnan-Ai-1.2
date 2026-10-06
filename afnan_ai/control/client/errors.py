"""Client-side protocol errors.

Every failure a remote client can hit is raised as a
:class:`ProtocolError` (or one of its subclasses) with a machine
readable ``code`` attribute.  Token values are never embedded in
error messages.
"""


class ProtocolError(Exception):
    """Base error for control-plane protocol failures."""

    def __init__(
        self,
        message: str = "",
        code: str = "protocol_error",
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status

    def __str__(self) -> str:
        base = super().__str__()
        if self.status is not None:
            return f"[{self.code}] HTTP {self.status}: {base}"
        return f"[{self.code}] {base}"


class AuthenticationError(ProtocolError):
    """The token was missing, expired, revoked or otherwise invalid."""

    def __init__(
        self,
        message: str = "authentication failed",
        code: str = "unauthorized",
        status: int | None = 401,
    ) -> None:
        super().__init__(message, code=code, status=status)


class AuthorizationError(ProtocolError):
    """The session is authenticated but lacks the capability."""

    def __init__(
        self,
        message: str = "not authorized",
        code: str = "forbidden",
        status: int | None = 403,
    ) -> None:
        super().__init__(message, code=code, status=status)


class ConnectionError(ProtocolError):
    """The transport itself failed (network, timeout, TLS, refused)."""

    def __init__(
        self,
        message: str = "connection failed",
        code: str = "connection_error",
        status: int | None = None,
    ) -> None:
        super().__init__(message, code=code, status=status)

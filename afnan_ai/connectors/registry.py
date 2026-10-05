"""ConnectorRegistry — dynamic registration and discovery.

Connectors are registered at runtime (``register``), removed
(``unregister``), looked up by id (``get`` — a structured
``invalid_connector`` error for unknown ids), and discovered
from installed distributions via the
``"afnan_ai.connectors"`` entry-point group (best-effort:
a broken entry point is recorded, never raised).

The registry also produces the *structured capability schema*
the Planner/AgentLoop consumes: which connectors exist, what
each one does, each operation's risk level, required scopes
and parameter schema — with no connector-specific logic in
the agent itself.
"""

from __future__ import annotations

from typing import Any

from afnan_ai.connectors.base import Connector
from afnan_ai.connectors.errors import (
    ConnectorErrorCode,
    ConnectorException,
)
from afnan_ai.log_config import get_logger

logger = get_logger(__name__)

ENTRY_POINT_GROUP = "afnan_ai.connectors"


class ConnectorRegistry:
    def __init__(self) -> None:
        self._connectors: dict[str, Connector] = {}
        self._discovery_errors: list[str] = []

    # -- registration --------------------------------------------------
    def register(self, connector: Connector) -> Connector:
        if not isinstance(connector, Connector):
            raise ConnectorException(
                f"Cannot register {connector!r}: not a Connector",
                code=ConnectorErrorCode.INVALID_CONNECTOR,
                details={"received": type(connector).__name__},
            )
        connector_id = (connector.connector_id or "").strip()
        if not connector_id:
            raise ConnectorException(
                "Cannot register a connector with an empty id",
                code=ConnectorErrorCode.INVALID_CONNECTOR,
            )
        if connector_id in self._connectors:
            raise ConnectorException(
                f"Connector {connector_id!r} is already registered",
                code=ConnectorErrorCode.INVALID_CONNECTOR,
                connector=connector_id,
                details={"duplicate": connector_id},
            )
        operation_names = [
            op.name for op in (connector.operations or ())
        ]
        duplicates = {
            name
            for name in operation_names
            if operation_names.count(name) > 1
        }
        if duplicates:
            raise ConnectorException(
                f"Connector {connector_id!r} declares duplicate "
                f"operation(s): {sorted(duplicates)}",
                code=ConnectorErrorCode.INVALID_CONNECTOR,
                connector=connector_id,
                details={"duplicates": sorted(duplicates)},
            )
        self._connectors[connector_id] = connector
        logger.info(
            "registered connector %r (%d operations)",
            connector_id, len(operation_names),
        )
        return connector

    def unregister(self, connector_id: str) -> Connector:
        connector = self.get(connector_id)  # structured if absent
        del self._connectors[connector_id]
        logger.info("unregistered connector %r", connector_id)
        return connector

    def register_many(
        self, connectors: list[Connector] | tuple[Connector, ...]
    ) -> None:
        for connector in connectors:
            self.register(connector)

    # -- discovery --------------------------------------------------------
    def discover(
        self,
        *,
        group: str = ENTRY_POINT_GROUP,
    ) -> list[str]:
        """Load connectors from the entry-point *group*.

        Returns the ids that were newly registered.  Entry
        points that fail to load are recorded in
        :attr:`discovery_errors` and skipped — discovery never
        breaks agent startup.
        """
        discovered: list[str] = []
        try:
            from importlib.metadata import entry_points
        except Exception:  # pragma: no cover - stdlib always here
            return discovered
        try:
            points = entry_points(group=group)
        except Exception as e:
            self._discovery_errors.append(
                f"entry_points({group!r}) failed: {e}"
            )
            return discovered
        for point in points:
            if point.name in self._connectors:
                continue  # already registered: keep the explicit one
            try:
                factory = point.load()
                connector = factory() if callable(factory) else factory
                self.register(connector)
                discovered.append(connector.connector_id)
            except Exception as e:  # noqa: BLE001 - best effort
                message = (
                    f"entry point {point.name!r} failed to load: {e}"
                )
                self._discovery_errors.append(message)
                logger.warning("%s", message)
        return discovered

    @property
    def discovery_errors(self) -> list[str]:
        return list(self._discovery_errors)

    # -- lookup --------------------------------------------------------------
    def get(self, connector_id: str) -> Connector:
        try:
            return self._connectors[connector_id]
        except KeyError:
            raise ConnectorException(
                f"Unknown connector {connector_id!r}. Available: "
                f"{', '.join(self.ids()) or '(none)'}",
                code=ConnectorErrorCode.INVALID_CONNECTOR,
                connector=connector_id,
                details={"available": self.ids()},
            ) from None

    def has(self, connector_id: str) -> bool:
        return connector_id in self._connectors

    __contains__ = has

    def ids(self) -> list[str]:
        return sorted(self._connectors)

    def list(self) -> list[Connector]:
        return [self._connectors[cid] for cid in self.ids()]

    def __len__(self) -> int:
        return len(self._connectors)

    # -- planner-facing schema -------------------------------------------------
    def capabilities_schema(self) -> list[dict[str, Any]]:
        """Structured capabilities for the Planner/AgentLoop.

        A list of connector infos (id, name, description,
        auth type, declared scopes, operations with risk level,
        required scopes and parameter schema).  The planner
        decides *which* connector/operation fits the goal; the
        service enforces permissions and approval at runtime.
        """
        return [connector.info().to_dict() for connector in self.list()]

    def summary_lines(self) -> list[str]:
        """Compact one-line-per-connector summary for prompts."""
        lines = []
        for connector in self.list():
            ops = ", ".join(
                f"{op.name}[{op.risk.value}]"
                for op in (connector.operations or ())
            )
            auth_type = connector.auth_type
            auth_name = (
                auth_type.value
                if hasattr(auth_type, "value")
                else str(auth_type)
            )
            lines.append(
                f"- {connector.connector_id} ({connector.name}): "
                f"{connector.description} | auth={auth_name}"
                f" | ops: {ops or '(none)'}"
            )
        return lines

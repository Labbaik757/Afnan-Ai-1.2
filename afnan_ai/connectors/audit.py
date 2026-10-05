"""Connector audit trail — safe, append-only operation records.

Every connector lifecycle event and operation execution is
recorded as an :class:`~afnan_ai.connectors.models.AuditRecord`
(connector, operation, risk level, timestamp, approval status,
execution result, verification result).  Records are kept in
memory (bounded) and, when a path is given, appended as JSONL
to a file.  Secrets and sensitive content are redacted before
anything is written.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from afnan_ai.connectors.models import AuditRecord
from afnan_ai.log_config import get_logger
from afnan_ai.redaction import redact_value

logger = get_logger(__name__)

_MAX_MEMORY_ENTRIES = 500


class ConnectorAuditLog:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(str(path)).expanduser() if path else None
        self._entries: deque[dict[str, Any]] = deque(
            maxlen=_MAX_MEMORY_ENTRIES
        )

    def record(self, entry: AuditRecord | dict[str, Any]) -> None:
        data = (
            entry.to_dict()
            if isinstance(entry, AuditRecord)
            else dict(entry)
        )
        # Backstop: nothing secret-shaped may reach the audit log.
        data = redact_value(data)
        self._entries.append(data)
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    import json

                    handle.write(
                        json.dumps(data, default=str) + "\n"
                    )
            except OSError as e:  # audit must never break execution
                logger.warning(
                    "connector audit write failed (%s): %s",
                    self.path, e,
                )

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        items = list(self._entries)
        return items[-limit:] if limit else items

    def __len__(self) -> int:
        return len(self._entries)

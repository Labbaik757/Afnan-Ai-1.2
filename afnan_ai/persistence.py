"""Tiny JSON file persistence shared by the memory/goal/task
managers.

One JSON document per file, written atomically (temp file +
``os.replace``) so a crash mid-write never corrupts the store,
and read defensively: a missing file is an empty document, and
a corrupted file is preserved as ``*.corrupt-*`` evidence while
the caller gets an empty document — persistence problems must
never crash the agent.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class JsonFileStore:
    """Atomic, corruption-tolerant JSON document store."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()

    def read(self, default: dict[str, Any]) -> dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return dict(default)
        except OSError:
            return dict(default)
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            self._preserve_corrupt()
            return dict(default)
        if not isinstance(data, dict):
            self._preserve_corrupt()
            return dict(default)
        return data

    def write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def _preserve_corrupt(self) -> None:
        try:
            stamp = time.strftime("%Y%m%d%H%M%S")
            self.path.replace(
                self.path.with_suffix(
                    self.path.suffix + f".corrupt-{stamp}"
                )
            )
        except OSError:
            pass  # evidence preservation is best-effort

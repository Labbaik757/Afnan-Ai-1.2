"""Skill execution history.

Records per execution: skill version, task, workspace,
input metadata (never raw sensitive values), duration,
tool usage, result status, verification, failures and
artifacts.  Powers optimization suggestions and audit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from afnan_ai.redaction import redact_text


def _redact_inputs(
    arguments: dict[str, Any]
) -> dict[str, Any]:
    """Input metadata only — values redacted."""
    out: dict[str, Any] = {}
    for key, value in (arguments or {}).items():
        text = str(value)
        if len(text) > 200 or any(
            marker in key.lower()
            for marker in (
                "password", "secret", "token", "api_key",
                "credential",
            )
        ):
            out[key] = {
                "type": type(value).__name__,
                "chars": len(text),
                "redacted": True,
            }
        else:
            out[key] = {
                "type": type(value).__name__,
                "preview": redact_text(text)[:60],
            }
    return out


@dataclass
class SkillExecutionRecord:
    skill_id: str
    version: str
    task_id: str = ""
    workspace_id: str = ""
    input_metadata: dict[str, Any] = field(
        default_factory=dict
    )
    started_at: float = field(default_factory=time.monotonic)
    duration_s: float = 0.0
    tools_used: list[str] = field(default_factory=list)
    status: str = "unknown"  # success|failed|uncertain
    verification: str = ""
    failures: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    retries: int = 0

    def finish(
        self,
        status: str,
        *,
        verification: str = "",
        failures: list[str] | None = None,
    ) -> None:
        self.status = status
        self.verification = verification[:300]
        self.failures = [f[:200] for f in (failures or [])]
        self.duration_s = (
            time.monotonic() - self.started_at
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "task_id": self.task_id,
            "workspace_id": self.workspace_id,
            "input_metadata": dict(self.input_metadata),
            "duration_s": round(self.duration_s, 3),
            "tools_used": list(self.tools_used),
            "status": self.status,
            "verification": self.verification,
            "failures": list(self.failures),
            "artifacts": list(self.artifacts),
            "retries": self.retries,
        }


class SkillHistory:
    """Bounded per-skill execution history."""

    def __init__(self, *, max_records: int = 200) -> None:
        self.max_records = max(1, max_records)
        self._records: list[SkillExecutionRecord] = []

    def start(
        self,
        skill_id: str,
        version: str,
        arguments: dict[str, Any] | None = None,
        *,
        task_id: str = "",
        workspace_id: str = "",
    ) -> SkillExecutionRecord:
        record = SkillExecutionRecord(
            skill_id=skill_id,
            version=version,
            task_id=task_id,
            workspace_id=workspace_id,
            input_metadata=_redact_inputs(arguments),
        )
        self._records.append(record)
        del self._records[: -self.max_records]
        return record

    def for_skill(
        self, skill_id: str, *, limit: int = 50
    ) -> list[SkillExecutionRecord]:
        matching = [
            r for r in self._records if r.skill_id == skill_id
        ]
        return matching[-limit:]

    def stats(self, skill_id: str) -> dict[str, Any]:
        records = self.for_skill(skill_id, limit=200)
        if not records:
            return {
                "runs": 0, "success_rate": None,
                "avg_duration_s": None,
            }
        successes = sum(
            1 for r in records if r.status == "success"
        )
        durations = [
            r.duration_s for r in records if r.duration_s > 0
        ]
        failures: dict[str, int] = {}
        for r in records:
            for f in r.failures:
                failures[f[:80]] = failures.get(f[:80], 0) + 1
        return {
            "runs": len(records),
            "success_rate": round(
                successes / len(records), 3
            ),
            "avg_duration_s": (
                round(
                    sum(durations) / len(durations), 3
                )
                if durations
                else None
            ),
            "total_retries": sum(r.retries for r in records),
            "top_failures": sorted(
                failures.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:5],
        }

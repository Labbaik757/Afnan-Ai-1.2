"""MemoryStore — persistent, safety-gated long-term memory.

AgentState is temporary working memory for one task; the
MemoryStore is what survives across tasks: user preferences,
verified facts, project context, summaries of completed tasks
and useful learned context.  Every record carries its source,
timestamps, a confidence score and metadata, and retrieval
ranks by relevance + confidence + recency.

Safety rules (enforced here, not by callers):

* only trusted sources may write — ``user`` (the user said
  it), ``verified_result`` (the agent verified it) and
  ``system`` (approved configuration).  Webpages, emails,
  documents and other untrusted content can never create
  permanent memory directly;
* content containing secret material (passwords, tokens, API
  keys, cookies, cards, JWTs — detected via the shared
  redactor) is refused, never stored;
* conflicting memories are resolved by confidence and source
  authority, never blindly overwritten — the loser is kept in
  the record's history;
* a corrupt or unreadable store degrades to empty memory;
  memory failures must never crash the agent.

The backend is replaceable: :class:`MemoryStore` is the
interface, :class:`LocalMemoryStore` the initial local JSON
implementation (a future cloud/database backend implements
the same contract).
"""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from afnan_ai.persistence import JsonFileStore
from afnan_ai.redaction import redact_text

MEMORY_KINDS = frozenset({
    "preference", "fact", "project", "task_summary", "learned",
})

TRUSTED_SOURCES = frozenset({"user", "verified_result", "system"})
_SOURCE_RANK = {"user": 3, "verified_result": 2, "system": 1}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryError(Exception):
    """Structured memory failure (never fatal to the agent)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message}


@dataclass
class MemoryRecord:
    memory_id: str
    kind: str
    content: str
    source: str
    confidence: float = 0.8
    key: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    last_accessed_at: str = field(default_factory=_now)
    access_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "kind": self.kind,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "key": self.key,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        return cls(
            memory_id=str(data.get("memory_id", "")),
            kind=str(data.get("kind", "fact")),
            content=str(data.get("content", "")),
            source=str(data.get("source", "system")),
            confidence=float(data.get("confidence", 0.8)),
            key=str(data.get("key", "")),
            metadata=dict(data.get("metadata") or {}),
            created_at=str(data.get("created_at", _now())),
            updated_at=str(data.get("updated_at", _now())),
            last_accessed_at=str(
                data.get("last_accessed_at", _now())
            ),
            access_count=int(data.get("access_count", 0)),
        )


@dataclass
class AddOutcome:
    record: MemoryRecord
    conflict: bool = False
    replaced: bool = False


def assert_no_secrets(text: str, *, what: str = "memory content") -> None:
    """Refuse content carrying secret material.

    Uses the shared redactor as the detector: if redacting the
    text would change it, it contains something that must
    never be persisted (credential assignments, bearer tokens,
    JWTs, card numbers...).
    """
    if redact_text(text) != text:
        raise MemoryError(
            "secret_content",
            f"Refusing to store {what}: it appears to contain "
            "secret material (password/token/key-like content)",
        )


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _recency(record: MemoryRecord) -> float:
    try:
        updated = datetime.fromisoformat(record.updated_at)
    except ValueError:
        return 0.5
    age_days = max(
        0.0,
        (datetime.now(timezone.utc) - updated).total_seconds()
        / 86400.0,
    )
    return 1.0 / (1.0 + age_days / 30.0)


class MemoryStore(ABC):
    """The replaceable long-term memory contract."""

    @abstractmethod
    def add(
        self,
        content: str,
        *,
        kind: str = "fact",
        source: str,
        confidence: float = 0.8,
        key: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AddOutcome:
        ...

    @abstractmethod
    def get(self, memory_id: str) -> MemoryRecord | None:
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        kinds: list[str] | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        ...

    @abstractmethod
    def list(
        self, *, kind: str | None = None
    ) -> list[MemoryRecord]:
        ...

    @abstractmethod
    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        confidence: float | None = None,
    ) -> MemoryRecord:
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        ...

    def summaries_for(
        self, query: str, *, limit: int = 3
    ) -> list[str]:
        """Ranked one-line memory summaries for prompt context."""
        return [
            f"[{r.kind}, confidence {r.confidence:.2f}] {r.content}"
            for r in self.search(query, limit=limit)
        ]


class LocalMemoryStore(MemoryStore):
    """Local JSON-file MemoryStore (atomic, corruption-safe)."""

    def __init__(self, path: str):
        self._store = JsonFileStore(path)
        self._records: dict[str, MemoryRecord] | None = None

    # -- storage ---------------------------------------------------
    def _load(self) -> dict[str, MemoryRecord]:
        if self._records is None:
            data = self._store.read({"version": 1, "records": []})
            records: dict[str, MemoryRecord] = {}
            for item in data.get("records") or []:
                try:
                    record = MemoryRecord.from_dict(item)
                except Exception:
                    continue  # a bad record never breaks memory
                if record.memory_id:
                    records[record.memory_id] = record
            self._records = records
        return self._records

    def _save(self) -> None:
        records = self._load()
        self._store.write({
            "version": 1,
            "records": [r.to_dict() for r in records.values()],
        })

    # -- writes ------------------------------------------------------
    def add(
        self,
        content: str,
        *,
        kind: str = "fact",
        source: str,
        confidence: float = 0.8,
        key: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AddOutcome:
        content = str(content or "").strip()
        if not content:
            raise MemoryError(
                "empty_content", "Memory content must not be empty"
            )
        if kind not in MEMORY_KINDS:
            raise MemoryError(
                "invalid_kind",
                f"Unknown memory kind {kind!r}; expected one of "
                f"{sorted(MEMORY_KINDS)}",
            )
        if source not in TRUSTED_SOURCES:
            raise MemoryError(
                "untrusted_source",
                f"Memories may only be created from trusted "
                f"sources {sorted(TRUSTED_SOURCES)}; got "
                f"{source!r}. Untrusted external content must be "
                "verified before it can become memory.",
            )
        confidence = max(0.0, min(1.0, float(confidence)))
        assert_no_secrets(content)
        for value in (metadata or {}).values():
            if isinstance(value, str):
                assert_no_secrets(value, what="memory metadata")
        records = self._load()
        topic = key or self._derive_key(kind, content)
        existing = next(
            (
                r for r in records.values()
                if r.kind == kind and r.key == topic
            ),
            None,
        )
        if existing is not None:
            if existing.content == content:
                existing.updated_at = _now()
                self._save()
                return AddOutcome(existing)
            # Conflict: resolve by confidence, then source rank.
            new_wins = (confidence, _SOURCE_RANK[source]) > (
                existing.confidence,
                _SOURCE_RANK.get(existing.source, 0),
            )
            history = list(existing.metadata.get("history") or [])
            loser = content if not new_wins else existing.content
            history.append({
                "content": loser,
                "source": source if not new_wins else existing.source,
                "at": _now(),
                "reason": "superseded" if new_wins else "kept-existing",
            })
            existing.metadata["history"] = history[-10:]
            if new_wins:
                existing.content = content
                existing.source = source
                existing.confidence = confidence
            existing.updated_at = _now()
            self._save()
            return AddOutcome(
                existing, conflict=True, replaced=new_wins
            )
        record = MemoryRecord(
            memory_id=f"mem_{uuid.uuid4().hex[:12]}",
            kind=kind,
            content=content,
            source=source,
            confidence=confidence,
            key=topic,
            metadata=dict(metadata or {}),
        )
        records[record.memory_id] = record
        self._save()
        return AddOutcome(record)

    @staticmethod
    def _derive_key(kind: str, content: str) -> str:
        words = sorted(_tokens(content))[:6]
        return f"{kind}:{' '.join(words)}"

    # -- reads ---------------------------------------------------------
    def get(self, memory_id: str) -> MemoryRecord | None:
        return self._load().get(memory_id)

    def list(
        self, *, kind: str | None = None
    ) -> list[MemoryRecord]:
        records = list(self._load().values())
        if kind is not None:
            records = [r for r in records if r.kind == kind]
        return sorted(records, key=lambda r: r.updated_at)

    def search(
        self,
        query: str,
        *,
        kinds: list[str] | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        query_tokens = _tokens(query)
        scored: list[tuple[float, MemoryRecord]] = []
        for record in self._load().values():
            if kinds and record.kind not in kinds:
                continue
            record_tokens = _tokens(
                record.content + " " + record.key
            )
            if query_tokens and record_tokens:
                overlap = (
                    2 * len(query_tokens & record_tokens)
                    / (len(query_tokens) + len(record_tokens))
                )
            else:
                overlap = 0.0
            score = (
                0.55 * overlap
                + 0.25 * record.confidence
                + 0.20 * _recency(record)
            )
            scored.append((score, record))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = [r for _, r in scored[: max(1, int(limit))]]
        if results:
            for record in results:
                record.access_count += 1
                record.last_accessed_at = _now()
            self._save()
        return results

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        confidence: float | None = None,
    ) -> MemoryRecord:
        record = self._load().get(memory_id)
        if record is None:
            raise MemoryError(
                "not_found", f"No memory with id {memory_id!r}"
            )
        if content is not None:
            assert_no_secrets(content)
            record.content = str(content)
        if confidence is not None:
            record.confidence = max(
                0.0, min(1.0, float(confidence))
            )
        record.updated_at = _now()
        self._save()
        return record

    def delete(self, memory_id: str) -> bool:
        records = self._load()
        if memory_id not in records:
            return False
        del records[memory_id]
        self._save()
        return True

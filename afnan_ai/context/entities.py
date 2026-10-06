"""Entity continuity + conflict detection.

Long tasks reference people, files, websites, projects and
more.  The EntityRegistry associates entity mentions with
trajectory/context, tracks their attributes, and detects
contradictions: when a newer *verified* source disagrees
with an older one, the conflict is identified, the newer
verified value wins, and the resolution is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afnan_ai.context.models import TrustZone
from afnan_ai.redaction import redact_text


@dataclass
class EntityRecord:
    name: str
    entity_type: str  # person|company|file|website|project|
    # repository|calendar_event|document|task|other
    attributes: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # attr -> {value, source, verified, at}
    mentions: list[str] = field(default_factory=list)

    def set_attribute(
        self,
        attr: str,
        value: str,
        *,
        source: str = "",
        verified: bool = False,
        at: str = "",
    ) -> dict[str, Any] | None:
        """Set an attribute; returns a conflict record if one.

        A conflict exists when a previous *verified* value
        differs from the new value.  Newer verified sources
        win; unverified sources never overwrite verified ones
        silently — the conflict is surfaced instead.
        """
        previous = self.attributes.get(attr)
        conflict = None
        if (
            previous is not None
            and previous.get("value") != value
            and previous.get("verified")
        ):
            conflict = {
                "entity": self.name,
                "attribute": attr,
                "old_value": previous.get("value"),
                "old_source": previous.get("source"),
                "new_value": value,
                "new_source": source,
                "resolution": (
                    "newer_verified_wins"
                    if verified
                    else "kept_verified_old"
                ),
            }
        if conflict is None or (
            conflict["resolution"] == "newer_verified_wins"
        ):
            self.attributes[attr] = {
                "value": value,
                "source": source,
                "verified": verified,
                "at": at,
            }
        return conflict

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": redact_text(self.name)[:160],
            "type": self.entity_type,
            "attributes": {
                k: {
                    "value": redact_text(str(v.get("value")))[
                        :200
                    ],
                    "source": str(v.get("source", ""))[:80],
                    "verified": bool(v.get("verified")),
                    "at": str(v.get("at", "")),
                }
                for k, v in self.attributes.items()
            },
            "mention_count": len(self.mentions),
        }


class EntityRegistry:
    """Track entities across a long task."""

    def __init__(self) -> None:
        self._entities: dict[str, EntityRecord] = {}
        self._conflicts: list[dict[str, Any]] = []

    @staticmethod
    def _key(name: str, entity_type: str) -> str:
        return f"{entity_type}::{name.strip().lower()}"

    def register(
        self, name: str, entity_type: str = "other"
    ) -> EntityRecord:
        key = self._key(name, entity_type)
        record = self._entities.get(key)
        if record is None:
            record = EntityRecord(
                name=name.strip(), entity_type=entity_type
            )
            self._entities[key] = record
        return record

    def mention(
        self,
        name: str,
        entity_type: str,
        event_ref: str,
        attributes: dict[str, str] | None = None,
        *,
        source: str = "",
        verified: bool = False,
        at: str = "",
    ) -> list[dict[str, Any]]:
        """Record a mention; returns conflicts found."""
        record = self.register(name, entity_type)
        record.mentions.append(event_ref)
        conflicts = []
        for attr, value in (attributes or {}).items():
            conflict = record.set_attribute(
                attr, value, source=source,
                verified=verified, at=at,
            )
            if conflict is not None:
                conflict["event_ref"] = event_ref
                self._conflicts.append(conflict)
                conflicts.append(conflict)
        return conflicts

    def get(
        self, name: str, entity_type: str = "other"
    ) -> EntityRecord | None:
        return self._entities.get(
            self._key(name, entity_type)
        )

    def conflicts(self) -> list[dict[str, Any]]:
        return list(self._conflicts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": {
                k: v.to_dict()
                for k, v in self._entities.items()
            },
            "conflicts": list(self._conflicts),
        }

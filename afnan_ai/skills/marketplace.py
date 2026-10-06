"""Skill marketplace metadata architecture.

Readies skills for a future internal/private marketplace:
category, version, author, compatibility, permissions,
dependencies, rating/quality and verification status.
Untrusted imported skills are never trusted by default —
marketplace trust derives from verification, not listing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afnan_ai.redaction import redact_text

CATEGORIES = (
    "research",
    "communication",
    "files",
    "browser",
    "computer",
    "data",
    "productivity",
    "system",
    "other",
)


@dataclass
class MarketplaceEntry:
    skill_id: str
    version: str
    name: str = ""
    category: str = "other"
    author: str = ""
    description: str = ""
    risk_level: str = "read_only"
    verification_status: str = "unverified"
    trust_level: float = 0.0
    rating: float | None = None  # 0..5, community quality
    downloads: int = 0
    compatibility: dict[str, Any] = field(
        default_factory=dict
    )
    permissions: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            self.category = "other"
        if self.rating is not None:
            self.rating = max(0.0, min(5.0, float(self.rating)))

    def trusted(self) -> bool:
        """Marketplace trust: verified + non-trivial trust."""
        return (
            self.verification_status == "verified"
            and self.trust_level >= 0.6
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "name": redact_text(self.name)[:160],
            "category": self.category,
            "author": redact_text(self.author)[:120],
            "description": redact_text(self.description)[:300],
            "risk_level": self.risk_level,
            "verification_status": self.verification_status,
            "trust_level": self.trust_level,
            "rating": self.rating,
            "downloads": self.downloads,
            "compatibility": dict(self.compatibility),
            "permissions": list(self.permissions),
            "dependencies": list(self.dependencies),
            "trusted": self.trusted(),
        }


def entry_from_manifest(
    manifest: Any, *, rating: float | None = None
) -> MarketplaceEntry:
    """Build a marketplace entry from a SkillManifest."""
    to_dict = getattr(manifest, "to_dict", None)
    data = to_dict() if to_dict else {}
    compat = data.get("compatibility", {})
    if hasattr(compat, "to_dict"):
        compat = compat.to_dict()
    return MarketplaceEntry(
        skill_id=data.get("skill_id", ""),
        version=data.get("version", "1.0.0"),
        name=data.get("name", ""),
        category=data.get("category", "other") or "other",
        author=data.get("author", ""),
        description=data.get("description", ""),
        risk_level=data.get("risk_level", "read_only"),
        verification_status=data.get(
            "verification_status", "unverified"
        ),
        trust_level=float(data.get("trust_level", 0.0)),
        rating=rating,
        compatibility=dict(compat or {}),
        permissions=list(
            data.get("required_permissions", [])
        ),
        dependencies=list(
            data.get("required_tools", [])
        )
        + list(data.get("skill_dependencies", [])),
    )

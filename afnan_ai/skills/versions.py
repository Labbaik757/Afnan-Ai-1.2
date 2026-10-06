"""Semantic versioning for skills.

MAJOR.MINOR.PATCH with change tracking: behavior,
dependency, permission, schema and security changes are
classified per version bump.  Breaking input/output
changes are detected so existing tasks are never silently
migrated to an incompatible version.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_SEMVER = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$"
)


def parse(version: str) -> tuple[int, int, int]:
    """Parse MAJOR.MINOR.PATCH; raises ValueError if invalid."""
    match = _SEMVER.match(str(version or "").strip())
    if not match:
        raise ValueError(
            f"invalid semantic version: {version!r}"
        )
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
    )


def is_valid(version: str) -> bool:
    try:
        parse(version)
        return True
    except ValueError:
        return False


def bump(
    version: str, kind: str = "patch"
) -> str:
    """Bump a version: kind in {major, minor, patch}."""
    major, minor, patch = parse(version)
    kind = kind.lower()
    if kind == "major":
        return f"{major + 1}.0.0"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def compare(a: str, b: str) -> int:
    """-1 if a<b, 0 if equal, 1 if a>b."""
    pa, pb = parse(a), parse(b)
    return (pa > pb) - (pa < pb)


def latest(versions: list[str]) -> str:
    """The highest valid version."""
    valid = [v for v in versions if is_valid(v)]
    if not valid:
        raise ValueError("no valid versions")
    return max(valid, key=parse)


@dataclass
class VersionChange:
    """What changed between two skill versions."""

    from_version: str
    to_version: str
    behavior_changed: bool = False
    dependencies_changed: bool = False
    permissions_changed: bool = False
    schema_changed: bool = False
    security_changed: bool = False
    breaking: bool = False
    notes: list[str] = field(default_factory=list)

    def bump_kind(self) -> str:
        if self.breaking or self.security_changed:
            return "major"
        if (
            self.behavior_changed
            or self.permissions_changed
            or self.schema_changed
        ):
            return "minor"
        return "patch"

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "behavior_changed": self.behavior_changed,
            "dependencies_changed": self.dependencies_changed,
            "permissions_changed": self.permissions_changed,
            "schema_changed": self.schema_changed,
            "security_changed": self.security_changed,
            "breaking": self.breaking,
            "suggested_bump": self.bump_kind(),
            "notes": list(self.notes),
        }


def diff_schemas(
    old: dict[str, Any], new: dict[str, Any]
) -> dict[str, Any]:
    """Detect breaking input/output schema changes."""
    old_props = set((old.get("properties") or {}))
    new_props = set((new.get("properties") or {}))
    old_req = set(old.get("required") or [])
    new_req = set(new.get("required") or [])
    removed = old_props - new_props
    added_required = new_req - old_req
    return {
        "removed_properties": sorted(removed),
        "added_required": sorted(added_required),
        "breaking": bool(removed or added_required),
    }


def diff_skills(old: Any, new: Any) -> VersionChange:
    """Classify changes between two Skill versions."""
    change = VersionChange(
        from_version=getattr(old, "version", "?"),
        to_version=getattr(new, "version", "?"),
    )
    if getattr(old, "steps", None) != getattr(
        new, "steps", None
    ):
        change.behavior_changed = True
        change.notes.append("steps changed")
    old_deps = getattr(old, "dependencies", None)
    new_deps = getattr(new, "dependencies", None)
    if old_deps is not None and new_deps is not None:
        if old_deps.to_dict() != new_deps.to_dict():
            change.dependencies_changed = True
            change.notes.append("dependencies changed")
    if getattr(old, "risk", None) != getattr(
        new, "risk", None
    ):
        change.permissions_changed = True
        change.security_changed = True
        change.notes.append("risk level changed")
    schema_diff = diff_schemas(
        getattr(old, "input_schema", None) or {},
        getattr(new, "input_schema", None) or {},
    )
    if schema_diff["breaking"]:
        change.schema_changed = True
        change.breaking = True
        change.notes.append(
            "breaking input schema change: "
            + ", ".join(
                schema_diff["removed_properties"]
                + schema_diff["added_required"]
            )[:160]
        )
    elif (
        getattr(old, "input_schema", None)
        != getattr(new, "input_schema", None)
        or getattr(old, "output_schema", None)
        != getattr(new, "output_schema", None)
    ):
        change.schema_changed = True
        change.notes.append("schema changed (compatible)")
    return change

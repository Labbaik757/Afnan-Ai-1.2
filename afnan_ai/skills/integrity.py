"""Skill integrity: hashes, signatures, tamper detection.

Every published skill carries an integrity hash over its
canonical manifest + steps.  The registry re-verifies on
load and on a schedule; a tampered skill (hash mismatch,
unexpected version, dependency change, manifest mismatch)
is automatically disabled — never executed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any


def canonical_json(data: Any) -> str:
    """Deterministic JSON for hashing."""
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, default=str,
    )


def manifest_hash(manifest_dict: dict[str, Any]) -> str:
    """Hash over the manifest minus volatile/meta fields.

    ``integrity_hash`` (self), ``created_at`` and
    ``signature`` are excluded: they change without the
    definition changing.
    """
    data = {
        k: v
        for k, v in (manifest_dict or {}).items()
        if k
        not in (
            "integrity_hash",
            "created_at",
            "signature",
        )
    }
    digest = hashlib.sha256(
        canonical_json(data).encode("utf-8")
    ).hexdigest()
    return "sha256:" + digest


def skill_hash(skill: Any) -> str:
    """Integrity hash for a Skill model instance."""
    to_dict = getattr(skill, "to_dict", None)
    data = to_dict() if to_dict else {}
    return manifest_hash(
        data if isinstance(data, dict) else {}
    )


def verify_skill(skill: Any, expected_hash: str) -> dict[str, Any]:
    """Check a skill against its recorded hash."""
    actual = skill_hash(skill)
    ok = bool(expected_hash) and hmac.compare_digest(
        actual, expected_hash
    )
    return {
        "ok": ok,
        "expected": expected_hash,
        "actual": actual,
        "tampered": not ok,
    }


def sign_manifest(
    manifest_dict: dict[str, Any], secret: bytes
) -> str:
    """HMAC signature for a manifest (publisher-side)."""
    data = {
        k: v
        for k, v in (manifest_dict or {}).items()
        if k not in ("integrity_hash", "signature")
    }
    return hmac.new(
        secret,
        canonical_json(data).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_signature(
    manifest_dict: dict[str, Any],
    signature: str,
    secret: bytes,
) -> bool:
    expected = sign_manifest(manifest_dict, secret)
    return bool(signature) and hmac.compare_digest(
        expected, signature
    )

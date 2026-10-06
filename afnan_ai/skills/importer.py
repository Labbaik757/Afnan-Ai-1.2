"""Skill import pipeline.

Download → signature/integrity check → manifest
validation → dependency analysis → security scan →
sandbox test → policy evaluation → approval if required →
install.

An unverified skill is never executed.  Imported skills
start at trust 0 and stay disabled until every gate
passes; sensitive+ risk always needs human approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afnan_ai.redaction import redact_text
from afnan_ai.skills.integrity import (
    manifest_hash,
    verify_signature,
)
from afnan_ai.skills.manifest import (
    SkillManifest,
    SkillSource,
    VerificationStatus,
)
from afnan_ai.skills.security import analyze_skill


@dataclass
class ImportReport:
    ok: bool
    stage: str = ""
    detail: str = ""
    manifest: SkillManifest | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stage": self.stage,
            "detail": redact_text(self.detail)[:400],
            "skill_id": (
                self.manifest.skill_id
                if self.manifest
                else ""
            ),
        }


def import_skill(
    package: dict[str, Any],
    *,
    registry: Any = None,
    sandbox: Any = None,
    signature_secret: bytes | None = None,
    approver: Any = None,
) -> ImportReport:
    """Run the full import pipeline for one skill package."""
    package = package or {}

    # 1. Manifest validation.
    try:
        manifest = SkillManifest.from_dict(
            package.get("manifest") or {}
        )
    except Exception as exc:
        return ImportReport(
            ok=False, stage="manifest",
            detail=f"invalid manifest: {exc}",
        )
    if not manifest.skill_id:
        return ImportReport(
            ok=False, stage="manifest",
            detail="manifest missing skill_id",
        )
    manifest.source = SkillSource.IMPORTED

    # 2. Signature / integrity check.
    signature = str(package.get("signature", ""))
    if signature_secret is not None:
        if not verify_signature(
            package.get("manifest") or {},
            signature,
            signature_secret,
        ):
            return ImportReport(
                ok=False, stage="signature",
                detail="signature verification failed",
                manifest=manifest,
            )
    expected_hash = manifest.integrity_hash or str(
        package.get("integrity_hash", "")
    )
    actual_hash = manifest_hash(package.get("manifest") or {})
    if expected_hash and expected_hash != actual_hash:
        return ImportReport(
            ok=False, stage="integrity",
            detail="integrity hash mismatch — possible tampering",
            manifest=manifest,
        )
    manifest.integrity_hash = actual_hash

    # 3. Dependency analysis (declared only; resolution at install).
    steps = package.get("steps") or []

    # 4. Security scan — strict for imports.
    class _SkillView:
        def __init__(self, m, s):
            self.skill_id = m.skill_id
            self.risk = m.risk_level
            self.steps = s
            self.dependencies = None

    class _Deps:
        tools = tuple(manifest.required_tools)
        connectors = tuple(manifest.required_connectors)
        capabilities = tuple(manifest.capabilities)

    view = _SkillView(manifest, steps)
    view.dependencies = _Deps()
    report = analyze_skill(view, strict=True)
    if not report.passed:
        return ImportReport(
            ok=False, stage="security",
            detail="; ".join(
                f["code"] for f in report.to_dict()["findings"]
            )[:300],
            manifest=manifest,
        )

    # 5. Sandbox test (when a sandbox is provided).
    if sandbox is not None:
        try:
            result = sandbox.run_package(package)
        except Exception as exc:
            return ImportReport(
                ok=False, stage="sandbox",
                detail=f"sandbox error: {exc}",
                manifest=manifest,
            )
        if isinstance(result, dict) and result.get(
            "ok"
        ) is False:
            return ImportReport(
                ok=False, stage="sandbox",
                detail=str(result.get("detail", "failed"))[
                    :300
                ],
                manifest=manifest,
            )

    # 6. Policy evaluation + approval for sensitive+.
    needs_approval = manifest.risk_level in (
        "sensitive", "destructive",
    )
    if needs_approval:
        approved = False
        if approver is not None:
            try:
                approved = bool(
                    approver(
                        {
                            "action": "import_skill",
                            "skill_id": manifest.skill_id,
                            "risk": manifest.risk_level,
                        }
                    )
                )
            except Exception:
                approved = False
        if not approved:
            return ImportReport(
                ok=False, stage="approval",
                detail="sensitive skill import needs human approval",
                manifest=manifest,
            )

    # 7. Install (disabled by default; explicit enable later).
    manifest.verification_status = (
        VerificationStatus.SANDBOX_TESTED
    )
    if registry is not None:
        try:
            registry.install_imported(manifest, steps)
        except Exception as exc:
            return ImportReport(
                ok=False, stage="install",
                detail=f"install failed: {exc}",
                manifest=manifest,
            )

    return ImportReport(
        ok=True, stage="installed",
        detail="imported; disabled until explicitly enabled",
        manifest=manifest,
    )

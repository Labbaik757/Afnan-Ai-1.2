"""Tests for the Artifact System.

Covers: creation (all builder types), update, versioning,
rollback on failed update, corruption detection, content
verification, source/evidence tracking, concurrent access,
background incremental generation, browser-to-artifact and
connector-to-artifact workflows, sensitive-data redaction,
approval-gated export/delete, and a full end-to-end
deliverable run through the real AgentLoop.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from afnan_ai.agent import AfnanAgent
from afnan_ai.artifacts import (
    ArtifactError,
    ArtifactManager,
    ArtifactStatus,
    ArtifactType,
    ArtifactVerifier,
    ArtifactWorkspace,
    VerificationState,
    builder_for,
)
from afnan_ai.artifacts.models import SourceRef
from afnan_ai.llm.base import LLMProvider

from test_browser_advanced import QueueLLM, plan_json, step

COMPLETE = '{"goal": "done", "complete": true, "steps": []}'


def _manager(**kwargs):
    tmp = tempfile.mkdtemp()
    return (
        ArtifactManager(
            ArtifactWorkspace(os.path.join(tmp, "artifacts")),
            audit_path=os.path.join(tmp, "audit.jsonl"),
            **kwargs,
        ),
        tmp,
    )


class TestBuilders(unittest.TestCase):
    def test_all_types_produce_bytes(self):
        data = {
            "title": "Report",
            "introduction": "Intro here",
            "sections": [
                {"heading": "Findings", "body": "Body text"}
            ],
            "headers": ["a", "b"],
            "rows": [[1, 2], [3, 4]],
            "data": {"k": "v"},
            "slides": [
                {"heading": "S1", "bullets": ["one", "two"]}
            ],
            "kind": "chart",
            "values": [1, 5, 3],
        }
        expectations = {
            "document": (b"# Report", ".md"),
            "report": (b"# Report", ".md"),
            "html": (b"<!DOCTYPE html>", ".html"),
            "pdf": (b"%PDF", ".pdf"),
            "spreadsheet": (b"a,b", ".csv"),
            "structured_data": (b'"k"', ".json"),
            "code_output": (b"Report", ".txt"),
            "presentation": (b"<section", ".html"),
            "image": (b"\x89PNG", ".png"),
        }
        for kind, (marker, ext) in expectations.items():
            builder = builder_for(kind)
            out = builder.build(data)
            self.assertIn(marker, out, kind)
            self.assertEqual(builder.extension, ext, kind)

    def test_custom_type_is_extensible(self):
        self.assertEqual(ArtifactType.coerce("my_custom"),
                         "my_custom")
        # Unknown types fall back to the safe text builder.
        out = builder_for("my_custom").build(
            {"body": "hello"}
        )
        self.assertIn(b"hello", out)

    def test_html_escapes_scripts(self):
        out = builder_for("html").build({
            "title": "T",
            "sections": [
                {"heading": "H",
                 "body": "<script>alert(1)</script>"}
            ],
        })
        self.assertNotIn(b"<script>", out)
        self.assertIn(b"&lt;script&gt;", out)

    def test_pdf_structure(self):
        out = builder_for("pdf").build({
            "title": "T",
            "sections": [
                {"heading": "H", "body": "line (one)"}
            ],
        })
        self.assertTrue(out.startswith(b"%PDF-1.4"))
        self.assertTrue(out.rstrip().endswith(b"%%EOF"))
        self.assertIn(b"startxref", out)


class TestLifecycle(unittest.TestCase):
    def test_create_and_read(self):
        manager, _ = _manager()
        artifact = manager.create(
            name="Q3 Report", artifact_type="report",
            data={"title": "Q3 Report",
                  "sections": [{"heading": "Findings",
                                "body": "Revenue grew"}]},
            description="Quarterly findings",
            source_task_id="task-1", project_id="proj-a",
            goal_id="goal-1",
        )
        self.assertEqual(artifact.current_version, 1)
        self.assertEqual(artifact.status,
                         ArtifactStatus.DRAFT)
        self.assertEqual(artifact.verification,
                         VerificationState.UNVERIFIED)
        self.assertEqual(artifact.source_task_id, "task-1")
        self.assertEqual(artifact.goal_id, "goal-1")
        content = manager.read(artifact.artifact_id,
                               project_id="proj-a")
        self.assertIn(b"Q3 Report", content)
        # Workspace layout: project/category/artifact/v1 file.
        self.assertTrue(
            os.path.isdir(
                os.path.join(
                    manager.workspace.root, "proj-a",
                    "drafts",
                )
            )
        )

    def test_update_versions_preserved(self):
        manager, _ = _manager()
        artifact = manager.create(
            name="Doc", data={"title": "Doc",
                              "sections": [{"heading": "A",
                                            "body": "v1"}]}
        )
        manager.update(
            artifact.artifact_id,
            data={"title": "Doc",
                  "sections": [{"heading": "A",
                                "body": "v2"}]},
            change_note="second draft",
        )
        manager.update(
            artifact.artifact_id,
            data={"title": "Doc",
                  "sections": [{"heading": "A",
                                "body": "v3"}]},
            change_note="third draft",
        )
        artifact = manager.get(artifact.artifact_id)
        self.assertEqual(artifact.current_version, 3)
        self.assertEqual(len(artifact.versions), 3)
        self.assertEqual(artifact.versions[1].change_note,
                         "second draft")
        # Previous versions are traceable and readable.
        self.assertIn(
            b"v1", manager.read(artifact.artifact_id, version=1)
        )
        self.assertIn(
            b"v3", manager.read(artifact.artifact_id)
        )
        # Updating resets verification (stale verdicts die).
        manager.set_verification(
            artifact.artifact_id, VerificationState.VERIFIED
        )
        manager.update(
            artifact.artifact_id,
            data={"title": "Doc",
                  "sections": [{"heading": "A",
                                "body": "v4"}]},
        )
        self.assertEqual(
            manager.get(artifact.artifact_id).verification,
            VerificationState.UNVERIFIED,
        )

    def test_failed_update_restores_previous(self):
        manager, _ = _manager()
        artifact = manager.create(
            name="Doc", data={"title": "Doc",
                              "sections": [{"heading": "A",
                                            "body": "v1"}]}
        )
        original_write = manager._write_version_file

        def _boom(path, content):
            raise OSError("disk exploded")

        manager._write_version_file = _boom
        try:
            with self.assertRaises(OSError):
                manager.update(
                    artifact.artifact_id,
                    data={"title": "Doc",
                          "sections": [{"heading": "A",
                                        "body": "v2"}]},
                )
        finally:
            manager._write_version_file = original_write
        artifact = manager.get(artifact.artifact_id)
        self.assertEqual(artifact.current_version, 1)
        self.assertEqual(len(artifact.versions), 1)
        self.assertIn(
            b"v1", manager.read(artifact.artifact_id)
        )

    def test_rename_duplicate_archive(self):
        manager, _ = _manager()
        artifact = manager.create(
            name="Original",
            data={"title": "T",
                  "sections": [{"heading": "H",
                                "body": "B"}]},
        )
        renamed = manager.rename(artifact.artifact_id,
                                 "Renamed")
        self.assertEqual(renamed.name, "Renamed")
        copy = manager.duplicate(artifact.artifact_id)
        self.assertNotEqual(copy.artifact_id,
                             artifact.artifact_id)
        self.assertEqual(copy.name, "Renamed (copy)")
        self.assertIn(b"# T", manager.read(copy.artifact_id))
        self.assertEqual(copy.current_version, 1)
        archived = manager.archive(artifact.artifact_id)
        self.assertEqual(archived.status,
                         ArtifactStatus.ARCHIVED)
        with self.assertRaises(ArtifactError) as ctx:
            manager.update(artifact.artifact_id,
                           data={"title": "x"})
        self.assertEqual(ctx.exception.code, "immutable")


class TestCorruptionDetection(unittest.TestCase):
    def test_tampered_file_detected(self):
        manager, tmp = _manager()
        artifact = manager.create(
            name="Doc", data={"title": "Doc",
                              "sections": [{"heading": "A",
                                            "body": "v1"}]}
        )
        entry = artifact.latest
        path = manager._version_path(
            artifact, entry, "drafts"
        )
        with open(path, "ab") as handle:
            handle.write(b"TAMPERED")
        with self.assertRaises(ArtifactError) as ctx:
            manager.read(artifact.artifact_id)
        self.assertEqual(ctx.exception.code, "corrupted")


class TestVerification(unittest.TestCase):
    def _artifact(self, manager, body="Revenue grew 12%"):
        return manager.create(
            name="Report", artifact_type="report",
            data={"title": "Q3",
                  "sections": [{"heading": "Findings",
                                "body": body}]},
        )

    def test_verified(self):
        manager, _ = _manager()
        artifact = self._artifact(manager)
        verdict = ArtifactVerifier().verify(
            artifact,
            lambda v: manager.read(artifact.artifact_id,
                                  version=v),
            expected_content=["Revenue"],
            required_sections=["Findings"],
        )
        self.assertEqual(verdict.state,
                         VerificationState.VERIFIED)
        manager.set_verification(artifact.artifact_id,
                                 verdict.state)
        stored = manager.get(artifact.artifact_id)
        self.assertEqual(stored.verification,
                         VerificationState.VERIFIED)
        self.assertEqual(stored.status,
                         ArtifactStatus.VERIFIED)

    def test_failed_when_content_missing(self):
        manager, _ = _manager()
        artifact = manager.create(
            name="Empty", artifact_type="document",
            content=b"",
        )
        verdict = ArtifactVerifier().verify(
            artifact,
            lambda v: manager.read(artifact.artifact_id,
                                  version=v),
        )
        self.assertEqual(verdict.state,
                         VerificationState.FAILED)

    def test_uncertain_when_expectations_unmet(self):
        manager, _ = _manager()
        artifact = self._artifact(manager)
        verdict = ArtifactVerifier().verify(
            artifact,
            lambda v: manager.read(artifact.artifact_id,
                                  version=v),
            expected_content=["nonexistent phrase xyz"],
            required_sections=["Missing Section"],
        )
        self.assertEqual(verdict.state,
                         VerificationState.UNCERTAIN)
        self.assertTrue(verdict.reasons)
        manager.set_verification(artifact.artifact_id,
                                 verdict.state)
        self.assertEqual(
            manager.get(artifact.artifact_id).verification,
            VerificationState.UNCERTAIN,
        )
        # Uncertain is never marked verified.
        self.assertNotEqual(
            manager.get(artifact.artifact_id).status,
            ArtifactStatus.VERIFIED,
        )

    def test_pdf_magic_checked(self):
        manager, _ = _manager()
        artifact = manager.create(
            name="P", artifact_type="pdf",
            data={"title": "P", "sections": []},
        )
        verdict = ArtifactVerifier().verify(
            artifact,
            lambda v: manager.read(artifact.artifact_id,
                                  version=v),
        )
        self.assertEqual(verdict.state,
                         VerificationState.VERIFIED)


class TestSources(unittest.TestCase):
    def test_evidence_tracking(self):
        manager, _ = _manager()
        artifact = manager.create(
            name="Research",
            data={"title": "R",
                  "sections": [{"heading": "H",
                                "body": "Claim A"}]},
        )
        ref = manager.add_source(
            artifact.artifact_id,
            claim="Revenue grew 12%",
            source="https://example.com/report",
            excerpt="...grew 12% YoY...",
        )
        # Unverified by default — model output is never
        # auto-marked as evidence.
        self.assertFalse(ref.verified)
        sources = manager.get_sources(artifact.artifact_id)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].claim, "Revenue grew 12%")
        manager.mark_source_verified(
            artifact.artifact_id, "Revenue grew 12%"
        )
        self.assertTrue(
            manager.get_sources(
                artifact.artifact_id)[0].verified
        )

    def test_source_secrets_redacted(self):
        ref = SourceRef(
            claim="x",
            source="https://h/?api_key=ABC123",
        )
        manager, _ = _manager()
        artifact = manager.create(name="R", data={})
        stored = manager.add_source(
            artifact.artifact_id, ref.claim, ref.source
        )
        self.assertNotIn("ABC123", stored.source)


class TestConcurrency(unittest.TestCase):
    def test_parallel_updates_are_safe(self):
        manager, _ = _manager()
        artifact = manager.create(
            name="Shared",
            data={"title": "S",
                  "sections": [{"heading": "H",
                                "body": "v1"}]},
        )
        errors: list[Exception] = []

        def worker(n):
            for _ in range(5):
                while True:
                    current = manager.get(
                        artifact.artifact_id
                    ).current_version
                    try:
                        manager.update(
                            artifact.artifact_id,
                            data={"title": "S", "sections": [
                                {"heading": "H",
                                 "body": f"writer {n}"}]},
                            expected_version=current,
                        )
                        break
                    except ArtifactError as e:
                        if e.code == "version_conflict":
                            continue  # retry with fresh version
                        errors.append(e)
                        return
                    except Exception as e:  # noqa: BLE001
                        errors.append(e)
                        return

        threads = [
            threading.Thread(target=worker, args=(n,))
            for n in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        final = manager.get(artifact.artifact_id)
        # 1 + 4 workers * 5 updates = 21, every version
        # intact and readable.
        self.assertEqual(final.current_version, 21)
        self.assertEqual(len(final.versions), 21)
        for v in range(1, 22):
            data = manager.read(artifact.artifact_id,
                                version=v)
            self.assertTrue(data.startswith(b"#"))

    def test_stale_expected_version_conflicts(self):
        manager, _ = _manager()
        artifact = manager.create(name="D", data={})
        with self.assertRaises(ArtifactError) as ctx:
            manager.update(
                artifact.artifact_id, data={},
                expected_version=999,
            )
        self.assertEqual(ctx.exception.code,
                         "version_conflict")


class TestBackgroundWorkflow(unittest.TestCase):
    def test_incremental_updates_and_checkpoint_resume(self):
        manager, tmp = _manager()
        artifact = manager.create(
            name="Long Report", artifact_type="report",
            data={"title": "Long",
                  "sections": [{"heading": "Part 1",
                                "body": "draft"}]},
        )
        # Simulate a background worker writing increments.
        for part in ("Part 2", "Part 3"):
            manager.update(
                artifact.artifact_id,
                data={"title": "Long", "sections": [
                    {"heading": f"Part {i + 1}",
                     "body": f"content {i + 1}"}
                    for i in range(3)
                ]},
                change_note=f"added {part}",
            )
        ref = manager.checkpoint_ref(artifact.artifact_id)
        self.assertEqual(ref["version"], 3)
        # "Crash": a fresh manager on the same workspace
        # resumes from the checkpoint.
        resumed = ArtifactManager(
            ArtifactWorkspace(os.path.join(tmp, "artifacts"))
        )
        state = resumed.get(ref["artifact_id"])
        self.assertEqual(state.current_version,
                         ref["version"])
        resumed.update(
            state.artifact_id,
            data={"title": "Long", "sections": [
                {"heading": f"Part {i + 1}",
                 "body": "final"} for i in range(4)]},
            change_note="added Part 4",
            expected_version=ref["version"],
        )
        final = resumed.get(state.artifact_id)
        self.assertEqual(final.current_version, 4)
        # Incomplete work is never marked verified.
        self.assertNotEqual(final.verification,
                            VerificationState.VERIFIED)
        self.assertNotEqual(final.status,
                            ArtifactStatus.VERIFIED)


class TestBrowserToArtifact(unittest.TestCase):
    def test_research_becomes_verified_report(self):
        # Simulated browser research findings (as the
        # browser tools would return them).
        findings = [
            {"heading": "Market size",
             "body": "The market reached $5B in 2026.",
             "source": "https://example.com/market"},
            {"heading": "Growth",
             "body": "Growth is 12% year over year.",
             "source": "https://example.com/growth"},
        ]
        manager, _ = _manager()
        artifact = manager.create(
            name="Market Research", artifact_type="report",
            data={"title": "Market Research",
                  "introduction": "Findings from web research.",
                  "sections": findings},
            source_task_id="research-task-1",
            category="research",
        )
        for finding in findings:
            manager.add_source(
                artifact.artifact_id, finding["heading"],
                finding["source"],
                excerpt=finding["body"],
            )
        verdict = ArtifactVerifier().verify(
            manager.get(artifact.artifact_id),
            lambda v: manager.read(artifact.artifact_id,
                                  version=v),
            expected_content=["$5B", "12%"],
            required_sections=["Market size", "Growth"],
        )
        manager.set_verification(artifact.artifact_id,
                                 verdict.state)
        self.assertEqual(verdict.state,
                         VerificationState.VERIFIED)
        self.assertEqual(len(manager.get_sources(
            artifact.artifact_id)), 2)


class TestConnectorToArtifact(unittest.TestCase):
    def test_connector_data_becomes_dataset(self):
        from afnan_ai.connectors import (
            AuthType, ConnectionResult, ConnectionStatus,
            Connector, ConnectorRegistry, ConnectorService,
            HealthResult, OperationResult, OperationSpec,
            RiskLevel,
        )

        class WeatherConnector(Connector):
            connector_id = "weather"
            name = "Weather"
            description = "Mock weather service"
            auth_type = AuthType.NONE
            declared_scopes = ("weather.read",)
            operations = (
                OperationSpec(
                    "get_forecast", "Get forecast",
                    RiskLevel.READ, ("weather.read",),
                    {"type": "object", "properties": {},
                     "required": []},
                ),
            )

            def authenticate(self, credentials, context):
                from afnan_ai.connectors import AuthSession
                return AuthSession(
                    auth_type=self.auth_type,
                    secret_ref="weather",
                    scopes=self.declared_scopes,
                )

            def connect(self, context):
                return ConnectionResult(
                    self.connector_id,
                    ConnectionStatus.CONNECTED,
                )

            def health_check(self, context):
                return HealthResult(self.connector_id, True)

            def execute_operation(self, operation,
                                  parameters, context):
                return OperationResult(
                    self.connector_id, operation, "ok",
                    summary="Sunny, 24C",
                    verification_hint="station-7",
                )

        registry = ConnectorRegistry()
        registry.register(WeatherConnector())
        service = ConnectorService(registry)
        service.grant_scopes("weather", ("weather.read",))
        result = service.execute("weather", "get_forecast",
                                 {})
        self.assertEqual(result["status"], "ok")
        # Connector output becomes a structured artifact.
        manager, _ = _manager()
        artifact = manager.create(
            name="Weather Data",
            artifact_type="structured_data",
            data={"data": {
                "forecast": result["summary"],
                "station": result["verification_hint"],
            }},
        )
        payload = json.loads(
            manager.read(artifact.artifact_id)
        )
        self.assertEqual(payload["forecast"], "Sunny, 24C")


class TestRedaction(unittest.TestCase):
    def test_secrets_never_land_in_artifacts(self):
        manager, _ = _manager()
        artifact = manager.create(
            name="Notes",
            data={"title": "N", "sections": [{
                "heading": "Keys",
                "body": "api_key=AKIAIOSFODNN7EXAMPLE and "
                        "password=hunter2-hunter2",
            }]},
        )
        content = manager.read(
            artifact.artifact_id).decode()
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", content)
        self.assertNotIn("hunter2", content)


class TestApprovals(unittest.TestCase):
    def test_delete_needs_human(self):
        manager, _ = _manager()  # no approver
        artifact = manager.create(name="D", data={})
        with self.assertRaises(ArtifactError) as ctx:
            manager.delete(artifact.artifact_id)
        self.assertEqual(ctx.exception.code,
                         "approval_required")
        self.assertTrue(
            ctx.exception.details.get("resumable")
        )

    def test_deny_and_allow(self):
        manager, _ = _manager(approver=lambda req: False)
        artifact = manager.create(name="D", data={})
        with self.assertRaises(ArtifactError) as ctx:
            manager.delete(artifact.artifact_id)
        self.assertEqual(ctx.exception.code,
                         "approval_denied")
        manager.set_approver(lambda req: True)
        deleted = manager.delete(artifact.artifact_id)
        self.assertEqual(deleted.status,
                         ArtifactStatus.DELETED)

    def test_sensitive_export_needs_approval(self):
        manager, _ = _manager()
        artifact = manager.create(
            name="Secret plan", data={"title": "S"},
            sensitive=True,
        )
        with self.assertRaises(ArtifactError) as ctx:
            manager.export(artifact.artifact_id,
                           "/tmp/should-not-exist-xyz")
        self.assertEqual(ctx.exception.code,
                         "approval_required")
        manager.set_approver(lambda req: True)
        dest = os.path.join(tempfile.mkdtemp(), "out.md")
        path = manager.export(artifact.artifact_id, dest)
        self.assertTrue(os.path.exists(path))

    def test_plain_export_without_approver_ok(self):
        manager, _ = _manager()
        artifact = manager.create(
            name="Public", data={"title": "P"}
        )
        dest = os.path.join(tempfile.mkdtemp(), "out.md")
        path = manager.export(artifact.artifact_id, dest)
        self.assertTrue(os.path.exists(path))


class TestEndToEndLoop(unittest.TestCase):
    """Research → draft → verify → verified deliverable,
    driven by the real AgentLoop through artifact tools."""

    def test_agent_builds_and_verifies_report(self):
        replies = [
            plan_json("Write the market report", [
                step("s1", "artifact_create",
                     {"name": "Market Report",
                      "artifact_type": "report",
                      "title": "Market Report",
                      "sections": [
                          {"heading": "Findings",
                           "body": "The market reached $5B."},
                          {"heading": "Outlook",
                           "body": "Growth continues."},
                      ]},
                     "status draft version 1"),
                step("s2", "artifact_add_source",
                     {"artifact_id": "__ART__",
                      "claim": "Market size",
                      "source": "https://example.com/m"},
                     "claim source verified"),
                step("s3", "artifact_verify",
                     {"artifact_id": "__ART__",
                      "expected_content": ["$5B"],
                      "required_sections": ["Findings"]},
                     "state verified"),
            ]),
            COMPLETE,
        ]
        agent = AfnanAgent(
            llm_provider=QueueLLM(replies),
            enable_browser_tools=False,
            enable_screen_tools=False,
            enable_computer_tools=False,
            enable_connector_tools=False,
            enable_skill_tools=False,
            memory_dir=tempfile.mkdtemp(),
            artifact_workspace_dir=tempfile.mkdtemp(),
        )
        # Rewrite the placeholder artifact ids with the id
        # the first tool call actually creates.
        created: list[str] = []
        original_create = agent.get_artifact_manager().create

        def _spy_create(**kwargs):
            artifact = original_create(**kwargs)
            created.append(artifact.artifact_id)
            return artifact

        agent.get_artifact_manager().create = _spy_create

        # Patch the queued step args lazily: the loop calls
        # tools in order, so resolve on first tool use.
        registry = agent.tools
        for tool in ("artifact_add_source",
                     "artifact_verify"):
            original = registry.get(tool)
            run = original.run

            def _patched(arguments, _run=run):
                arguments = dict(arguments or {})
                arguments["artifact_id"] = created[0]
                return _run(arguments)

            original.run = _patched

        result = agent.run_agent_loop("Write the market report")
        self.assertTrue(result.success, result.error)
        self.assertEqual(len(created), 1)
        manager = agent.get_artifact_manager()
        artifact = manager.get(created[0])
        self.assertEqual(artifact.verification,
                         VerificationState.VERIFIED)
        self.assertEqual(artifact.status,
                         ArtifactStatus.VERIFIED)
        content = manager.read(created[0]).decode()
        self.assertIn("Findings", content)
        self.assertIn("$5B", content)


if __name__ == "__main__":
    unittest.main()

"""Production Skill System tests: manifest, semver, dependency
graph, conditions/loops, contracts, integrity, security
analysis, import pipeline, history, observability,
optimization, marketplace, dynamic tool builder, and an
end-to-end skill lifecycle.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__))
)

from afnan_ai.skills import (
    DependencyError,
    DependencyGraph,
    ExecutionLimits,
    ImportReport,
    MarketplaceEntry,
    Skill,
    SkillCompatibility,
    SkillDependencies,
    SkillManifest,
    SkillSource,
    SkillStep,
    StepCondition,
    ToolBuilder,
    ToolProposalStatus,
    VerificationStatus,
    analyze_skill,
    bump,
    check_postconditions,
    check_preconditions,
    compare,
    diff_schemas,
    diff_skills,
    entry_from_manifest,
    evaluate_predicate,
    import_skill,
    is_valid,
    latest,
    loop_items,
    manifest_from_skill,
    manifest_hash,
    parse,
    should_run_step,
    sign_manifest,
    skill_hash,
    suggest_optimizations,
    verify_signature,
    verify_skill,
)
from afnan_ai.skills.history import SkillHistory
from afnan_ai.skills.integrity import canonical_json
from afnan_ai.skills.observability import (
    emit_activity,
    emit_audit,
)


def make_skill(**kw):
    base = dict(
        skill_id="test_skill",
        name="Test Skill",
        description="A test skill.",
        version="1.0.0",
        risk="read_only",
        input_schema={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
        steps=[
            SkillStep(
                step_id="s1", tool="web_search",
                arguments={"query": "{{input.q}}"},
            ),
        ],
        dependencies=SkillDependencies(
            tools=("web_search",), capabilities=("network",)
        ),
    )
    base.update(kw)
    return Skill(**base)


# ------------------------------------------------------------------
# Manifest
# ------------------------------------------------------------------

class ManifestTests(unittest.TestCase):
    def test_manifest_roundtrip(self):
        m = SkillManifest(
            skill_id="s1", name="S", description="d",
            source="agent_generated",
        )
        d = m.to_dict()
        m2 = SkillManifest.from_dict(d)
        self.assertEqual(m2.skill_id, "s1")
        self.assertEqual(m2.source, SkillSource.AGENT_GENERATED)

    def test_source_trust(self):
        self.assertEqual(
            SkillManifest(
                skill_id="a", name="a", description="a",
                source="system",
            ).trust_level(),
            1.0,
        )
        self.assertEqual(
            SkillManifest(
                skill_id="a", name="a", description="a",
                source="imported",
            ).trust_level(),
            0.0,
        )
        imported_verified = SkillManifest(
            skill_id="a", name="a", description="a",
            source="imported",
            verification_status="verified",
        )
        self.assertGreater(
            imported_verified.trust_level(), 0.0
        )

    def test_manifest_from_skill(self):
        skill = make_skill()
        m = manifest_from_skill(skill)
        self.assertEqual(m.skill_id, "test_skill")
        self.assertIn("web_search", m.required_tools)

    def test_invalid_source_rejected(self):
        with self.assertRaises(ValueError):
            make_skill(source="hacker")

    def test_execution_limits_bounded(self):
        lim = ExecutionLimits(
            max_steps=0, max_subskill_depth=-5
        )
        self.assertGreaterEqual(lim.max_steps, 1)
        self.assertGreaterEqual(lim.max_subskill_depth, 0)


# ------------------------------------------------------------------
# Versions
# ------------------------------------------------------------------

class VersionTests(unittest.TestCase):
    def test_parse_and_bump(self):
        self.assertEqual(parse("1.2.3"), (1, 2, 3))
        self.assertEqual(bump("1.2.3", "major"), "2.0.0")
        self.assertEqual(bump("1.2.3", "minor"), "1.3.0")
        self.assertEqual(bump("1.2.3"), "1.2.4")
        with self.assertRaises(ValueError):
            parse("not-a-version")
        self.assertFalse(is_valid("1.2"))

    def test_compare_latest(self):
        self.assertEqual(compare("1.0.0", "2.0.0"), -1)
        self.assertEqual(compare("2.0.0", "2.0.0"), 0)
        self.assertEqual(
            latest(["1.0.0", "1.10.0", "1.2.0"]), "1.10.0"
        )

    def test_breaking_schema_detection(self):
        old = {
            "type": "object",
            "properties": {"a": {}, "b": {}},
            "required": ["a"],
        }
        new = {
            "type": "object",
            "properties": {"a": {}},
            "required": ["a"],
        }
        diff = diff_schemas(old, new)
        self.assertTrue(diff["breaking"])
        self.assertIn("b", diff["removed_properties"])

    def test_diff_skills(self):
        old = make_skill()
        new = make_skill(
            version="2.0.0", risk="destructive",
            steps=[
                SkillStep(
                    step_id="s1", tool="file_delete",
                    arguments={"path": "/tmp/x"},
                )
            ],
        )
        change = diff_skills(old, new)
        self.assertTrue(change.behavior_changed)
        self.assertTrue(change.security_changed)
        self.assertEqual(change.bump_kind(), "major")


# ------------------------------------------------------------------
# Dependency graph
# ------------------------------------------------------------------

class GraphTests(unittest.TestCase):
    def test_circular_rejected(self):
        g = DependencyGraph()
        g.add_skill("a", ["b"])
        g.add_skill("b", ["c"])
        with self.assertRaises(DependencyError) as ctx:
            g.add_skill("c", ["a"])
        self.assertEqual(
            ctx.exception.code, "circular_dependency"
        )
        # Failed add rolled back.
        self.assertNotIn("c", g._edges)

    def test_self_dependency_ignored(self):
        g = DependencyGraph()
        g.add_skill("a", ["a"])
        self.assertEqual(g._edges["a"], set())

    def test_execution_order(self):
        g = DependencyGraph()
        g.add_skill("report", ["research", "verify"])
        g.add_skill("research", [])
        g.add_skill("verify", ["research"])
        order = g.execution_order("report")
        self.assertLess(
            order.index("research"), order.index("verify")
        )
        self.assertLess(
            order.index("verify"), order.index("report")
        )

    def test_missing_dependency_fails_fast(self):
        g = DependencyGraph()
        g.add_skill("a", ["b"])
        with self.assertRaises(DependencyError) as ctx:
            g.check_available("a", {"a"})
        self.assertEqual(
            ctx.exception.code, "missing_skill_dependency"
        )
        g.check_available("a", {"a", "b"})  # no raise

    def test_dependents_and_depth(self):
        g = DependencyGraph()
        g.add_skill("a", ["b"])
        g.add_skill("c", ["b"])
        self.assertEqual(g.dependents("b"), ["a", "c"])
        self.assertEqual(g.max_depth("a"), 1)


# ------------------------------------------------------------------
# Conditions / loops
# ------------------------------------------------------------------

class ConditionTests(unittest.TestCase):
    def test_predicates(self):
        state = {
            "steps": {
                "search": {"found": True, "count": 5,
                            "items": [1, 2, 3]}
            }
        }
        self.assertTrue(
            evaluate_predicate(
                {"path": "steps.search.found",
                 "equals": True},
                state,
            )
        )
        self.assertTrue(
            evaluate_predicate(
                {"path": "steps.search.count", "gt": 3},
                state,
            )
        )
        self.assertFalse(
            evaluate_predicate(
                {"path": "steps.search.count", "lt": 3},
                state,
            )
        )
        # Unknown operator fails closed.
        self.assertFalse(
            evaluate_predicate(
                {"path": "steps.search.count",
                 "frobnicates": 1},
                state,
            )
        )

    def test_should_run(self):
        state = {"steps": {"s": {"found": False}}}
        cond = StepCondition(
            kind="when",
            predicate={"path": "steps.s.found",
                       "equals": True},
        )
        self.assertFalse(should_run_step(cond, state))
        skip = StepCondition(
            kind="skip_when",
            predicate={"path": "steps.s.found",
                       "equals": True},
        )
        self.assertTrue(should_run_step(skip, state))
        self.assertTrue(should_run_step(None, state))

    def test_bounded_loop(self):
        cond = StepCondition(
            kind="loop", over="steps.search.items",
            max_iterations=2,
        )
        state = {
            "steps": {"search": {"items": [1, 2, 3, 4, 5]}}
        }
        items = loop_items(cond, state, hard_max=100)
        self.assertEqual(items, [1, 2])
        # hard_max also enforced
        cond2 = StepCondition(
            kind="loop", over="steps.search.items",
            max_iterations=100,
        )
        self.assertEqual(
            len(loop_items(cond2, state, hard_max=3)), 3
        )

    def test_invalid_condition_kind(self):
        with self.assertRaises(ValueError):
            StepCondition(kind="whenever")


# ------------------------------------------------------------------
# Contracts
# ------------------------------------------------------------------

class ContractTests(unittest.TestCase):
    def test_precondition_missing_input(self):
        skill = make_skill()
        report = check_preconditions(skill, {})
        self.assertFalse(report.passed)
        self.assertIn("input_valid", report.failures())

    def test_precondition_tools(self):
        skill = make_skill()

        class FakeRegistry:
            def list_tools(self):
                return ["other_tool"]

        report = check_preconditions(
            skill, {"q": "x"}, tool_registry=FakeRegistry()
        )
        self.assertFalse(report.passed)
        self.assertIn("tools_available", report.failures())

    def test_precondition_connector_down(self):
        skill = make_skill(
            dependencies=SkillDependencies(
                tools=(), connectors=("gmail",)
            )
        )
        report = check_preconditions(
            skill, {"q": "x"},
            connector_status={"gmail": False},
        )
        self.assertFalse(report.passed)

    def test_postcondition_output_schema(self):
        skill = make_skill()
        bad = check_postconditions(skill, {"output": {}})
        self.assertFalse(bad.passed)
        good = check_postconditions(
            skill, {"output": {"answer": "42"}}
        )
        self.assertTrue(good.passed)

    def test_postcondition_artifacts(self):
        skill = make_skill()
        report = check_postconditions(
            skill,
            {"output": {"answer": "x"}, "artifacts": []},
            expected_artifacts=["report.pdf"],
        )
        self.assertFalse(report.passed)


# ------------------------------------------------------------------
# Integrity
# ------------------------------------------------------------------

class IntegrityTests(unittest.TestCase):
    def test_hash_stable(self):
        skill = make_skill()
        h1 = skill_hash(skill)
        h2 = skill_hash(make_skill())
        self.assertEqual(h1, h2)
        self.assertTrue(h1.startswith("sha256:"))

    def test_tamper_detected(self):
        skill = make_skill()
        h = skill_hash(skill)
        self.assertTrue(verify_skill(skill, h)["ok"])
        skill.description = "tampered!"
        result = verify_skill(skill, h)
        self.assertFalse(result["ok"])
        self.assertTrue(result["tampered"])

    def test_canonical_json_deterministic(self):
        a = canonical_json({"b": 1, "a": 2})
        b = canonical_json({"a": 2, "b": 1})
        self.assertEqual(a, b)

    def test_signature(self):
        secret = b"test-secret"
        manifest = {"skill_id": "s1", "version": "1.0.0"}
        sig = sign_manifest(manifest, secret)
        self.assertTrue(
            verify_signature(manifest, sig, secret)
        )
        self.assertFalse(
            verify_signature(manifest, sig, b"wrong")
        )
        tampered = dict(manifest, version="9.9.9")
        self.assertFalse(
            verify_signature(tampered, sig, secret)
        )


# ------------------------------------------------------------------
# Security analysis
# ------------------------------------------------------------------

class SecurityTests(unittest.TestCase):
    def test_undeclared_tool_blocked(self):
        skill = make_skill(
            steps=[
                SkillStep(
                    step_id="s1", tool="shell_exec",
                    arguments={"cmd": "rm -rf /"},
                )
            ]
        )
        report = analyze_skill(skill)
        self.assertFalse(report.passed)
        self.assertTrue(
            any(
                f.code == "undeclared_tool"
                for f in report.blockers()
            )
        )

    def test_credential_access_strict(self):
        skill = make_skill(
            dependencies=SkillDependencies(
                tools=("web_search",),
                capabilities=("network",),
            ),
            steps=[
                SkillStep(
                    step_id="s1", tool="web_search",
                    arguments={
                        "query": "x",
                        "api_key": "secret!",
                    },
                )
            ],
        )
        report = analyze_skill(skill, strict=True)
        self.assertFalse(report.passed)

    def test_excessive_permission(self):
        skill = make_skill(
            risk="read_only",
            dependencies=SkillDependencies(
                tools=("file_delete",)
            ),
            steps=[
                SkillStep(
                    step_id="s1", tool="file_delete",
                    arguments={"path": "/tmp/x"},
                )
            ],
        )
        report = analyze_skill(skill)
        self.assertFalse(report.passed)
        self.assertTrue(
            any(
                f.code == "excessive_permission"
                for f in report.blockers()
            )
        )

    def test_clean_skill_passes(self):
        skill = make_skill()
        report = analyze_skill(skill)
        self.assertTrue(report.passed)


# ------------------------------------------------------------------
# Import pipeline
# ------------------------------------------------------------------

class ImportTests(unittest.TestCase):
    def _package(self, **over):
        manifest = {
            "skill_id": "imported_s",
            "name": "Imported",
            "description": "d",
            "version": "1.0.0",
            "risk_level": "read_only",
            "required_tools": ["web_search"],
            "capabilities": ["network"],
        }
        pkg = {
            "manifest": manifest,
            "steps": [
                {
                    "step_id": "s1",
                    "tool": "web_search",
                    "arguments": {"query": "x"},
                }
            ],
        }
        pkg.update(over)
        return pkg

    def test_import_ok(self):
        installed = []

        class FakeRegistry:
            def install_imported(self, manifest, steps):
                installed.append(manifest.skill_id)

        report = import_skill(
            self._package(), registry=FakeRegistry()
        )
        self.assertTrue(report.ok)
        self.assertEqual(report.stage, "installed")
        self.assertEqual(installed, ["imported_s"])
        self.assertEqual(
            report.manifest.verification_status,
            VerificationStatus.SANDBOX_TESTED,
        )

    def test_tampered_import_blocked(self):
        pkg = self._package()
        pkg["integrity_hash"] = "sha256:deadbeef"
        report = import_skill(pkg)
        self.assertFalse(report.ok)
        self.assertEqual(report.stage, "integrity")

    def test_bad_signature_blocked(self):
        pkg = self._package()
        pkg["signature"] = "bogus"
        report = import_skill(
            pkg, signature_secret=b"real-secret"
        )
        self.assertFalse(report.ok)
        self.assertEqual(report.stage, "signature")

    def test_sensitive_needs_approval(self):
        pkg = self._package()
        pkg["manifest"]["risk_level"] = "sensitive"
        report = import_skill(pkg, approver=None)
        self.assertFalse(report.ok)
        self.assertEqual(report.stage, "approval")
        report2 = import_skill(
            pkg, approver=lambda req: True
        )
        self.assertTrue(report2.ok)

    def test_malicious_import_blocked(self):
        pkg = self._package()
        pkg["steps"] = [
            {
                "step_id": "s1",
                "tool": "shell_exec",
                "arguments": {"cmd": "evil"},
            }
        ]
        report = import_skill(pkg)
        self.assertFalse(report.ok)
        self.assertEqual(report.stage, "security")


# ------------------------------------------------------------------
# History / optimization
# ------------------------------------------------------------------

class HistoryTests(unittest.TestCase):
    def test_redacted_inputs(self):
        h = SkillHistory()
        rec = h.start(
            "s1", "1.0.0",
            {"q": "hello", "password": "s3cret-value"},
        )
        meta = rec.input_metadata
        self.assertIn("preview", meta["q"])
        self.assertTrue(meta["password"]["redacted"])
        self.assertNotIn("s3cret", str(meta))

    def test_stats(self):
        h = SkillHistory()
        for i in range(4):
            rec = h.start("s1", "1.0.0", {})
            rec.finish(
                "success" if i < 3 else "failed",
                failures=["timeout"] if i == 3 else [],
            )
        stats = h.stats("s1")
        self.assertEqual(stats["runs"], 4)
        self.assertEqual(stats["success_rate"], 0.75)

    def test_optimization_suggestions(self):
        stats = {
            "runs": 10, "success_rate": 0.5,
            "avg_duration_s": 120.0, "total_retries": 25,
            "top_failures": [("timeout", 6)],
        }
        suggestions = suggest_optimizations(stats)
        codes = {s.code for s in suggestions}
        self.assertTrue(
            {"low_success_rate", "slow_execution",
             "excessive_retries",
             "recurring_failure"} <= codes
        )
        # New version, never overwrite.
        self.assertEqual(
            suggest_optimizations({"runs": 0}), []
        )


# ------------------------------------------------------------------
# Observability / marketplace / toolsmith
# ------------------------------------------------------------------

class ObservabilityTests(unittest.TestCase):
    def test_emit_activity(self):
        events = []

        class FakeCenter:
            def emit(self, *a, **k):
                events.append((a, k))

        ok = emit_activity(
            FakeCenter(), "skill_started", "s1",
            "running", task_id="t1",
            details={"note": "x"},
        )
        self.assertTrue(ok)
        self.assertEqual(len(events), 1)

    def test_emit_activity_broken_center(self):
        class Broken:
            def emit(self, *a, **k):
                raise RuntimeError("down")

        self.assertFalse(
            emit_activity(Broken(), "skill_started", "s1",
                          "x")
        )

    def test_emit_audit(self):
        records = []

        class FakeAudit:
            def record(self, rec):
                records.append(rec)

        ok = emit_audit(
            FakeAudit(), "publication", "s1",
            "published v2",
        )
        self.assertTrue(ok)
        self.assertEqual(records[0]["event"], "publication")
        # Unknown audit event → False, not an exception.
        self.assertFalse(
            emit_audit(FakeAudit(), "nope", "s1", "x")
        )


class MarketplaceTests(unittest.TestCase):
    def test_entry_trust(self):
        m = SkillManifest(
            skill_id="s1", name="S", description="d",
            source="imported",
        )
        entry = entry_from_manifest(m)
        self.assertFalse(entry.trusted())
        m2 = SkillManifest(
            skill_id="s2", name="S", description="d",
            source="system",
            verification_status="verified",
        )
        entry2 = entry_from_manifest(m2)
        self.assertTrue(entry2.trusted())

    def test_category_normalized(self):
        e = MarketplaceEntry(
            skill_id="s", version="1.0.0", category="bogus"
        )
        self.assertEqual(e.category, "other")


class ToolsmithTests(unittest.TestCase):
    def test_propose_composition(self):
        builder = ToolBuilder()
        p = builder.propose(
            "summarize_page", "Summarize a page.",
            composed_of=["page_open", "page_extract"],
            available_tools={"page_open", "page_extract",
                             "web_search"},
        )
        self.assertEqual(
            p.status, ToolProposalStatus.DRAFT
        )
        # Never privileged by default.
        self.assertEqual(p.risk, "read_only")

    def test_privileged_base_rejected(self):
        builder = ToolBuilder()
        p = builder.propose(
            "evil", "d", composed_of=["shell_exec"],
            available_tools={"shell_exec"},
        )
        self.assertEqual(
            p.status, ToolProposalStatus.REJECTED
        )

    def test_unknown_base_rejected(self):
        builder = ToolBuilder()
        p = builder.propose(
            "x", "d", composed_of=["nope_tool"],
            available_tools={"web_search"},
        )
        self.assertEqual(
            p.status, ToolProposalStatus.REJECTED
        )

    def test_pipeline_order(self):
        builder = ToolBuilder()
        p = builder.propose("t", "d", composed_of=[])
        # Security review before sandbox → rejected.
        p2 = builder.mark_security_reviewed(p, passed=True)
        self.assertEqual(
            p2.status, ToolProposalStatus.REJECTED
        )
        p3 = builder.mark_sandbox_tested(p, passed=True)
        p4 = builder.mark_security_reviewed(p3, passed=True)
        self.assertEqual(
            p4.status, ToolProposalStatus.SECURITY_REVIEWED
        )


# ------------------------------------------------------------------
# End-to-end lifecycle
# ------------------------------------------------------------------

class LifecycleTests(unittest.TestCase):
    def test_full_lifecycle(self):
        # 1. Compose.
        skill = make_skill()
        # 2. Manifest.
        manifest = manifest_from_skill(skill)
        self.assertEqual(manifest.skill_id, "test_skill")
        # 3. Security analysis passes.
        report = analyze_skill(skill)
        self.assertTrue(report.passed)
        # 4. Integrity hash recorded.
        h = skill_hash(skill)
        skill.integrity_hash = h
        self.assertTrue(verify_skill(skill, h)["ok"])
        # 5. Preconditions pass.
        pre = check_preconditions(
            skill, {"q": "acme"},
            tool_registry=None,
            connector_status={},
            workspace_ready=True,
        )
        self.assertTrue(pre.passed)
        # 6. Execute (simulated) → postconditions.
        post = check_postconditions(
            skill, {"output": {"answer": "ACME Inc"}},
            expected_artifacts=[],
        )
        self.assertTrue(post.passed)
        # 7. History recorded with redaction.
        history = SkillHistory()
        rec = history.start(
            skill.skill_id, skill.version, {"q": "acme"}
        )
        rec.finish("success", verification="answer present")
        self.assertEqual(
            history.stats("test_skill")["success_rate"], 1.0
        )
        # 8. Version bump for a behavior change.
        new = make_skill(
            version=bump("1.0.0", "minor"),
            steps=[
                SkillStep(
                    step_id="s1", tool="web_search",
                    arguments={"query": "{{input.q}}"},
                ),
                SkillStep(
                    step_id="s2", tool="page_extract",
                    arguments={},
                ),
            ],
        )
        change = diff_skills(skill, new)
        self.assertTrue(change.behavior_changed)
        # 9. Dependency graph: no cycles.
        graph = DependencyGraph()
        graph.add_skill("a", ["b"])
        graph.add_skill("b", [])
        self.assertEqual(
            graph.execution_order("a"), ["b", "a"]
        )


if __name__ == "__main__":
    unittest.main()

"""Long-horizon context tests: tiers, freshness, hierarchy,
loss-aware compression + verification, entities + conflicts,
failure learning, progress, lineage, subagent scoping,
sensitivity, cache, activity, sources, promotion and a
simulated 100+ step task with checkpoint restore.
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__))
)

from afnan_ai.context import (
    ContextItemV2,
    ContextTier,
    EntityRegistry,
    FailureLearner,
    FreshnessTracker,
    ItemKind,
    LineageRegistry,
    LossAwareCompressor,
    ProgressTracker,
    Sensitivity,
    SubGoal,
    SubGoalStatus,
    TierAssigner,
    TrajectoryEvent,
    TrajectoryPhase,
    TrustZone,
    build_activity_summary,
    build_step_summary,
    build_subagent_scope,
    build_subtask_summary,
    build_task_summary,
    collect_all,
    filter_for_prompt,
    promote_to_memory,
    promotion_candidates,
    sanitize_items,
    validate_subagent_result,
)
from afnan_ai.context.cache import ContextCache
from afnan_ai.context.compression import CompressionVerifier
from afnan_ai.context.hierarchy import retrieve_level


def item(kind, text, **kw):
    zone = kw.pop("zone", TrustZone.AGENT_STATE)
    return ContextItemV2(kind=kind, zone=zone, text=text, **kw)


def old_item(kind, text, **kw):
    it = item(kind, text, **kw)
    it.created_at = (
        datetime.now(timezone.utc) - timedelta(hours=5)
    ).isoformat()
    return it


# ------------------------------------------------------------------
# Tiers
# ------------------------------------------------------------------

class TierTests(unittest.TestCase):
    def test_goal_is_hot(self):
        t = TierAssigner()
        self.assertEqual(
            t.tier_of(item(ItemKind.GOAL, "do research")),
            ContextTier.HOT,
        )

    def test_old_observation_is_cold(self):
        t = TierAssigner()
        self.assertEqual(
            t.tier_of(
                old_item(ItemKind.OBSERVATION, "page loaded")
            ),
            ContextTier.COLD,
        )

    def test_summary_is_archived(self):
        t = TierAssigner()
        self.assertEqual(
            t.tier_of(item(ItemKind.SUMMARY, "folded")),
            ContextTier.ARCHIVED,
        )

    def test_pinned_is_hot(self):
        t = TierAssigner(pinned_ids={"ctx-abc"})
        it = item(ItemKind.OBSERVATION, "x")
        it.context_id = "ctx-abc"
        self.assertEqual(t.tier_of(it), ContextTier.HOT)

    def test_working_set(self):
        t = TierAssigner()
        items = [
            item(ItemKind.GOAL, "goal"),
            item(ItemKind.OBSERVATION, "recent", importance=0.9),
            old_item(ItemKind.OBSERVATION, "old"),
        ]
        working = t.working_set(items)
        texts = [i.text for i in working]
        self.assertIn("goal", texts)
        self.assertIn("recent", texts)
        self.assertNotIn("old", texts)


# ------------------------------------------------------------------
# Freshness
# ------------------------------------------------------------------

class FreshnessTests(unittest.TestCase):
    def test_fresh_then_stale(self):
        ft = FreshnessTracker(fresh_s=60, stale_s=120)
        ft.record("browser", "page1")
        self.assertEqual(
            ft.freshness_of("browser", "page1"), "fresh"
        )
        ft.mark_stale("browser", "page1")
        self.assertEqual(
            ft.freshness_of("browser", "page1"), "stale"
        )

    def test_state_change_marks_stale(self):
        ft = FreshnessTracker()
        ft.record("computer", "screen1")
        ft.record("computer", "screen2")
        n = ft.mark_stale("computer")
        self.assertEqual(n, 2)
        self.assertTrue(
            ft.needs_reobservation("computer", "screen1")
        )

    def test_critical_needs_reobserve_on_potentially_stale(self):
        ft = FreshnessTracker(fresh_s=0.01, stale_s=1000)
        ft.record("browser", "p")
        import time

        time.sleep(0.02)
        self.assertEqual(
            ft.freshness_of("browser", "p"),
            "potentially_stale",
        )
        self.assertTrue(
            ft.needs_reobservation(
                "browser", "p", critical=True
            )
        )
        self.assertFalse(
            ft.needs_reobservation(
                "browser", "p", critical=False
            )
        )

    def test_unknown_needs_reobserve(self):
        ft = FreshnessTracker()
        self.assertTrue(
            ft.needs_reobservation("browser", "nope")
        )


# ------------------------------------------------------------------
# Hierarchy
# ------------------------------------------------------------------

class HierarchyTests(unittest.TestCase):
    def _events(self):
        return [
            TrajectoryEvent(
                kind="action", summary="clicked save",
                verified=True,
            ),
            TrajectoryEvent(
                kind="decision",
                summary="chose CSV format",
            ),
            TrajectoryEvent(
                kind="verification", summary="file exists",
                verified=True,
            ),
            TrajectoryEvent(
                kind="failure", summary="timeout",
                failure="locator timed out",
            ),
        ]

    def test_levels(self):
        steps = [
            build_step_summary(self._events(), "step 1"),
            build_step_summary(self._events(), "step 2"),
        ]
        sub = build_subtask_summary(steps, "extract")
        task = build_task_summary([sub], "research", "goal")
        self.assertEqual(task.level, "task")
        self.assertEqual(sub.level, "subtask")
        self.assertEqual(steps[0].level, "step")
        # Decisions folded upward.
        self.assertTrue(
            any("CSV" in d for d in sub.decisions)
        )
        # Raw refs preserved.
        self.assertTrue(task.event_refs)

    def test_retrieve_level(self):
        steps = [build_step_summary(self._events(), "s1")]
        sub = build_subtask_summary(steps, "sub")
        task = build_task_summary([sub], "t", "g")
        found = retrieve_level(task, "step")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].title, "s1")


# ------------------------------------------------------------------
# Compression
# ------------------------------------------------------------------

class CompressionTests(unittest.TestCase):
    def test_protected_kinds_survive(self):
        comp = LossAwareCompressor(keep_recent=2)
        items = [
            item(ItemKind.GOAL, "research ACME"),
            item(ItemKind.CONSTRAINT, "no purchases"),
            item(ItemKind.DECISION, "use CSV"),
            item(ItemKind.FACT, "deadline Monday",
                 zone=TrustZone.USER),
        ] + [
            old_item(ItemKind.OBSERVATION, f"page {i}")
            for i in range(20)
        ]
        kept, report = comp.compress(items)
        self.assertTrue(report.ok)
        self.assertLess(len(kept), len(items))
        texts = " ".join(i.text for i in kept)
        self.assertIn("no purchases", texts)
        self.assertIn("use CSV", texts)
        self.assertIn("deadline Monday", texts)
        # A summary folded the old observations.
        self.assertTrue(
            any(
                i.kind == ItemKind.SUMMARY for i in kept
            )
        )

    def test_small_sets_untouched(self):
        comp = LossAwareCompressor(keep_recent=8)
        items = [item(ItemKind.OBSERVATION, f"o{i}") for i in range(5)]
        kept, report = comp.compress(items)
        self.assertEqual(len(kept), 5)
        self.assertTrue(report.ok)

    def test_verifier_detects_loss(self):
        verifier = CompressionVerifier()
        original = [
            item(ItemKind.GOAL, "research ACME corp"),
            item(ItemKind.CONSTRAINT, "never buy anything"),
        ]
        good = verifier.verify(
            original,
            "goal: research ACME corp. constraints: never buy anything.",
        )
        self.assertTrue(good["ok"])
        bad = verifier.verify(original, "did some stuff")
        self.assertFalse(bad["ok"])
        self.assertIn("goal", bad["missing"])


# ------------------------------------------------------------------
# Entities + conflicts
# ------------------------------------------------------------------

class EntityTests(unittest.TestCase):
    def test_conflict_newer_verified_wins(self):
        reg = EntityRegistry()
        reg.mention(
            "ACME", "company", "e1",
            {"deadline": "Friday"},
            source="memory", verified=True,
        )
        conflicts = reg.mention(
            "ACME", "company", "e2",
            {"deadline": "Monday"},
            source="verified_result", verified=True,
        )
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(
            conflicts[0]["resolution"], "newer_verified_wins"
        )
        rec = reg.get("ACME", "company")
        self.assertEqual(
            rec.attributes["deadline"]["value"], "Monday"
        )

    def test_unverified_does_not_overwrite_verified(self):
        reg = EntityRegistry()
        reg.mention(
            "ACME", "company", "e1",
            {"deadline": "Monday"},
            source="verified_result", verified=True,
        )
        conflicts = reg.mention(
            "ACME", "company", "e2",
            {"deadline": "Friday"},
            source="webpage", verified=False,
        )
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(
            conflicts[0]["resolution"], "kept_verified_old"
        )
        rec = reg.get("ACME", "company")
        self.assertEqual(
            rec.attributes["deadline"]["value"], "Monday"
        )

    def test_no_conflict_same_value(self):
        reg = EntityRegistry()
        reg.mention("f", "file", "e1", {"path": "/a"},
                    verified=True)
        conflicts = reg.mention("f", "file", "e2",
                                {"path": "/a"}, verified=True)
        self.assertEqual(conflicts, [])


# ------------------------------------------------------------------
# Failure learning
# ------------------------------------------------------------------

class LearningTests(unittest.TestCase):
    def test_ineffective_after_threshold(self):
        fl = FailureLearner(failure_threshold=3)
        for _ in range(2):
            fl.record_failure("xpath locator", "timeout")
        self.assertFalse(fl.is_ineffective("xpath locator"))
        fl.record_failure("xpath locator", "timeout")
        self.assertTrue(fl.is_ineffective("xpath locator"))
        guidance = fl.guidance()
        self.assertTrue(
            any("xpath locator" in g for g in guidance)
        )

    def test_success_rehabilitates(self):
        fl = FailureLearner(failure_threshold=2)
        fl.record_failure("s", "x")
        fl.record_failure("s", "x")
        self.assertTrue(fl.is_ineffective("s"))
        fl.record_success("s")
        self.assertFalse(fl.is_ineffective("s"))


# ------------------------------------------------------------------
# Progress
# ------------------------------------------------------------------

class ProgressTests(unittest.TestCase):
    def test_honest_progress(self):
        tr = ProgressTracker()
        subs = [
            SubGoal("a", "collect", SubGoalStatus.COMPLETED),
            SubGoal("b", "verify", SubGoalStatus.PENDING,
                    depends_on=("a",)),
            SubGoal("c", "report", SubGoalStatus.PENDING,
                    depends_on=("b",)),
        ]
        report = tr.report(subs)
        self.assertAlmostEqual(report.percent, 33.3, places=1)
        self.assertEqual(len(report.remaining), 2)
        # Only "b" is ready: "a" verified, "c" blocked on "b".
        ready = tr.ready_subgoals(subs)
        self.assertEqual([s.subgoal_id for s in ready], ["b"])

    def test_no_fake_percent_without_subgoals(self):
        tr = ProgressTracker()
        report = tr.report([])
        self.assertIsNone(report.percent)
        self.assertIn("0 milestones", report.summary_text())

    def test_blockers(self):
        tr = ProgressTracker()
        tr.add_blocker("waiting for approval")
        tr.set_pending_approvals(2)
        report = tr.report([])
        self.assertIn("approval", report.summary_text())
        tr.clear_blocker("waiting for approval")
        self.assertEqual(tr.report([]).blockers, [])


# ------------------------------------------------------------------
# Lineage
# ------------------------------------------------------------------

class LineageTests(unittest.TestCase):
    def test_lineage_tracking(self):
        reg = LineageRegistry()
        lin = reg.register(
            "art-1", task_id="t1", subtask_id="s1",
            source_refs=["evt-1"],
        )
        reg.record_action("art-1", "evt-2")
        lin.add_version("draft updated")
        lin.mark_verified("checked against sources")
        d = lin.to_dict()
        self.assertTrue(d["verified"])
        self.assertEqual(d["versions"][0]["version"], 1)
        self.assertIn("evt-2", d["action_refs"])
        self.assertEqual(reg.unverified(), [])


# ------------------------------------------------------------------
# Scoping
# ------------------------------------------------------------------

class ScopingTests(unittest.TestCase):
    def test_subagent_gets_minimum(self):
        items = [
            item(ItemKind.GOAL, "research"),
            item(ItemKind.FACT, "deadline Monday",
                 zone=TrustZone.USER),
            item(ItemKind.SECRET if hasattr(ItemKind, "SECRET")
                 else ItemKind.FACT, "token",
                 sensitivity=Sensitivity.SECRET),
            item(ItemKind.FAILURE, "parent failed X"),
            item(ItemKind.OBSERVATION, "page text",
                 zone=TrustZone.UNTRUSTED_EXTERNAL),
        ]
        scope = build_subagent_scope(
            goal="extract data", items=items,
            allowed_tools=["browser_read"],
        )
        kinds = [i.kind.value for i in scope.items]
        self.assertIn("goal", kinds)
        self.assertIn("fact", kinds)
        # Secret excluded, parent failure excluded.
        self.assertNotIn("failure", kinds)
        for i in scope.items:
            self.assertNotEqual(
                i.sensitivity, Sensitivity.SECRET
            )

    def test_result_validation(self):
        ok = validate_subagent_result(
            {"summary": "extracted 3 rows", "verified": True}
        )
        self.assertTrue(ok["ok"])
        bad = validate_subagent_result({"summary": ""})
        self.assertFalse(bad["ok"])
        leaky = validate_subagent_result(
            {"summary": "here is the api_key abc123"}
        )
        self.assertFalse(leaky["ok"])


# ------------------------------------------------------------------
# Sensitivity
# ------------------------------------------------------------------

class SensitivityTests(unittest.TestCase):
    def test_prompt_filter(self):
        items = [
            item(ItemKind.FACT, "public fact",
                 sensitivity=Sensitivity.PUBLIC),
            item(ItemKind.FACT, "pw=123",
                 sensitivity=Sensitivity.SECRET),
        ]
        safe = filter_for_prompt(items)
        self.assertEqual(len(safe), 1)
        self.assertIn("public", safe[0].text)

    def test_sanitize_replaces_secret(self):
        items = [
            item(ItemKind.FACT, "token abc",
                 sensitivity=Sensitivity.SECRET,
                 source="vault"),
        ]
        clean = sanitize_items(items)
        self.assertNotIn("abc", clean[0].text)
        self.assertIn("secret", clean[0].text)

    def test_expired_excluded(self):
        it = item(ItemKind.FACT, "old")
        it.expires_at = "2000-01-01T00:00:00+00:00"
        self.assertTrue(it.is_expired())
        self.assertEqual(filter_for_prompt([it]), [])


# ------------------------------------------------------------------
# Cache
# ------------------------------------------------------------------

class CacheTests(unittest.TestCase):
    def test_hit_and_invalidate(self):
        cache = ContextCache(default_ttl_s=60)
        calls = []
        v1 = cache.get_or_compute(
            "q", lambda: calls.append(1) or "result"
        )
        v2 = cache.get_or_compute(
            "q", lambda: calls.append(1) or "result"
        )
        self.assertEqual(v1, v2)
        self.assertEqual(len(calls), 1)
        cache.invalidate(reason="state change")
        cache.get_or_compute(
            "q", lambda: calls.append(1) or "result"
        )
        self.assertEqual(len(calls), 2)

    def test_critical_always_misses(self):
        cache = ContextCache()
        cache.put("q", "cached")
        self.assertIsNone(
            cache.get("q", critical=True)
        )
        self.assertEqual(cache.get("q"), "cached")

    def test_ttl_expiry(self):
        import time

        cache = ContextCache(default_ttl_s=0.02)
        cache.put("q", "v")
        time.sleep(0.03)
        self.assertIsNone(cache.get("q"))


# ------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------

class SourceTests(unittest.TestCase):
    def test_collect_all(self):
        items = collect_all(
            goal="research ACME",
            task_id="t1",
            memories=[{"text": "ACME is a client",
                       "confidence": 0.9}],
            trajectory_events=[
                TrajectoryEvent(
                    kind="verification",
                    summary="source confirmed",
                    verified=True,
                )
            ],
            browser_obs={"summary": "homepage loaded"},
            connector_results=[
                {"summary": "search hits",
                 "connector": "web"}
            ],
            artifacts=[{"artifact_id": "a1",
                         "title": "report"}],
        )
        sources = {i.source for i in items}
        self.assertTrue(
            {"goal", "memory", "trajectory", "browser",
             "connector", "artifacts"} <= sources
        )
        # Connector content is untrusted.
        conn = [i for i in items if i.source == "connector"]
        self.assertEqual(
            conn[0].zone, TrustZone.UNTRUSTED_EXTERNAL
        )

    def test_broken_sources_yield_nothing(self):
        class Broken:
            def to_dict(self):
                raise RuntimeError("boom")

        items = collect_all(agent_state=Broken())
        self.assertEqual(items, [])


# ------------------------------------------------------------------
# Activity + promotion
# ------------------------------------------------------------------

class ActivityPromotionTests(unittest.TestCase):
    def test_activity_summary_redacted(self):
        from afnan_ai.context.progress import ProgressReport

        items = [
            item(ItemKind.FACT, "done X",
                 sensitivity=Sensitivity.SENSITIVE),
            item(ItemKind.FACT, "token",
                 sensitivity=Sensitivity.SECRET),
        ]
        summary = build_activity_summary(
            goal="research",
            progress=ProgressReport(
                completed_milestones=2,
                total_milestones=4, percent=50.0,
            ),
            current_phase="extract",
            next_action="verify sources",
            recent_items=items,
        )
        self.assertIn("50%", summary["progress_text"])
        texts = " ".join(summary["recent_verified_work"])
        self.assertNotIn("token", texts)

    def test_promotion_rules(self):
        items = [
            item(ItemKind.FACT, "verified durable fact",
                 zone=TrustZone.USER, confidence=0.9),
            item(ItemKind.OBSERVATION, "transient screen"),
            item(ItemKind.FACT, "web rumor",
                 zone=TrustZone.UNTRUSTED_EXTERNAL,
                 confidence=0.9),
            item(ItemKind.FACT, "low conf",
                 zone=TrustZone.USER, confidence=0.3),
        ]
        cands = promotion_candidates(items)
        self.assertEqual(len(cands), 1)
        self.assertIn("durable", cands[0].text)

        class FakeStore:
            def __init__(self):
                self.records = []

            def record(self, rec):
                self.records.append(rec)

        store = FakeStore()
        report = promote_to_memory(
            store, items, task_id="t1"
        )
        self.assertEqual(report["promoted"], 1)
        self.assertEqual(
            store.records[0]["provenance"], ""
        )


# ------------------------------------------------------------------
# Long-horizon simulation: 100+ steps + checkpoint restore
# ------------------------------------------------------------------

class LongHorizonSimulationTests(unittest.TestCase):
    def test_100_step_task_with_restore(self):
        import tempfile

        from afnan_ai.context.trajectory import TrajectoryStore

        store = TrajectoryStore(
            os.path.join(tempfile.mkdtemp(), "traj.json")
        )
        task_id = "long-task-1"
        learner = FailureLearner(failure_threshold=3)
        entities = EntityRegistry()
        progress = ProgressTracker()

        # Simulate 110 steps.
        for step in range(110):
            store.record(
                task_id,
                TrajectoryEvent(
                    task_id=task_id,
                    phase=TrajectoryPhase.ACT,
                    kind=(
                        "failure"
                        if step % 37 == 0
                        else "action"
                    ),
                    summary=f"step {step} done",
                    verified=step % 37 != 0,
                    failure=(
                        "locator timeout"
                        if step % 37 == 0
                        else ""
                    ),
                ),
                goal="long research task",
            )
            if step % 37 == 0:
                learner.record_failure(
                    "css locator", "timeout"
                )
                progress.record_failed_attempt()

        # Failure learning kicked in.
        self.assertTrue(learner.is_ineffective("css locator"))
        self.assertTrue(learner.guidance())

        # Checkpoint restore: recover trajectory, rebuild context.
        recovered = store.recover(task_id)
        self.assertEqual(len(recovered), 110)
        items = collect_all(
            goal="long research task",
            task_id=task_id,
            trajectory_events=recovered,
        )
        self.assertTrue(len(items) > 0)

        # Compression keeps the set bounded.
        comp = LossAwareCompressor(keep_recent=8)
        kept, report = comp.compress(items)
        self.assertLessEqual(len(kept), 30)
        self.assertTrue(report.ok)

        # Entity conflict across the long run.
        entities.mention(
            "ACME", "company", "e10",
            {"deadline": "Friday"}, verified=True,
        )
        conflicts = entities.mention(
            "ACME", "company", "e99",
            {"deadline": "Monday"}, verified=True,
        )
        self.assertEqual(len(conflicts), 1)


if __name__ == "__main__":
    unittest.main()

"""Tests for the TLOS-driven voice nudge engine.

Covers: nudge model validation, TLOS template bootstrap,
scan -> nudge creation, graceful LLM failure, dedup,
dial filtering, delivery marking + voice, dial validation,
scheduler registration, and persistence across restarts.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from afnan_ai.proactive import (
    DIALS,
    TLOS_TEMPLATE,
    Nudge,
    TlosNudgeEngine,
    ensure_tlos,
    lookup_scan_engine,
    register_proactive_scan,
)


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------

class FakeLLM:
    def __init__(self, reply: str = "[]", fail: bool = False):
        self.reply = reply
        self.fail = fail
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.fail:
            raise ConnectionError("LLM is down")
        return self.reply


class FakeRecord:
    def __init__(self, content: str, kind: str = "note"):
        self.content = content
        self.kind = kind


class FakeMemory:
    def __init__(self, records=None):
        self._records = list(records or [])

    def list(self, **kwargs):
        return list(self._records)


class FakeAgent:
    def __init__(self, llm=None, memory=None):
        self.llm = llm or FakeLLM()
        self.memory_store = memory or FakeMemory()
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class FakeScheduler:
    """Minimal stand-in with the real schedule_task signature."""

    def __init__(self):
        self.jobs: list[dict] = []

    def schedule_task(self, goal_text, **kwargs):
        from types import SimpleNamespace
        job = SimpleNamespace(
            schedule_id=f"sched_{len(self.jobs)}",
            goal_text=goal_text,
            recurrence=kwargs.get("recurrence"),
            interval_seconds=kwargs.get("interval_seconds"),
            metadata=kwargs.get("metadata") or {},
        )
        self.jobs.append(job)
        return job


def make_engine(tmp, agent=None, dial="low"):
    return TlosNudgeEngine(
        agent or FakeAgent(),
        tlos_path=os.path.join(tmp, "TLOS.md"),
        nudges_path=os.path.join(tmp, "nudges.json"),
        dial=dial,
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

class TestNudgeModel(unittest.TestCase):
    def test_defaults(self):
        n = Nudge(id="", text="Drink water")
        self.assertTrue(n.id.startswith("nudge_"))
        self.assertEqual(n.priority, "medium")
        self.assertEqual(n.category, "reminder")
        self.assertFalse(n.delivered)
        self.assertTrue(n.created_at)

    def test_invalid_priority_falls_back_to_medium(self):
        n = Nudge(id="x", text="hi", priority="URGENT!!")
        self.assertEqual(n.priority, "medium")

    def test_priority_case_insensitive(self):
        n = Nudge(id="x", text="hi", priority="HIGH")
        self.assertEqual(n.priority, "high")

    def test_roundtrip(self):
        n = Nudge(id="n1", text="Take medicine", priority="high",
                  category="health", delivered=True)
        n2 = Nudge.from_dict(n.to_dict())
        self.assertEqual(n2.to_dict(), n.to_dict())

    def test_signature_stable(self):
        a = Nudge(id="a", text="  Take MEDICINE ")
        b = Nudge(id="b", text="take medicine")
        self.assertEqual(a.signature(), b.signature())


class TestTlosTemplate(unittest.TestCase):
    def test_template_has_required_sections(self):
        for section in ("Daily routines", "Goals I'm working on",
                        "Things I don't want to miss", "Preferences"):
            self.assertIn(section, TLOS_TEMPLATE)

    def test_ensure_creates_on_first_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = ensure_tlos(os.path.join(tmp, "TLOS.md"))
            self.assertTrue(os.path.exists(p))
            self.assertIn("Daily routines", p.read_text())

    def test_ensure_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "TLOS.md")
            with open(p, "w") as f:
                f.write("MY CUSTOM TLOS")
            ensure_tlos(p)
            with open(p) as f:
                self.assertEqual(f.read(), "MY CUSTOM TLOS")


class TestScan(unittest.TestCase):
    def test_scan_creates_nudges_from_llm_json(self):
        reply = json.dumps([
            {"text": "Take your medicine", "priority": "high",
             "category": "health"},
            {"text": "Practice Python", "priority": "medium",
             "category": "goal"},
        ])
        agent = FakeAgent(llm=FakeLLM(reply=reply))
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp, agent)
            created = engine.scan()
        self.assertEqual(len(created), 2)
        self.assertEqual(created[0].priority, "high")
        self.assertEqual(created[1].category, "goal")
        self.assertFalse(created[0].delivered)

    def test_scan_single_llm_call(self):
        agent = FakeAgent(llm=FakeLLM(reply="[]"))
        with tempfile.TemporaryDirectory() as tmp:
            make_engine(tmp, agent).scan()
        self.assertEqual(len(agent.llm.prompts), 1)

    def test_scan_prompt_includes_tlos_and_memory(self):
        agent = FakeAgent(
            llm=FakeLLM(reply="[]"),
            memory=FakeMemory([FakeRecord("went for a walk")]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            tlos = os.path.join(tmp, "TLOS.md")
            with open(tlos, "w") as f:
                f.write("Take medicine at 9am")
            engine = TlosNudgeEngine(
                agent, tlos_path=tlos,
                nudges_path=os.path.join(tmp, "n.json"))
            engine.scan()
        prompt = agent.llm.prompts[0]
        self.assertIn("Take medicine at 9am", prompt)
        self.assertIn("went for a walk", prompt)

    def test_scan_graceful_when_llm_down(self):
        agent = FakeAgent(llm=FakeLLM(fail=True))
        with tempfile.TemporaryDirectory() as tmp:
            created = make_engine(tmp, agent).scan()
        self.assertEqual(created, [])

    def test_scan_graceful_on_garbage_reply(self):
        agent = FakeAgent(llm=FakeLLM(reply="not json at all"))
        with tempfile.TemporaryDirectory() as tmp:
            created = make_engine(tmp, agent).scan()
        self.assertEqual(created, [])

    def test_scan_graceful_with_no_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = TlosNudgeEngine(
                None, tlos_path=os.path.join(tmp, "TLOS.md"),
                nudges_path=os.path.join(tmp, "n2.json"))
            self.assertEqual(engine.scan(), [])

    def test_scan_dedupes_identical_nudges(self):
        reply = json.dumps(
            [{"text": "Drink water", "priority": "low"}])
        agent = FakeAgent(llm=FakeLLM(reply=reply))
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp, agent)
            first = engine.scan()
            second = engine.scan()
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 0)
        self.assertEqual(len(engine.pending()), 1)

    def test_scan_skipped_when_dial_off(self):
        agent = FakeAgent(llm=FakeLLM(reply='[{"text": "x"}]'))
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp, agent, dial="off")
            self.assertEqual(engine.scan(), [])
        self.assertEqual(len(agent.llm.prompts), 0)


class TestDelivery(unittest.TestCase):
    def _seed(self, engine):
        reply = json.dumps([
            {"text": "Urgent: take medicine", "priority": "high",
             "category": "health"},
            {"text": "Read a book", "priority": "low",
             "category": "reminder"},
        ])
        engine.agent.llm.reply = reply
        engine.scan()

    def test_low_dial_delivers_only_high(self):
        agent = FakeAgent()
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp, agent, dial="low")
            self._seed(engine)
            delivered = engine.deliver_pending()
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].priority, "high")
        self.assertEqual(agent.spoken, ["Urgent: take medicine"])

    def test_high_dial_delivers_all(self):
        agent = FakeAgent()
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp, agent, dial="high")
            self._seed(engine)
            delivered = engine.deliver_pending()
        self.assertEqual(len(delivered), 2)
        self.assertEqual(len(agent.spoken), 2)

    def test_off_dial_delivers_nothing(self):
        agent = FakeAgent()
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp, agent, dial="low")
            self._seed(engine)
            engine.proactivity_dial = "off"
            self.assertEqual(engine.deliver_pending(), [])
        self.assertEqual(agent.spoken, [])

    def test_delivery_marks_delivered(self):
        agent = FakeAgent()
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp, agent, dial="high")
            self._seed(engine)
            engine.deliver_pending()
            self.assertEqual(engine.pending(), [])
            # second delivery finds nothing new
            self.assertEqual(engine.deliver_pending(), [])

    def test_invalid_dial_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp, dial="low")
            with self.assertRaises(ValueError):
                engine.proactivity_dial = "maximum"
            self.assertEqual(engine.proactivity_dial, "low")

    def test_all_dials_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp)
            for dial in DIALS:
                engine.proactivity_dial = dial
                self.assertEqual(engine.proactivity_dial, dial)


class TestSchedulerIntegration(unittest.TestCase):
    def test_register_creates_interval_schedule(self):
        sched = FakeScheduler()
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp)
            job = register_proactive_scan(
                sched, engine, interval_hours=1)
        self.assertEqual(job.recurrence, "interval")
        self.assertEqual(job.interval_seconds, 3600)
        self.assertTrue(job.metadata.get("proactive_nudge_scan"))
        self.assertEqual(len(sched.jobs), 1)

    def test_lookup_finds_engine(self):
        sched = FakeScheduler()
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp)
            job = register_proactive_scan(sched, engine)
            found = lookup_scan_engine(job.schedule_id)
        self.assertIs(found, engine)

    def test_lookup_unknown_returns_none(self):
        self.assertIsNone(lookup_scan_engine("nope"))


class TestPersistence(unittest.TestCase):
    def test_nudges_survive_restart(self):
        agent = FakeAgent(llm=FakeLLM(
            reply=json.dumps(
                [{"text": "Call mom", "priority": "medium"}])))
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp, agent, dial="high")
            engine.scan()
            engine.deliver_pending()
            # reload from disk
            engine2 = make_engine(tmp, agent, dial="low")
            self.assertEqual(engine2.proactivity_dial, "high")
            self.assertEqual(engine2.pending(), [])


if __name__ == "__main__":
    unittest.main()

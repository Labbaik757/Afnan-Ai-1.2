"""BackgroundTaskRunner + TaskScheduler tests.

Scheduled execution, recurrence, background resume,
crash/restart + checkpoint recovery, approval parking,
timeout, retry limits, failure handling, concurrent task
isolation and audit-trail redaction, plus an end-to-end run
through the real AgentLoop.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from afnan_ai.background_runner import AuditLog, BackgroundTaskRunner
from afnan_ai.llm.base import LLMProvider
from afnan_ai.scheduler import TaskScheduler
from afnan_ai.task_manager import TaskManager

from test_architecture_upgrade import EchoTool
from test_memory_goals_tasks import CapturingLLM


def ok_result():
    return SimpleNamespace(
        status="completed", error=None, state=None, events=[]
    )


def fail_result(code="boom", message="it broke"):
    return SimpleNamespace(
        status="failed",
        error={"code": code, "message": message},
        state=None,
        events=[],
    )


class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = os.path.join(self.tmp.name)
        self.tasks = TaskManager(os.path.join(base, "tasks.json"))
        self.scheduler = TaskScheduler(
            os.path.join(base, "schedules.json"), self.tasks
        )
        self.now = datetime(2026, 10, 5, 10, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_time_schedule_fires_once(self):
        schedule = self.scheduler.schedule_task(
            "send the report",
            run_at=self.now - timedelta(minutes=1),
        )
        created = self.scheduler.tick(now=self.now)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].goal_text, "send the report")
        self.assertEqual(
            self.scheduler.get(schedule.schedule_id).status,
            "completed",
        )
        # A repeated tick never duplicates.
        self.assertEqual(self.scheduler.tick(now=self.now), [])

    def test_future_schedule_does_not_fire_early(self):
        self.scheduler.schedule_task(
            "later", run_at=self.now + timedelta(hours=2)
        )
        self.assertEqual(self.scheduler.tick(now=self.now), [])

    def test_interval_recurrence_advances(self):
        schedule = self.scheduler.schedule_task(
            "poll the inbox",
            run_at=self.now - timedelta(seconds=10),
            recurrence="interval", interval_seconds=60,
        )
        self.assertEqual(len(self.scheduler.tick(now=self.now)), 1)
        nxt = self.scheduler.get(schedule.schedule_id).next_run_at
        self.assertEqual(
            nxt,
            (self.now + timedelta(seconds=50)).isoformat(),
        )
        later = self.now + timedelta(minutes=5)
        self.assertEqual(len(self.scheduler.tick(now=later)), 1)

    def test_daily_recurrence_next_day(self):
        schedule = self.scheduler.schedule_task(
            "morning briefing",
            run_at="2026-10-05T09:30:00+00:00",
            recurrence="daily", time_of_day="09:30",
        )
        self.scheduler.tick(now=self.now)
        self.assertEqual(
            self.scheduler.get(schedule.schedule_id).next_run_at,
            "2026-10-06T09:30:00+00:00",
        )

    def test_pause_and_cancel(self):
        schedule = self.scheduler.schedule_task(
            "x", run_at=self.now - timedelta(minutes=1)
        )
        self.scheduler.pause(schedule.schedule_id)
        self.assertEqual(self.scheduler.tick(now=self.now), [])
        self.scheduler.resume(schedule.schedule_id)
        self.assertEqual(len(self.scheduler.tick(now=self.now)), 1)


class StubCheckpointer:
    def __init__(self, snapshots=None, fail=False):
        self.snapshots = snapshots or {}
        self.fail = fail

    def load(self, ref):
        if self.fail or ref not in self.snapshots:
            raise ValueError("corrupt checkpoint")
        return self.snapshots[ref]


class TestBackgroundRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = os.path.join(self.tmp.name)
        self.tasks_path = os.path.join(base, "tasks.json")
        self.tasks = TaskManager(self.tasks_path)
        self.scheduler = TaskScheduler(
            os.path.join(base, "schedules.json"), self.tasks
        )
        self.audit_path = os.path.join(base, "audit.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def make_runner(self, run_task, **kwargs):
        return BackgroundTaskRunner(
            self.tasks,
            run_task,
            scheduler=self.scheduler,
            audit_log=AuditLog(self.audit_path),
            **kwargs,
        )

    def test_scheduled_task_executes_to_completion(self):
        past = datetime.now(timezone.utc) - timedelta(seconds=5)
        self.scheduler.schedule_task("do it now", run_at=past)
        runner = self.make_runner(lambda task, resume: ok_result())
        done = runner.process_available()
        self.assertEqual(done[0].status, "completed")
        types = [e["type"] for e in runner.events]
        self.assertIn("task_scheduled", types)
        self.assertIn("task_completed", types)

    def test_approval_pending_parks_then_resumes(self):
        calls = []

        def run_task(task, resume):
            calls.append(1)
            if len(calls) == 1:
                return fail_result(
                    "approval_required", "needs a human"
                )
            return ok_result()

        task = self.tasks.enqueue(
            "pay the invoice", max_retries=0
        )
        runner = self.make_runner(run_task)
        done = runner.process_available()
        self.assertEqual(done[0].status, "waiting_for_approval")
        types = [e["type"] for e in runner.events]
        self.assertIn("task_waiting_approval", types)
        resumed = runner.resume_task(task.task_id)
        self.assertEqual(resumed.status, "completed")

    def test_retry_limit_then_failure_events(self):
        runs = []

        def run_task(task, resume):
            runs.append(1)
            return fail_result()

        task = self.tasks.enqueue("flaky", max_retries=1)
        runner = self.make_runner(run_task)
        runner.process_available()
        final = self.tasks.get(task.task_id)
        self.assertEqual(final.status, "failed")
        self.assertEqual(len(runs), 2)  # 1 run + 1 retry
        types = [e["type"] for e in runner.events]
        self.assertIn("task_retry_scheduled", types)
        self.assertIn("task_failed", types)

    def test_runner_exception_becomes_structured_failure(self):
        def run_task(task, resume):
            raise RuntimeError("model exploded")

        task = self.tasks.enqueue("boom task", max_retries=0)
        runner = self.make_runner(run_task)
        runner.process_available()
        final = self.tasks.get(task.task_id)
        self.assertEqual(final.status, "failed")
        self.assertIn("model exploded", final.error)

    def test_crash_restart_recovers_and_resumes_checkpoint(self):
        task = self.tasks.enqueue("long research", max_retries=0)
        self.tasks.claim_next()  # process "dies" while running
        task.checkpoint_ref = "cp-1"
        self.tasks._save()
        captured = {}

        def run_task(one, resume):
            captured["resume"] = resume
            return ok_result()

        # Restart: fresh managers over the same files.
        tasks2 = TaskManager(self.tasks_path)
        runner = BackgroundTaskRunner(
            tasks2,
            run_task,
            checkpointer=StubCheckpointer(
                {"cp-1": {"state": {"task_id": "t-9"}}}
            ),
            audit_log=AuditLog(self.audit_path),
        )
        requeued = runner.recover()
        self.assertEqual(len(requeued), 1)
        done = runner.process_available()
        self.assertEqual(done[0].status, "completed")
        self.assertEqual(
            captured["resume"], {"state": {"task_id": "t-9"}}
        )
        types = [e["type"] for e in runner.events]
        self.assertIn("task_recovered", types)

    def test_corrupt_checkpoint_starts_fresh_with_audit_note(self):
        task = self.tasks.enqueue("resume me", max_retries=0)
        task.checkpoint_ref = "broken"
        runner = BackgroundTaskRunner(
            self.tasks,
            lambda one, resume: ok_result(),
            checkpointer=StubCheckpointer(fail=True),
            audit_log=AuditLog(self.audit_path),
        )
        self.tasks.claim_next()
        # Simulate the crashed claim -> recover -> run fresh.
        runner.recover()
        runner.process_available()
        self.assertEqual(
            self.tasks.get(task.task_id).status, "completed"
        )
        types = [e["type"] for e in runner.events]
        self.assertIn("checkpoint_invalid", types)

    def test_paused_task_resume(self):
        def run_task(task, resume):
            return ok_result()

        task = self.tasks.enqueue("pausable", max_retries=0)
        self.tasks.claim_next()
        self.tasks.pause(task.task_id)
        runner = self.make_runner(run_task)
        resumed = runner.resume_task(task.task_id)
        self.assertEqual(resumed.status, "completed")

    def test_concurrent_tasks_stay_isolated(self):
        barrier = threading.Barrier(2)
        seen = {}
        guard = threading.Lock()

        def run_task(task, resume):
            with guard:
                seen[task.task_id] = task.goal_text
            barrier.wait(timeout=10)
            return SimpleNamespace(
                status="completed", error=None,
                state=SimpleNamespace(task_id=task.task_id),
                events=[],
            )

        one = self.tasks.enqueue("first task", max_retries=0)
        two = self.tasks.enqueue("second task", max_retries=0)
        runner = self.make_runner(run_task, max_concurrent=2)
        done = runner.process_available()
        self.assertEqual(
            {t.task_id for t in done}, {one.task_id, two.task_id}
        )
        self.assertTrue(all(t.status == "completed" for t in done))
        self.assertEqual(seen[one.task_id], "first task")
        self.assertEqual(seen[two.task_id], "second task")
        started = {
            e["task_id"] for e in runner.events
            if e["type"] == "task_started"
        }
        self.assertEqual(started, {one.task_id, two.task_id})

    def test_audit_trail_is_written_and_redacted(self):
        self.tasks.enqueue(
            "look up api_key=supersecretvalue please",
            max_retries=0,
        )
        runner = self.make_runner(lambda t, r: ok_result())
        runner.process_available()
        with open(self.audit_path, encoding="utf-8") as handle:
            raw = handle.read()
        self.assertNotIn("supersecretvalue", raw)
        entries = [
            json.loads(line) for line in raw.splitlines() if line
        ]
        self.assertTrue(
            any(e["type"] == "task_started" for e in entries)
        )

    def test_threaded_start_stop_lifecycle(self):
        self.tasks.enqueue("background job", max_retries=0)
        runner = self.make_runner(
            lambda t, r: ok_result(), poll_interval_s=0.05
        )
        runner.start()
        try:
            deadline = time.time() + 10
            while time.time() < deadline:
                if not self.tasks.list(status="pending"):
                    break
                time.sleep(0.05)
        finally:
            runner.stop()
        self.assertFalse(runner.running)
        self.assertEqual(
            len(self.tasks.list(status="completed")), 1
        )


class SlowLLM(LLMProvider):
    name = "slow"
    display_name = "Slow"
    model = "slow-1"

    def __init__(self, replies, delay=0.3):
        self.replies = list(replies)
        self.delay = delay

    def chat(self, messages):
        raise AssertionError("unused")

    def generate(self, prompt):
        time.sleep(self.delay)
        if not self.replies:
            raise AssertionError("SlowLLM ran out of replies")
        return self.replies.pop(0)


class TestBackgroundEndToEnd(unittest.TestCase):
    def _agent(self, tmp, replies, llm_cls=CapturingLLM):
        from afnan_ai.agent import AfnanAgent
        from afnan_ai.executor import Executor
        from afnan_ai.planner import Planner
        from afnan_ai.tools.registry import ToolRegistry
        from afnan_ai.verifier import Verifier

        echo = EchoTool()
        registry = ToolRegistry([echo])
        planner = Planner(llm_cls(replies), registry)
        return AfnanAgent(
            planner=planner,
            executor=Executor(registry),
            verifier=Verifier(),
            memory_dir=tmp,
            enable_browser_tools=False,
            enable_screen_tools=False,
        )

    def test_scheduled_background_run_completes_and_remembers(self):
        from test_architecture_upgrade import COMPLETE, echo_step
        from test_browser_advanced import plan_json

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        agent = self._agent(
            tmp.name,
            [plan_json("g", [echo_step("s1", "a")]), COMPLETE],
        )
        scheduler = agent.get_scheduler()
        past = datetime.now(timezone.utc) - timedelta(seconds=5)
        scheduler.schedule_task("answer the question", run_at=past)
        runner = agent.get_background_runner()
        done = runner.process_available()
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0].status, "completed")
        summaries = agent.get_memory_store().list(
            kind="task_summary"
        )
        self.assertEqual(len(summaries), 1)
        status = runner.status()
        self.assertEqual(status["tasks"].get("completed"), 1)

    def test_task_timeout_is_enforced_and_recorded(self):
        from test_architecture_upgrade import echo_step
        from test_browser_advanced import plan_json

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        agent = self._agent(
            tmp.name,
            [plan_json("g", [echo_step("s1", "a")]),
             plan_json("g", [echo_step("s1", "a")])],
            llm_cls=SlowLLM,
        )
        task = agent.get_task_manager().enqueue(
            "slow task", max_retries=0, timeout_s=0.05,
        )
        runner = agent.get_background_runner()
        runner.process_available()
        final = agent.get_task_manager().get(task.task_id)
        self.assertEqual(final.status, "failed")
        self.assertTrue(final.error)


if __name__ == "__main__":
    unittest.main()

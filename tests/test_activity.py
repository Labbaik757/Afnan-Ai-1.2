"""Tests for the Activity Center + Agent Control System.

Covers: event model, store/persistence, bus/dedup/replay,
status tracking, progress (no fake percentages), filtering,
control commands (pause/resume/stop/cancel/retry), approval
lifecycle, error presentation, evidence, voice intents,
notifications, query services, and real integration with
AgentLoop events / TaskManager / AuditLogger.
"""

import json
import os
import tempfile
import unittest

from afnan_ai.activity import events as E
from afnan_ai.activity.approval import ApprovalCenter
from afnan_ai.activity.center import ActivityCenter, EventStore
from afnan_ai.activity.control import (
    ControlCommand,
    ControlError,
    TaskControlService,
)
from afnan_ai.activity.errors import infer_code, present_error
from afnan_ai.activity.evidence import Evidence, from_verification
from afnan_ai.activity.integration import (
    attach_agent_loop,
    attach_audit_logger,
    attach_task_manager,
    build_integration,
)
from afnan_ai.activity.notifications import NotificationService
from afnan_ai.activity.query import (
    ActivityQueryService,
    TaskQueryService,
)
from afnan_ai.activity.status import AgentStatusTracker, TERMINAL
from afnan_ai.activity.timeline import (
    ActivityFilter,
    ActivityTimeline,
    ProgressTracker,
    summarize,
)
from afnan_ai.activity.voice import answer_query, parse_voice_command


def _center(tmpdir=None):
    store = EventStore(
        os.path.join(tempfile.mkdtemp(), "activity")
        if tmpdir is None
        else os.path.join(tmpdir, "activity")
    )
    return ActivityCenter(store=store)


class EventModelTests(unittest.TestCase):
    def test_valid_event(self):
        e = E.ActivityEvent(
            type=E.ACTION_STARTED, summary="doing a thing",
            task_id="t1",
        )
        self.assertEqual(e.seq, 0)
        self.assertTrue(e.event_id)
        self.assertTrue(e.at)

    def test_unknown_type_rejected(self):
        with self.assertRaises(ValueError):
            E.ActivityEvent(
                type="mind_reading", summary="x"
            )

    def test_secrets_redacted_in_details(self):
        e = E.ActivityEvent(
            type=E.ACTION_STARTED,
            summary="x",
            details={
                "api_key": "sk-secret-123",
                "password": "hunter2",
                "safe": "hello",
            },
        )
        self.assertNotIn("sk-secret-123", str(e.details))
        self.assertNotIn("hunter2", str(e.details))
        self.assertEqual(e.details["safe"], "hello")

    def test_roundtrip(self):
        e = E.ActivityEvent(
            type=E.TASK_COMPLETED, summary="done",
            task_id="t9", audit_ref="abc",
        )
        e2 = E.ActivityEvent.from_dict(e.to_dict())
        self.assertEqual(e2.type, e.type)
        self.assertEqual(e2.task_id, "t9")
        self.assertEqual(e2.audit_ref, "abc")

    def test_from_loop_event_mapping(self):
        from afnan_ai.agent_loop import LoopEvent

        le = LoopEvent(
            type="action_started",
            message="clicking",
            step_id="s3",
            details={"tool": "browser_click", "task_id": "t1"},
        )
        e = E.from_loop_event(le)
        self.assertEqual(e.type, E.ACTION_STARTED)
        self.assertEqual(e.task_id, "t1")
        self.assertEqual(e.source, "loop")
        self.assertEqual(e.details["step_id"], "s3")

    def test_from_audit_record_filters_noise(self):
        # Routine allows are not user-visible.
        self.assertIsNone(
            E.from_audit_record({"event": "allow", "action": "read"})
        )
        e = E.from_audit_record(
            {
                "event": "approval_required",
                "action": "delete file",
                "task_id": "t2",
                "risk_level": "SENSITIVE",
                "hash": "h123",
                "at": "2026-01-01T00:00:00",
            }
        )
        self.assertIsNotNone(e)
        self.assertEqual(e.type, E.APPROVAL_REQUIRED)
        self.assertEqual(e.audit_ref, "h123")


class EventStoreTests(unittest.TestCase):
    def test_seq_and_dedup(self):
        store = EventStore(
            os.path.join(tempfile.mkdtemp(), "a")
        )
        e = E.ActivityEvent(
            type=E.TASK_STARTED, summary="s", task_id="t1"
        )
        stored = store.append(e)
        self.assertEqual(stored.seq, 1)
        # Duplicate id → None, no seq consumed.
        dup = E.ActivityEvent(
            type=E.TASK_STARTED, summary="s", task_id="t1"
        )
        dup.event_id = e.event_id
        self.assertIsNone(store.append(dup))
        e2 = E.ActivityEvent(
            type=E.TASK_COMPLETED, summary="d", task_id="t1"
        )
        self.assertEqual(store.append(e2).seq, 2)

    def test_replay_after_seq(self):
        d = tempfile.mkdtemp()
        store = EventStore(os.path.join(d, "a"))
        for i in range(5):
            store.append(
                E.ActivityEvent(
                    type=E.WAITING, summary=f"s{i}",
                    task_id="t1" if i % 2 == 0 else "t2",
                )
            )
        replayed = store.replay(after_seq=2)
        self.assertEqual([e.seq for e in replayed], [3, 4, 5])
        only_t1 = store.replay(after_seq=0, task_id="t1")
        self.assertTrue(
            all(e.task_id == "t1" for e in only_t1)
        )

    def test_corrupt_line_skipped(self):
        d = tempfile.mkdtemp()
        adir = os.path.join(d, "a")
        os.makedirs(adir)
        with open(
            os.path.join(adir, "events.jsonl"), "w"
        ) as fh:
            fh.write("{not json\n")
            fh.write(
                json.dumps(
                    E.ActivityEvent(
                        type=E.WAITING, summary="ok"
                    ).to_dict()
                )
                + "\n"
            )
        store = EventStore(adir)  # recover must not crash
        self.assertEqual(store.last_seq, 0)
        # New appends continue cleanly.
        e = store.append(
            E.ActivityEvent(type=E.WAITING, summary="ok2")
        )
        self.assertEqual(e.seq, 1)

    def test_persistence_across_restart(self):
        d = tempfile.mkdtemp()
        adir = os.path.join(d, "a")
        s1 = EventStore(adir)
        s1.append(
            E.ActivityEvent(
                type=E.TASK_STARTED, summary="s", task_id="t1"
            )
        )
        s2 = EventStore(adir)  # new instance, same dir
        self.assertEqual(s2.last_seq, 1)
        e = s2.append(
            E.ActivityEvent(type=E.WAITING, summary="w")
        )
        self.assertEqual(e.seq, 2)


class CenterBusTests(unittest.TestCase):
    def test_publish_fans_out_and_updates_status(self):
        c = _center()
        received = []
        c.subscribe(received.append)
        c.emit(
            E.ACTION_STARTED, "doing", task_id="t1",
            details={"tool": "x"},
        )
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].seq, 1)
        self.assertEqual(c.status.get("t1"), "EXECUTING")

    def test_bad_subscriber_does_not_break_bus(self):
        c = _center()
        good = []
        c.subscribe(lambda e: 1 / 0)
        c.subscribe(good.append)
        c.emit(E.WAITING, "w", task_id="t1")
        self.assertEqual(len(good), 1)

    def test_sync_replays_missed(self):
        c = _center()
        for i in range(3):
            c.emit(E.WAITING, f"w{i}", task_id="t1")
        sync = c.sync(after_seq=1, task_id="t1")
        self.assertEqual(sync["last_seq"], 3)
        self.assertEqual(len(sync["missed_events"]), 2)

    def test_unsubscribe(self):
        c = _center()
        received = []
        sid = c.subscribe(received.append)
        c.unsubscribe(sid)
        c.emit(E.WAITING, "w")
        self.assertEqual(received, [])


class StatusTrackerTests(unittest.TestCase):
    def test_lifecycle(self):
        t = AgentStatusTracker()
        self.assertEqual(t.get("t1"), "IDLE")
        t.update("t1", E.ACTION_STARTED, "2026-01-01T00:00:00")
        self.assertEqual(t.get("t1"), "EXECUTING")
        t.update("t1", E.APPROVAL_REQUIRED, "2026-01-01T00:01:00")
        self.assertEqual(t.get("t1"), "WAITING_FOR_APPROVAL")
        t.update("t1", E.TASK_COMPLETED, "2026-01-01T00:02:00")
        self.assertEqual(t.get("t1"), "COMPLETED")

    def test_terminal_never_resurrects(self):
        t = AgentStatusTracker()
        t.update("t1", E.TASK_COMPLETED, "2026-01-01T00:00:00")
        # A late duplicate must not reopen the task.
        t.update("t1", E.ACTION_STARTED, "2026-01-01T00:03:00")
        self.assertEqual(t.get("t1"), "COMPLETED")
        self.assertIn("COMPLETED", TERMINAL)


class ProgressTrackerTests(unittest.TestCase):
    def _feed(self, tracker, events):
        for type_, summary, details in events:
            tracker.apply(
                E.ActivityEvent(
                    type=type_, summary=summary,
                    task_id="t1", details=details or {},
                )
            )

    def test_step_tracking(self):
        p = ProgressTracker()
        self._feed(
            p,
            [
                (E.TASK_STARTED, "start", {"planned_steps": 4}),
                (E.ACTION_STARTED, "a", {"step_id": "s1", "tool": "browser_search"}),
                (E.ACTION_COMPLETED, "b", {"step_id": "s1"}),
                (E.ACTION_FAILED, "c", {"step_id": "s2"}),
                (E.RECOVERY_STARTED, "r", {}),
            ],
        )
        prog = p.get("t1")
        self.assertEqual(prog.completed_steps, ["s1"])
        self.assertEqual(prog.failed_steps, ["s2"])
        self.assertEqual(prog.retry_count, 1)
        self.assertEqual(prog.recovery_count, 1)
        self.assertEqual(prog.tool_category, "browser_search")
        # Honest fraction: 1/4 with a plan.
        self.assertAlmostEqual(prog.progress_fraction, 0.25)

    def test_no_fake_progress_without_plan(self):
        p = ProgressTracker()
        self._feed(
            p,
            [
                (E.TASK_STARTED, "start", {}),
                (E.ACTION_COMPLETED, "b", {"step_id": "s1"}),
            ],
        )
        prog = p.get("t1")
        self.assertIsNone(prog.progress_fraction)

    def test_approval_tracking(self):
        p = ProgressTracker()
        self._feed(
            p,
            [
                (E.APPROVAL_REQUIRED, "need ok", {}),
                (E.APPROVAL_GRANTED, "ok", {}),
            ],
        )
        prog = p.get("t1")
        self.assertEqual(prog.approvals_pending, 0)


class FilterTimelineTests(unittest.TestCase):
    def _events(self):
        return [
            E.ActivityEvent(
                type=E.ACTION_STARTED, summary="search web",
                task_id="t1", details={"tool": "browser_search"},
                at="2026-01-01T00:00:00",
            ),
            E.ActivityEvent(
                type=E.ACTION_FAILED, summary="click failed",
                task_id="t1",
                details={"tool": "browser_click", "error_code": "BROWSER_ELEMENT_NOT_FOUND"},
                at="2026-01-02T00:00:00",
            ),
            E.ActivityEvent(
                type=E.TASK_COMPLETED, summary="done",
                task_id="t2", at="2026-01-03T00:00:00",
            ),
        ]

    def test_filter_combinations(self):
        events = self._events()
        f = ActivityFilter(
            types=[E.ACTION_STARTED, E.ACTION_FAILED],
            task_id="t1",
        )
        self.assertEqual(
            sum(1 for e in events if f.matches(e)), 2
        )
        f2 = ActivityFilter(error_code="BROWSER_ELEMENT_NOT_FOUND")
        matched = [e for e in events if f2.matches(e)]
        self.assertEqual(len(matched), 1)
        f3 = ActivityFilter(
            since="2026-01-02T00:00:00",
            until="2026-01-02T23:59:59",
        )
        self.assertEqual(
            sum(1 for e in events if f3.matches(e)), 1
        )
        f4 = ActivityFilter(text="search")
        self.assertEqual(
            sum(1 for e in events if f4.matches(e)), 1
        )

    def test_timeline_hides_nothing_unsafe(self):
        tl = ActivityTimeline(self._events())
        items = tl.items()
        self.assertEqual(len(items), 3)
        # Safe detail subset only.
        for item in items:
            self.assertNotIn("chain_of_thought", str(item))

    def test_summarize(self):
        s = summarize("t1", self._events())
        d = s.to_dict()
        self.assertEqual(d["task_id"], "t1")
        self.assertTrue(
            any("Failed" in f for f in d["failures_recoveries"])
        )


class FakeTaskManager:
    """Minimal TaskManager stand-in for control tests."""

    def __init__(self):
        self.tasks = {}

    def get(self, task_id):
        return self.tasks.get(task_id)

    def pause(self, task_id):
        self.tasks[task_id]["status"] = "paused"

    def resume(self, task_id):
        self.tasks[task_id]["status"] = "pending"

    def cancel(self, task_id):
        self.tasks[task_id]["status"] = "cancelled"


class ControlServiceTests(unittest.TestCase):
    def _svc(self, **kw):
        center = _center()
        tm = FakeTaskManager()
        tm.tasks["t1"] = {"status": "running", "goal_text": "g"}
        svc = TaskControlService(
            center=center, task_manager=tm, **kw
        )
        return svc, tm, center

    def test_pause_resume_stop_cancel(self):
        svc, tm, center = self._svc()
        svc.execute(ControlCommand(command="pause", task_id="t1"))
        self.assertEqual(tm.tasks["t1"]["status"], "paused")
        svc.execute(ControlCommand(command="resume", task_id="t1"))
        self.assertEqual(tm.tasks["t1"]["status"], "pending")
        svc.execute(ControlCommand(command="stop", task_id="t1"))
        # stop → safe boundary → paused (resumable), recorded as stopped
        self.assertEqual(tm.tasks["t1"]["status"], "paused")
        svc.execute(ControlCommand(command="cancel", task_id="t1"))
        self.assertEqual(tm.tasks["t1"]["status"], "cancelled")

    def test_unknown_task_rejected(self):
        svc, _, _ = self._svc()
        with self.assertRaises(ControlError):
            svc.execute(
                ControlCommand(command="pause", task_id="nope")
            )

    def test_invalid_command_rejected(self):
        with self.assertRaises(ValueError):
            ControlCommand(command="self_destruct", task_id="t1")

    def test_authorizer_can_deny(self):
        def deny(cmd):
            raise ControlError("forbidden", "no")

        svc, _, _ = self._svc(authorizer=deny)
        with self.assertRaises(ControlError):
            svc.execute(ControlCommand(command="pause", task_id="t1"))

    def test_retry_failed_task(self):
        svc, tm, _ = self._svc()
        tm.tasks["t1"]["status"] = "failed"
        svc.execute(ControlCommand(command="retry", task_id="t1"))
        self.assertEqual(tm.tasks["t1"]["status"], "pending")
        # Cannot retry a running task.
        tm.tasks["t1"]["status"] = "running"
        with self.assertRaises(ControlError):
            svc.execute(ControlCommand(command="retry", task_id="t1"))

    def test_loop_control_pause_signal(self):
        class FakeLoopControl:
            def __init__(self):
                self.paused = False

            def request_pause(self):
                self.paused = True

        svc, tm, _ = self._svc()
        lc = FakeLoopControl()
        svc.register_loop("t1", lc)
        svc.execute(ControlCommand(command="pause", task_id="t1"))
        self.assertTrue(lc.paused)

    def test_emergency_stop(self):
        called = []
        svc, _, center = self._svc(
            emergency=lambda r: called.append(r)
        )
        svc.execute(
            ControlCommand(
                command="emergency_stop", reason="test"
            )
        )
        self.assertEqual(called, ["test"])


class ApprovalCenterTests(unittest.TestCase):
    def _ac(self):
        return ApprovalCenter(center=_center())

    def test_request_approve_cycle(self):
        ac = self._ac()
        a = ac.request(
            task_id="t1", action="delete file",
            reason="cleanup", risk_level="SENSITIVE",
        )
        self.assertEqual(len(ac.pending()), 1)
        result = ac.decide(
            a.approval_id, granted=True, actor="user"
        )
        self.assertEqual(result["status"], "granted")
        self.assertEqual(len(ac.pending()), 0)

    def test_deny(self):
        ac = self._ac()
        a = ac.request(
            task_id="t1", action="pay", reason="buy"
        )
        ac.decide(a.approval_id, granted=False)
        self.assertEqual(
            ac.get(a.approval_id).status, "denied"
        )

    def test_double_decide_rejected(self):
        ac = self._ac()
        a = ac.request(task_id="t1", action="x", reason="y")
        ac.decide(a.approval_id, granted=True)
        with self.assertRaises(ValueError):
            ac.decide(a.approval_id, granted=True)

    def test_expiry(self):
        ac = self._ac()
        a = ac.request(
            task_id="t1", action="x", reason="y", ttl_s=0.01
        )
        import time

        time.sleep(0.03)
        self.assertEqual(len(ac.pending()), 0)
        self.assertEqual(ac.get(a.approval_id).status, "expired")
        with self.assertRaises(ValueError):
            ac.decide(a.approval_id, granted=True)

    def test_secrets_redacted(self):
        ac = self._ac()
        a = ac.request(
            task_id="t1", action="login",
            reason="use password=hunter2 here",
        )
        self.assertNotIn("hunter2", a.reason)

    def test_scope_recorded(self):
        ac = self._ac()
        a = ac.request(task_id="t1", action="x", reason="y")
        result = ac.decide(
            a.approval_id, granted=True, scope="task"
        )
        self.assertEqual(result["scope"], "task")

    def test_from_security_request_reuse(self):
        from afnan_ai.security.models import ApprovalRequest

        ac = self._ac()
        req = ApprovalRequest(
            action="send email", reason="user asked",
            target="boss@example.com",
        )
        tracked = ac.from_security_request(
            req, task_id="t1", audit_ref="h1"
        )
        self.assertEqual(tracked.action, "send email")
        self.assertEqual(tracked.audit_ref, "h1")


class ErrorPresentationTests(unittest.TestCase):
    def test_known_code(self):
        e = present_error("BROWSER_ELEMENT_NOT_FOUND")
        self.assertIn("button nahi mila", e.user_message)
        self.assertTrue(e.recoverable)

    def test_unknown_code_safe(self):
        e = present_error(
            "SOMETHING_WEIRD",
            technical="stack trace token=abc123",
        )
        self.assertEqual(e.error_code, "SOMETHING_WEIRD")
        self.assertNotIn("abc123", e.technical_summary)

    def test_infer_code(self):
        self.assertEqual(
            infer_code({"code": "approval_denied"}),
            "APPROVAL_DENIED",
        )
        self.assertEqual(infer_code(ValueError("timeout")), "TOOL_TIMEOUT")
        self.assertEqual(infer_code("???"), "UNKNOWN")


class EvidenceTests(unittest.TestCase):
    def test_redaction(self):
        ev = Evidence(
            evidence_id="e1",
            type="tool_result",
            task_id="t1",
            summary="ok",
            metadata={"token": "sekret", "url": "https://x"},
        )
        self.assertNotIn("sekret", str(ev.metadata))
        self.assertEqual(ev.metadata["url"], "https://x")

    def test_from_verification(self):
        ev = from_verification(
            {"status": "verified", "url": "https://x",
             "password": "p"}, task_id="t1",
        )
        self.assertIsNotNone(ev)
        self.assertNotIn("p", str(ev.metadata.get("password", "")))
        self.assertIsNone(from_verification({"status": "ok"}))

    def test_bad_type_rejected(self):
        with self.assertRaises(ValueError):
            Evidence(
                evidence_id="e", type="telepathy", task_id="t"
            )


class VoiceTests(unittest.TestCase):
    def test_parse_pause(self):
        intent = parse_voice_command(
            "Task pause karo", current_task_id="t1"
        )
        self.assertEqual(intent.kind, "command")
        self.assertEqual(intent.command.command, "pause")
        self.assertEqual(intent.command.task_id, "t1")

    def test_parse_deny(self):
        intent = parse_voice_command("Is approval ko deny karo")
        self.assertEqual(intent.command.command, "deny")

    def test_parse_status_query(self):
        intent = parse_voice_command(
            "Current task ka status batao"
        )
        self.assertEqual(intent.kind, "status_query")

    def test_parse_unknown(self):
        intent = parse_voice_command("hello world xyz")
        self.assertEqual(intent.kind, "unknown")

    def test_answer_status(self):
        class TQ:
            def get(self, tid):
                return {
                    "live_status": "EXECUTING",
                    "progress": {"current_step": "Searching"},
                }

        out = answer_query(
            parse_voice_command("status batao"),
            task_query=TQ(), activity_query=None, task_id="t1",
        )
        self.assertIn("EXECUTING", out)


class NotificationTests(unittest.TestCase):
    def test_channel_receives(self):
        svc = NotificationService()
        got = []
        svc.add_channel("test", got.append)
        center = _center()
        center.notifier = svc
        center.emit(E.TASK_FAILED, "boom", task_id="t1")
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["kind"], "task_failed")
        self.assertEqual(got[0]["severity"], "critical")

    def test_bad_channel_isolated(self):
        svc = NotificationService()
        svc.add_channel("bad", lambda n: 1 / 0)
        got = []
        svc.add_channel("good", got.append)
        center = _center()
        center.notifier = svc
        center.emit(E.TASK_COMPLETED, "done", task_id="t1")
        self.assertEqual(len(got), 1)

    def test_non_notifying_event_ignored(self):
        svc = NotificationService()
        got = []
        svc.add_channel("t", got.append)
        center = _center()
        center.notifier = svc
        center.emit(E.OBSERVING, "looking")
        self.assertEqual(got, [])


class QueryServiceTests(unittest.TestCase):
    def test_search_and_timeline(self):
        center = _center()
        center.emit(
            E.ACTION_STARTED, "search", task_id="t1",
            details={"tool": "browser_search"},
        )
        center.emit(E.TASK_COMPLETED, "done", task_id="t1")
        qs = ActivityQueryService(center=center)
        res = qs.search(ActivityFilter(task_id="t1", limit=10))
        self.assertEqual(res["total"], 2)
        self.assertEqual(res["api_version"], "v1")
        tl = qs.timeline("t1")
        self.assertEqual(tl["count"], 2)
        sm = qs.summary("t1")
        self.assertEqual(sm["task_id"], "t1")

    def test_pagination(self):
        center = _center()
        for i in range(5):
            center.emit(E.WAITING, f"w{i}")
        qs = ActivityQueryService(center=center)
        page = qs.search(ActivityFilter(limit=2, offset=2))
        self.assertEqual(len(page["items"]), 2)
        self.assertEqual(page["total"], 5)


class RealIntegrationTests(unittest.TestCase):
    """Real TaskManager + real LoopEvents + real AuditLogger."""

    def _real(self):
        d = tempfile.mkdtemp()
        from afnan_ai.task_manager import TaskManager

        tm = TaskManager(os.path.join(d, "tasks.json"))
        integ = build_integration(
            store_path=os.path.join(d, "activity"),
            task_manager=tm,
        )
        return integ, tm

    def test_task_manager_transitions_emit(self):
        integ, tm = self._real()
        t = tm.enqueue("do research")
        tm._transition(t, "running")
        tm._transition(t, "paused")
        events = integ.center.store.replay(task_id=t.task_id)
        types = [e.type for e in events]
        self.assertIn(E.TASK_STARTED, types)
        self.assertIn(E.TASK_PAUSED, types)

    def test_loop_sink_end_to_end(self):
        from afnan_ai.agent_loop import LoopEvent

        integ, tm = self._real()
        t = tm.enqueue("browse")
        sink = attach_agent_loop(
            integ.center, task_id=t.task_id, task_manager=tm
        )
        sink(
            LoopEvent(
                type="action_started", message="opening page",
                details={"tool": "browser_navigate"},
            )
        )
        sink(
            LoopEvent(
                type="verification_completed",
                message="verified",
            )
        )
        self.assertEqual(
            integ.center.status.get(t.task_id), "EXECUTING"
        )
        prog = integ.center.progress.get(t.task_id)
        self.assertEqual(prog.tool_category, "browser_navigate")

    def test_audit_approval_surfaces(self):
        from afnan_ai.security.audit import AuditLogger

        integ, _ = self._real()
        audit = AuditLogger(
            path=os.path.join(tempfile.mkdtemp(), "audit.jsonl")
        )
        attach_audit_logger(integ.center, audit)
        audit.log(
            "approval_required", actor="agent",
            action="delete", task_id="t1",
            risk_level="SENSITIVE",
        )
        audit.log("allow", actor="agent", action="read")
        approvals = integ.approvals.pending()
        self.assertEqual(len(approvals), 1)
        self.assertEqual(approvals[0].action, "delete")

    def test_control_pause_resume_real_manager(self):
        from afnan_ai.activity.control import ControlCommand

        integ, tm = self._real()
        t = tm.enqueue("work")
        tm._transition(t, "running")
        integ.control.execute(
            ControlCommand(command="pause", task_id=t.task_id)
        )
        self.assertEqual(tm.get(t.task_id).status, "paused")
        integ.control.execute(
            ControlCommand(command="resume", task_id=t.task_id)
        )
        self.assertEqual(tm.get(t.task_id).status, "pending")

    def test_approval_decision_resumes_waiting_task(self):
        integ, tm = self._real()
        t = tm.enqueue("sensitive work")
        tm._transition(t, "running")
        tm._transition(t, "waiting_for_approval")
        a = integ.approvals.request(
            task_id=t.task_id, action="pay", reason="buying"
        )
        integ.approvals.decide(a.approval_id, granted=True)
        self.assertEqual(tm.get(t.task_id).status, "pending")

    def test_task_query_service(self):
        integ, tm = self._real()
        t = tm.enqueue("q")
        qs = TaskQueryService(
            center=integ.center, task_manager=tm,
            progress=integ.center.progress,
        )
        listed = qs.list()
        self.assertEqual(listed["total"], 1)
        got = qs.get(t.task_id)
        self.assertEqual(got["live_status"], "IDLE")
        self.assertNotIn("checkpoint_ref", got)


class FailureInjectionTests(unittest.TestCase):
    def test_ui_disconnect_then_reconnect(self):
        center = _center()
        live = []
        sid = center.subscribe(live.append)
        center.emit(E.TASK_STARTED, "s", task_id="t1")
        # UI disconnects...
        center.unsubscribe(sid)
        # ...agent keeps working, events persist...
        center.emit(E.ACTION_STARTED, "a", task_id="t1")
        center.emit(E.TASK_COMPLETED, "d", task_id="t1")
        self.assertEqual(len(live), 1)
        # ...reconnect replays missed events, no corruption.
        sync = center.sync(after_seq=1, task_id="t1")
        self.assertEqual(len(sync["missed_events"]), 2)
        seqs = [e["seq"] for e in sync["missed_events"]]
        self.assertEqual(seqs, [2, 3])

    def test_duplicate_delivery_harmless(self):
        center = _center()
        e = E.ActivityEvent(
            type=E.ACTION_STARTED, summary="a", task_id="t1"
        )
        center.publish(e)
        center.publish(e)  # redelivered
        self.assertEqual(center.store.last_seq, 1)

    def test_stale_control_on_finished_task(self):
        # Real TaskManager rejects invalid transitions; the
        # control plane surfaces a clean error, never corrupts.
        from afnan_ai.task_manager import TaskManager

        d = tempfile.mkdtemp()
        tm = TaskManager(os.path.join(d, "tasks.json"))
        t = tm.enqueue("done work")
        tm._transition(t, "running")
        tm._transition(t, "completed")
        svc = TaskControlService(
            center=_center(), task_manager=tm
        )
        with self.assertRaises(ControlError):
            svc.execute(
                ControlCommand(command="pause", task_id=t.task_id)
            )
        self.assertEqual(
            tm.get(t.task_id).status, "completed"
        )

    def test_unauthorized_control_blocked(self):
        center = _center()
        tm = FakeTaskManager()
        tm.tasks["t1"] = {"status": "running", "goal_text": "g"}

        def deny(cmd):
            raise ControlError("denied", "not allowed")

        svc = TaskControlService(
            center=center, task_manager=tm, authorizer=deny
        )
        with self.assertRaises(ControlError):
            svc.execute(
                ControlCommand(command="stop", task_id="t1")
            )
        # Task untouched.
        self.assertEqual(tm.tasks["t1"]["status"], "running")

    def test_concurrent_pause_resume_no_crash(self):
        import threading

        center = _center()
        tm = FakeTaskManager()
        tm.tasks["t1"] = {"status": "running", "goal_text": "g"}
        svc = TaskControlService(
            center=center, task_manager=tm
        )

        def pause():
            try:
                svc.execute(
                    ControlCommand(command="pause", task_id="t1")
                )
            except Exception:
                pass

        def resume():
            try:
                svc.execute(
                    ControlCommand(command="resume", task_id="t1")
                )
            except Exception:
                pass

        threads = [
            threading.Thread(target=pause) for _ in range(5)
        ] + [
            threading.Thread(target=resume) for _ in range(5)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        # No crash; status is one of the valid ones.
        self.assertIn(
            tm.tasks["t1"]["status"], ("paused", "pending")
        )


class EndToEndTests(unittest.TestCase):
    """Full user journey through the real stack."""

    def test_natural_language_task_to_completion(self):
        d = tempfile.mkdtemp()
        from afnan_ai.agent_loop import LoopEvent
        from afnan_ai.task_manager import TaskManager

        tm = TaskManager(os.path.join(d, "tasks.json"))
        integ = build_integration(
            store_path=os.path.join(d, "activity"),
            task_manager=tm,
        )
        center = integ.center

        # 1. Natural-language task arrives.
        t = tm.enqueue("Find today's weather in Lahore")
        # 2. Loop starts; live activity flows.
        sink = attach_agent_loop(
            center, task_id=t.task_id, task_manager=tm
        )
        tm._transition(t, "running")
        sink(
            LoopEvent(
                type="observation_received",
                message="Browser on weather page",
                details={"url": "https://weather.example"},
            )
        )
        sink(
            LoopEvent(
                type="action_started", message="Extracting forecast",
                step_id="s1",
                details={"tool": "browser_extract"},
            )
        )
        # 3. Sensitive action → approval.
        sink(
            LoopEvent(
                type="approval_required",
                message="Share location for precise forecast?",
                details={
                    "action": "share_location",
                    "target": "weather.example",
                    "risk_level": "SENSITIVE",
                    "reason": "Precise forecast needs location",
                },
            )
        )
        self.assertEqual(len(integ.approvals.pending()), 1)
        # 4. User approves via control plane.
        appr = integ.approvals.pending()[0]
        integ.control.execute(
            ControlCommand(
                command="approve", approval_id=appr.approval_id
            )
        )
        # 5. Action + verification.
        sink(
            LoopEvent(
                type="action_completed", message="Forecast extracted",
                step_id="s1",
                details={"tool": "browser_extract"},
            )
        )
        sink(
            LoopEvent(
                type="verification_completed",
                message="Forecast verified against page",
            )
        )
        # 6. Artifact created.
        center.emit(
            E.ARTIFACT_CREATED, "Artifact created: forecast.md",
            task_id=t.task_id, source="artifact",
            details={
                "artifact_name": "forecast.md",
                "artifact_type": "markdown",
            },
        )
        # 7. Completion.
        sink(
            LoopEvent(type="task_completed", message="Done")
        )
        tm._transition(t, "completed")

        # 8. Historical activity available + accurate.
        tl = integ.activity_query.timeline(t.task_id)
        types = [i["type"] for i in tl["items"]]
        for expected in (
            E.TASK_STARTED, E.OBSERVING, E.ACTION_STARTED,
            E.APPROVAL_REQUIRED, E.APPROVAL_GRANTED,
            E.ACTION_COMPLETED, E.ARTIFACT_CREATED,
            E.TASK_COMPLETED,
        ):
            self.assertIn(expected, types)
        summary = integ.activity_query.summary(t.task_id)
        self.assertIn("forecast.md", summary["artifacts"])
        self.assertEqual(
            center.status.get(t.task_id), "COMPLETED"
        )
        # 9. Audit correlation present.
        appr_events = [
            i for i in tl["items"]
            if i["type"] == E.APPROVAL_GRANTED
        ]
        self.assertTrue(appr_events)


if __name__ == "__main__":
    unittest.main()

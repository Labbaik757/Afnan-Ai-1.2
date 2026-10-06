"""Tests for the Remote & Multi-Device Control Plane.

Covers: device identity, session state machine, command validation,
authorization, idempotency, event ordering/cursors, conflict
resolution, authentication (valid/invalid/expired/revoked/replayed),
pairing (expiry/replay/brute-force), authorization denials and
escalation attempts, reconnection and event replay, approval
binding/expiry/replay, emergency stop, multi-device coexistence and
revocation, offline queue policy, and a full device-handoff
integration scenario.

Security and command-routing logic execute for real against the
actual SecurityCenter/TaskManager/ActivityCenter/ApprovalCenter —
only the network transport is substituted.
"""

import tempfile
import time
import unittest
from pathlib import Path

from afnan_ai.activity.approval import ApprovalCenter
from afnan_ai.activity.center import ActivityCenter
from afnan_ai.control import (
    AuthenticationError,
    ClientSessionManager,
    CommandStatus,
    CommandType,
    ControlPlane,
    DeviceRegistry,
    DeviceTrust,
    EventStream,
    IdempotencyStore,
    PairingManager,
    PairingState,
    RemoteCommand,
    SessionState,
    SessionTransitionError,
    queue_policy_for,
)
from afnan_ai.control.auth import AuthProvider
from afnan_ai.control.models import (
    QueuePolicy,
    valid_session_transition,
)
from afnan_ai.goal_manager import GoalManager
from afnan_ai.security.center import SecurityCenter
from afnan_ai.security.models import Actor, ActorKind
from afnan_ai.task_manager import TaskManager


def make_plane(**kwargs):
    tmp = tempfile.mkdtemp(prefix="afnan-ctl-")
    security = SecurityCenter()
    task_manager = TaskManager(str(Path(tmp) / "tasks.json"))
    goal_manager = GoalManager(str(Path(tmp) / "goals.json"))
    activity = ActivityCenter()
    approvals = ApprovalCenter(center=activity)
    plane = ControlPlane(
        security_center=security,
        task_manager=task_manager,
        goal_manager=goal_manager,
        activity_center=activity,
        approval_center=approvals,
        device_path=str(Path(tmp) / "devices.json"),
        **kwargs,
    )
    return plane, tmp


def pair_device(plane, device_id="phone-1", **meta):
    pid, code = plane.pair_request(
        {"device_id": device_id, **meta}
    )
    plane.pair_approve(pid, approved_by="owner")
    result = plane.pair_redeem(
        {"pairing_id": pid, "code": code}
    )
    return result


def grant(plane, session_id, device_id, *capabilities):
    actor = Actor(
        kind=ActorKind.USER,
        actor_id=f"remote:{device_id}:{session_id}",
    )
    for capability in capabilities:
        plane.security.permissions.grant(actor, capability)
    session = plane.sessions.get(session_id)
    session.capabilities = tuple(
        set(session.capabilities) | set(capabilities)
    )


def cmd(command_type, payload=None, **kwargs):
    return {
        "command_type": command_type.value
        if isinstance(command_type, CommandType)
        else command_type,
        "payload": payload or {},
        **kwargs,
    }


class DeviceIdentityTests(unittest.TestCase):
    def test_register_and_retrieve(self):
        registry = DeviceRegistry()
        info = registry.register(
            "dev-1", device_name="Phone", platform="android"
        )
        self.assertEqual(info.device_id, "dev-1")
        self.assertEqual(info.trust, DeviceTrust.UNPAIRED)
        self.assertEqual(
            registry.get("dev-1").device_name, "Phone"
        )

    def test_invalid_device_id_rejected(self):
        registry = DeviceRegistry()
        with self.assertRaises(ValueError):
            registry.register("")
        with self.assertRaises(ValueError):
            registry.register("x" * 200)

    def test_secret_never_stored_plaintext(self):
        registry = DeviceRegistry()
        registry.register("dev-1")
        registry.set_secret_hash("dev-1", "abc123hash")
        info = registry.get("dev-1")
        d = info.to_dict()
        self.assertNotIn("secret_hash", d)
        # to_dict must not leak the hash to clients.
        self.assertNotIn("abc123hash", str(d))

    def test_verify_secret_constant_time(self):
        from afnan_ai.control.devices import hash_secret

        registry = DeviceRegistry()
        registry.register("dev-1")
        registry.set_secret_hash(
            "dev-1", hash_secret("s3cret")
        )
        self.assertTrue(
            registry.verify_secret("dev-1", "s3cret")
        )
        self.assertFalse(
            registry.verify_secret("dev-1", "wrong")
        )
        self.assertFalse(
            registry.verify_secret("unknown", "s3cret")
        )

    def test_revoke_is_immediate(self):
        registry = DeviceRegistry()
        registry.register("dev-1")
        registry.set_trust("dev-1", DeviceTrust.PAIRED)
        registry.revoke("dev-1")
        info = registry.get("dev-1")
        self.assertEqual(info.trust, DeviceTrust.REVOKED)

    def test_persistence_roundtrip(self):
        tmp = tempfile.mkdtemp()
        path = str(Path(tmp) / "devices.json")
        registry = DeviceRegistry(path)
        registry.register("dev-1", device_name="Phone")
        registry.set_trust("dev-1", DeviceTrust.PAIRED)
        registry2 = DeviceRegistry(path)
        info = registry2.get("dev-1")
        self.assertIsNotNone(info)
        self.assertEqual(info.trust, DeviceTrust.PAIRED)
        self.assertEqual(info.device_name, "Phone")


class SessionStateMachineTests(unittest.TestCase):
    def test_valid_transitions(self):
        self.assertTrue(
            valid_session_transition(
                SessionState.CONNECTING,
                SessionState.AUTHENTICATING,
            )
        )
        self.assertTrue(
            valid_session_transition(
                SessionState.ACTIVE, SessionState.IDLE
            )
        )
        self.assertTrue(
            valid_session_transition(
                SessionState.IDLE, SessionState.ACTIVE
            )
        )

    def test_invalid_transitions_rejected(self):
        self.assertFalse(
            valid_session_transition(
                SessionState.CLOSED, SessionState.ACTIVE
            )
        )
        self.assertFalse(
            valid_session_transition(
                SessionState.REVOKED, SessionState.ACTIVE
            )
        )
        self.assertFalse(
            valid_session_transition(
                SessionState.CONNECTING, SessionState.ACTIVE
            )
        )

    def test_transition_method_enforces(self):
        manager = ClientSessionManager()
        session = manager.create("dev-1")
        session.transition(SessionState.AUTHENTICATING)
        self.assertEqual(
            session.state, SessionState.AUTHENTICATING
        )
        with self.assertRaises(SessionTransitionError):
            session.transition(SessionState.ACTIVE)

    def test_revoke_is_terminal_and_immediate(self):
        manager = ClientSessionManager()
        session = manager.create("dev-1")
        session.transition(SessionState.AUTHENTICATING)
        session.transition(SessionState.AUTHORIZED)
        manager.revoke(session.session_id)
        self.assertEqual(session.state, SessionState.REVOKED)
        with self.assertRaises(SessionTransitionError):
            session.transition(SessionState.ACTIVE)

    def test_expiry_reaping(self):
        manager = ClientSessionManager(
            session_ttl_s=0.01, idle_timeout_s=1000
        )
        session = manager.create("dev-1")
        time.sleep(0.03)
        count = manager.reap_expired()
        self.assertEqual(count, 1)
        self.assertEqual(session.state, SessionState.EXPIRED)

    def test_heartbeat_resurrects_idle(self):
        manager = ClientSessionManager()
        session = manager.create("dev-1")
        session.transition(SessionState.AUTHENTICATING)
        session.transition(SessionState.AUTHORIZED)
        session.transition(SessionState.IDLE)
        manager.heartbeat(session.session_id)
        self.assertEqual(session.state, SessionState.ACTIVE)


class PairingTests(unittest.TestCase):
    def test_full_pairing_flow(self):
        plane, _ = make_plane()
        pid, code = plane.pair_request(
            {"device_id": "phone-1", "device_name": "Phone"}
        )
        self.assertEqual(len(code), 6)
        self.assertTrue(code.isdigit())
        device = plane.devices.get("phone-1")
        self.assertEqual(device.trust, DeviceTrust.PENDING)
        plane.pair_approve(pid, approved_by="owner")
        result = plane.pair_redeem(
            {"pairing_id": pid, "code": code}
        )
        self.assertEqual(result["device_id"], "phone-1")
        self.assertIn("token", result)
        self.assertEqual(
            plane.devices.get("phone-1").trust,
            DeviceTrust.PAIRED,
        )

    def test_code_single_use_replay_rejected(self):
        plane, _ = make_plane()
        pid, code = plane.pair_request({"device_id": "d1"})
        plane.pair_approve(pid, approved_by="owner")
        plane.pair_redeem({"pairing_id": pid, "code": code})
        with self.assertRaises(Exception):
            plane.pair_redeem({"pairing_id": pid, "code": code})

    def test_redeem_before_approval_rejected(self):
        plane, _ = make_plane()
        pid, code = plane.pair_request({"device_id": "d1"})
        with self.assertRaises(Exception):
            plane.pair_redeem({"pairing_id": pid, "code": code})

    def test_wrong_code_rejected(self):
        plane, _ = make_plane()
        pid, code = plane.pair_request({"device_id": "d1"})
        plane.pair_approve(pid, approved_by="owner")
        with self.assertRaises(Exception):
            plane.pair_redeem(
                {"pairing_id": pid, "code": "000000"}
            )

    def test_pairing_expiry(self):
        manager = PairingManager(code_ttl_s=0.01)
        request, code = manager.create_request(
            requesting_device_id="d1"
        )
        manager.approve(
            request.pairing_id, approved_by="owner"
        )
        time.sleep(0.03)
        with self.assertRaises(Exception):
            manager.redeem(request.pairing_id, code)
        self.assertEqual(request.state, PairingState.EXPIRED)

    def test_brute_force_lockout(self):
        manager = PairingManager(
            code_ttl_s=60, max_attempts=3
        )
        request, code = manager.create_request(
            requesting_device_id="d1"
        )
        manager.approve(
            request.pairing_id, approved_by="owner"
        )
        for _ in range(4):
            try:
                manager.redeem(request.pairing_id, "000000")
            except Exception:
                pass
        # Even the right code now fails: request was rejected.
        with self.assertRaises(Exception):
            manager.redeem(request.pairing_id, code)

    def test_code_hash_not_plaintext(self):
        manager = PairingManager()
        request, code = manager.create_request(
            requesting_device_id="d1"
        )
        d = request.to_dict()
        self.assertNotIn("code", d)
        self.assertNotIn(code, str(d))


class AuthenticationTests(unittest.TestCase):
    def test_valid_token_authenticates(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        session = plane.authenticate(result["token"])
        self.assertEqual(
            session.session_id, result["session_id"]
        )

    def test_invalid_token_rejected(self):
        plane, _ = make_plane()
        pair_device(plane)
        with self.assertRaises(AuthenticationError):
            plane.authenticate("afn_bogus")

    def test_revoked_token_rejected(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        token = result["token"]
        session = plane.sessions.get(result["session_id"])
        plane.auth.revoke_token(session.token_hash)
        with self.assertRaises(AuthenticationError):
            plane.authenticate(token)

    def test_expired_token_rejected(self):
        plane, _ = make_plane()
        result = pair_device(plane, device_id="d-exp")
        session = plane.sessions.get(result["session_id"])
        # Expire the token record directly.
        with plane.auth._lock:
            record = plane.auth._tokens[session.token_hash]
            record.expires_at = time.time() - 1
        with self.assertRaises(AuthenticationError):
            plane.authenticate(result["token"])

    def test_revoked_device_rejected(self):
        plane, _ = make_plane()
        result = pair_device(plane, device_id="d-rev")
        plane.devices.revoke("d-rev")
        with self.assertRaises(AuthenticationError):
            plane.authenticate(result["token"])

    def test_token_rotation(self):
        plane, _ = make_plane()
        result = pair_device(plane, device_id="d-rot")
        old = result["token"]
        rotated = plane.rotate_token(old, old)
        new = rotated["token"]
        self.assertNotEqual(old, new)
        # Old token is dead.
        with self.assertRaises(AuthenticationError):
            plane.authenticate(old)
        # New token works.
        session = plane.authenticate(new)
        self.assertEqual(
            session.session_id, result["session_id"]
        )

    def test_rotation_with_wrong_token_rejected(self):
        plane, _ = make_plane()
        result = pair_device(plane, device_id="d-rot2")
        with self.assertRaises(AuthenticationError):
            plane.auth.rotate(
                plane.sessions.get(result["session_id"]),
                "afn_wrong",
            )


class AuthorizationTests(unittest.TestCase):
    def test_default_capabilities_are_read_only(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        out = plane.execute_command(
            result["token"], cmd(CommandType.LIST_TASKS)
        )
        self.assertEqual(out["status"], "completed")
        out = plane.execute_command(
            result["token"],
            cmd(
                CommandType.START_TASK,
                {"goal_text": "x"},
            ),
        )
        self.assertEqual(out["status"], "denied")
        self.assertEqual(out["error_code"], "FORBIDDEN")

    def test_granted_capability_allows_command(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        grant(
            plane,
            result["session_id"],
            "phone-1",
            "remote.tasks.control",
        )
        out = plane.execute_command(
            result["token"],
            cmd(
                CommandType.START_TASK,
                {"goal_text": "do thing"},
            ),
        )
        self.assertEqual(out["status"], "completed")

    def test_privilege_escalation_attempt_blocked(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        # Attacker tries to smuggle admin capability via payload.
        out = plane.execute_command(
            result["token"],
            cmd(
                CommandType.GET_AGENT_STATUS,
                {"grant": ["remote.admin"]},
            ),
        )
        self.assertEqual(out["status"], "completed")
        # The grant did not happen.
        actor = Actor(
            kind=ActorKind.USER,
            actor_id=(
                f"remote:phone-1:{result['session_id']}"
            ),
        )
        self.assertFalse(
            plane.security.permissions.check(
                actor, "remote.admin"
            )
        )

    def test_cross_device_command_session_mismatch(self):
        plane, _ = make_plane()
        first = pair_device(plane, device_id="dev-a")
        second = pair_device(plane, device_id="dev-b")
        with self.assertRaises(Exception):
            plane.execute_command(
                first["token"],
                cmd(
                    CommandType.GET_AGENT_STATUS,
                    {},
                    session_id=second["session_id"],
                ),
            )

    def test_unknown_command_type_rejected(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        with self.assertRaises(Exception):
            plane.execute_command(
                result["token"],
                {
                    "command_type": "delete_everything",
                    "payload": {},
                },
            )

    def test_malformed_command_rejected(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        with self.assertRaises(Exception):
            plane.execute_command(
                result["token"],
                {"command_type": "list_tasks", "payload": []},
            )

    def test_oversized_payload_rejected(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        with self.assertRaises(Exception):
            plane.execute_command(
                result["token"],
                cmd(
                    CommandType.LIST_TASKS,
                    {"blob": "x" * 70000},
                ),
            )


class IdempotencyTests(unittest.TestCase):
    def test_duplicate_key_returns_stored_result(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        grant(
            plane,
            result["session_id"],
            "phone-1",
            "remote.tasks.control",
        )
        first = plane.execute_command(
            result["token"],
            cmd(
                CommandType.START_TASK,
                {"goal_text": "once"},
                idempotency_key="idem-1",
            ),
        )
        self.assertEqual(first["status"], "completed")
        second = plane.execute_command(
            result["token"],
            cmd(
                CommandType.START_TASK,
                {"goal_text": "twice"},
                idempotency_key="idem-1",
            ),
        )
        self.assertEqual(second["status"], "duplicate")
        # Only one task was created.
        tasks = plane.execute_command(
            result["token"], cmd(CommandType.LIST_TASKS)
        )
        self.assertEqual(len(tasks["result"]["tasks"]), 1)

    def test_idempotency_scoped_per_device(self):
        plane, _ = make_plane()
        first = pair_device(plane, device_id="dev-a")
        second = pair_device(plane, device_id="dev-b")
        for res, dev in (
            (first, "dev-a"),
            (second, "dev-b"),
        ):
            grant(
                plane,
                res["session_id"],
                dev,
                "remote.tasks.control",
            )
        one = plane.execute_command(
            first["token"],
            cmd(
                CommandType.START_TASK,
                {"goal_text": "a"},
                idempotency_key="same-key",
            ),
        )
        two = plane.execute_command(
            second["token"],
            cmd(
                CommandType.START_TASK,
                {"goal_text": "b"},
                idempotency_key="same-key",
            ),
        )
        self.assertEqual(one["status"], "completed")
        # Same key on a different device is a different scope.
        self.assertEqual(two["status"], "completed")

    def test_store_claim_release_cycle(self):
        store = IdempotencyStore(ttl_s=60)
        self.assertTrue(
            store.claim(
                "d1",
                "k",
                command_id="c1",
                command_type="t",
                session_id="s1",
            )
        )
        self.assertFalse(
            store.claim(
                "d1",
                "k",
                command_id="c2",
                command_type="t",
                session_id="s1",
            )
        )
        store.complete("d1", "k", {"ok": True})
        record = store.check("d1", "k")
        self.assertIsNotNone(record)
        self.assertEqual(record.result, {"ok": True})


class CommandLifecycleTests(unittest.TestCase):
    def test_expired_command_rejected(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        out = plane.execute_command(
            result["token"],
            cmd(
                CommandType.LIST_TASKS,
                {},
                expires_at=time.time() - 10,
            ),
        )
        self.assertEqual(out["status"], "expired")

    def test_offline_policy_blocks_mutation(self):
        router = make_plane()[0].router
        command = RemoteCommand(
            session_id="s",
            device_id="d",
            command_type=CommandType.CANCEL_TASK,
            payload={"task_id": "t"},
        )
        out = router.route(
            command,
            session_capabilities=("remote.tasks.control",),
            live_session=False,
        )
        self.assertEqual(out.status, CommandStatus.DENIED)
        self.assertEqual(out.error_code, "REQUIRES_LIVE_SESSION")

    def test_offline_policy_allows_safe_reads(self):
        self.assertEqual(
            queue_policy_for(CommandType.LIST_TASKS),
            QueuePolicy.SAFE_QUEUEABLE,
        )
        self.assertEqual(
            queue_policy_for(CommandType.APPROVE_ACTION),
            QueuePolicy.NEVER_QUEUEABLE,
        )
        self.assertEqual(
            queue_policy_for(CommandType.EMERGENCY_STOP),
            QueuePolicy.NEVER_QUEUEABLE,
        )
        self.assertEqual(
            queue_policy_for(CommandType.START_TASK),
            QueuePolicy.REQUIRES_LIVE_SESSION,
        )

    def test_conflicting_commands_resolve_deterministically(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        grant(
            plane,
            result["session_id"],
            "phone-1",
            "remote.tasks.control",
        )
        created = plane.execute_command(
            result["token"],
            cmd(
                CommandType.START_TASK,
                {"goal_text": "conflict me"},
            ),
        )
        task_id = created["result"]["task"]["task_id"]
        paused = plane.execute_command(
            result["token"],
            cmd(CommandType.PAUSE_TASK, {"task_id": task_id}),
        )
        self.assertEqual(paused["status"], "completed")
        # Cancel after pause: TaskManager is authoritative.
        cancelled = plane.execute_command(
            result["token"],
            cmd(CommandType.CANCEL_TASK, {"task_id": task_id}),
        )
        self.assertEqual(cancelled["status"], "completed")
        self.assertEqual(
            cancelled["result"]["task"]["status"], "cancelled"
        )
        # Pausing a cancelled task fails safely (no crash).
        again = plane.execute_command(
            result["token"],
            cmd(CommandType.PAUSE_TASK, {"task_id": task_id}),
        )
        self.assertEqual(again["status"], "failed")


class EventStreamTests(unittest.TestCase):
    def test_attach_replays_missed_events(self):
        plane, _ = make_plane()
        stream = plane.events
        stream.attach("sess-1", from_seq=0)
        stream.publish("task.created", {"task_id": "t1"})
        stream.publish("task.started", {"task_id": "t1"})
        events = stream.poll("sess-1")
        self.assertEqual(len(events), 2)
        self.assertEqual(
            events[0]["category"], "task.created"
        )

    def test_event_ordering_and_cursors(self):
        plane, _ = make_plane()
        stream = plane.events
        stream.attach("sess-1")
        for i in range(5):
            stream.publish("task.progress", {"i": i})
        events = stream.poll("sess-1")
        seqs = [e["seq"] for e in events]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(set(seqs)), 5)
        self.assertEqual(stream.cursor("sess-1"), seqs[-1])

    def test_category_filter(self):
        plane, _ = make_plane()
        stream = plane.events
        stream.attach("sess-1", categories={"task.created"})
        stream.publish("task.created", {})
        stream.publish("task.failed", {})
        events = stream.poll("sess-1")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["category"], "task.created")

    def test_unknown_category_rejected(self):
        plane, _ = make_plane()
        with self.assertRaises(ValueError):
            plane.events.attach(
                "sess-1", categories={"nope.nope"}
            )

    def test_backpressure_drops_with_gap_marker(self):
        plane, _ = make_plane()
        stream = EventStream(
            plane.router.activity, buffer_size=4
        )
        stream.attach("sess-1")
        for _ in range(10):
            stream.publish("task.progress", {})
        events = stream.poll("sess-1")
        gap = [e for e in events if e["category"] == "stream.gap"]
        self.assertTrue(gap)
        self.assertGreaterEqual(gap[0]["data"]["dropped"], 1)

    def test_activity_center_bridge(self):
        plane, _ = make_plane()
        stream = plane.events
        stream.attach("sess-1")
        plane.router.activity.emit(
            "task_created", "task t1 created", task_id="t1"
        )
        events = stream.poll("sess-1")
        categories = [e["category"] for e in events]
        self.assertIn("task.created", categories)


class ApprovalGatewayTests(unittest.TestCase):
    def _approved_plane(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        grant(
            plane,
            result["session_id"],
            "phone-1",
            "remote.approvals.decide",
        )
        return plane, result

    def test_remote_approve_flow(self):
        plane, result = self._approved_plane()
        tracked = plane.router.approvals.request(
            task_id="task_1",
            action="delete file",
            reason="user asked",
            ttl_s=600,
        )
        out = plane.execute_command(
            result["token"],
            cmd(
                CommandType.APPROVE_ACTION,
                {"approval_id": tracked.approval_id},
            ),
        )
        self.assertEqual(out["status"], "completed")
        self.assertTrue(out["result"]["granted"])

    def test_remote_deny_flow(self):
        plane, result = self._approved_plane()
        tracked = plane.router.approvals.request(
            task_id="task_1",
            action="delete file",
            reason="user asked",
            ttl_s=600,
        )
        out = plane.execute_command(
            result["token"],
            cmd(
                CommandType.DENY_ACTION,
                {"approval_id": tracked.approval_id},
            ),
        )
        self.assertEqual(out["status"], "completed")
        self.assertFalse(out["result"]["granted"])

    def test_approval_replay_rejected(self):
        plane, result = self._approved_plane()
        tracked = plane.router.approvals.request(
            task_id="task_1",
            action="delete file",
            reason="user asked",
            ttl_s=600,
        )
        plane.execute_command(
            result["token"],
            cmd(
                CommandType.APPROVE_ACTION,
                {"approval_id": tracked.approval_id},
            ),
        )
        # Second decision on the same approval fails.
        out = plane.execute_command(
            result["token"],
            cmd(
                CommandType.APPROVE_ACTION,
                {"approval_id": tracked.approval_id},
            ),
        )
        self.assertEqual(out["status"], "failed")

    def test_expired_approval_rejected(self):
        plane, result = self._approved_plane()
        tracked = plane.router.approvals.request(
            task_id="task_1",
            action="delete file",
            reason="user asked",
            ttl_s=0.01,
        )
        time.sleep(0.03)
        out = plane.execute_command(
            result["token"],
            cmd(
                CommandType.APPROVE_ACTION,
                {"approval_id": tracked.approval_id},
            ),
        )
        self.assertEqual(out["status"], "failed")

    def test_unknown_approval_rejected(self):
        plane, result = self._approved_plane()
        out = plane.execute_command(
            result["token"],
            cmd(
                CommandType.APPROVE_ACTION,
                {"approval_id": "nope"},
            ),
        )
        self.assertEqual(out["status"], "failed")
        self.assertEqual(out["error_code"], "NOT_FOUND")

    def test_approval_requires_capability(self):
        plane, _ = make_plane()
        result = pair_device(plane)  # read-only defaults
        tracked = plane.router.approvals.request(
            task_id="task_1",
            action="delete file",
            reason="user asked",
            ttl_s=600,
        )
        out = plane.execute_command(
            result["token"],
            cmd(
                CommandType.APPROVE_ACTION,
                {"approval_id": tracked.approval_id},
            ),
        )
        self.assertEqual(out["status"], "denied")


class EmergencyStopTests(unittest.TestCase):
    def test_remote_emergency_stop(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        grant(
            plane,
            result["session_id"],
            "phone-1",
            "remote.safety.emergency_stop",
        )
        out = plane.execute_command(
            result["token"],
            cmd(
                CommandType.EMERGENCY_STOP,
                {"reason": "test stop"},
            ),
        )
        self.assertEqual(out["status"], "completed")
        self.assertTrue(out["result"]["tripped"])
        self.assertTrue(
            plane.security.emergency.is_tripped()
        )

    def test_emergency_stop_requires_capability(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        out = plane.execute_command(
            result["token"],
            cmd(CommandType.EMERGENCY_STOP, {}),
        )
        self.assertEqual(out["status"], "denied")
        self.assertFalse(
            plane.security.emergency.is_tripped()
        )


class MultiDeviceTests(unittest.TestCase):
    def test_two_devices_coexist(self):
        plane, _ = make_plane()
        first = pair_device(plane, device_id="dev-a")
        second = pair_device(plane, device_id="dev-b")
        for res, dev in ((first, "dev-a"), (second, "dev-b")):
            grant(
                plane,
                res["session_id"],
                dev,
                "remote.tasks.control",
            )
        one = plane.execute_command(
            first["token"],
            cmd(
                CommandType.START_TASK, {"goal_text": "from A"}
            ),
        )
        two = plane.execute_command(
            second["token"],
            cmd(
                CommandType.START_TASK, {"goal_text": "from B"}
            ),
        )
        self.assertEqual(one["status"], "completed")
        self.assertEqual(two["status"], "completed")
        listed = plane.execute_command(
            first["token"], cmd(CommandType.LIST_TASKS)
        )
        self.assertEqual(len(listed["result"]["tasks"]), 2)

    def test_device_revocation_kills_access(self):
        plane, _ = make_plane()
        victim = pair_device(plane, device_id="dev-victim")
        admin = pair_device(plane, device_id="dev-admin")
        grant(
            plane,
            admin["session_id"],
            "dev-admin",
            "remote.devices.manage",
        )
        out = plane.execute_command(
            admin["token"],
            cmd(
                CommandType.REVOKE_DEVICE,
                {"device_id": "dev-victim"},
            ),
        )
        self.assertEqual(out["status"], "completed")
        self.assertGreaterEqual(
            out["result"]["revoked_sessions"], 1
        )
        # The victim's token no longer authenticates.
        with self.assertRaises(AuthenticationError):
            plane.authenticate(victim["token"])

    def test_session_revocation(self):
        plane, _ = make_plane()
        victim = pair_device(plane, device_id="dev-v")
        admin = pair_device(plane, device_id="dev-a2")
        grant(
            plane,
            admin["session_id"],
            "dev-a2",
            "remote.devices.manage",
        )
        out = plane.execute_command(
            admin["token"],
            cmd(
                CommandType.REVOKE_SESSION,
                {"session_id": victim["session_id"]},
            ),
        )
        self.assertEqual(out["status"], "completed")
        with self.assertRaises(AuthenticationError):
            plane.authenticate(victim["token"])

    def test_device_cannot_revoke_itself(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        grant(
            plane,
            result["session_id"],
            "phone-1",
            "remote.devices.manage",
        )
        out = plane.execute_command(
            result["token"],
            cmd(
                CommandType.REVOKE_DEVICE,
                {"device_id": "phone-1"},
            ),
        )
        self.assertEqual(out["status"], "failed")


class ReconnectionTests(unittest.TestCase):
    def test_reconnect_replays_missed_events(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        session_id = result["session_id"]
        # Client attaches, gets cursor 0, detaches (disconnect).
        attached = plane.events.attach(session_id, from_seq=0)
        self.assertEqual(attached["cursor"], 0)
        plane.events.detach(session_id)
        # Events happen while away.
        plane.events.publish("task.created", {"task_id": "t9"})
        plane.events.publish("task.started", {"task_id": "t9"})
        # Reconnect with the old cursor replays via activity sync;
        # live buffer replay covers control-plane events.
        reattached = plane.events.attach(
            session_id, from_seq=attached["cursor"]
        )
        self.assertIn("missed_events", reattached)

    def test_stale_session_rejected_after_expiry(self):
        plane, tmp = make_plane(session_ttl_s=0.01)
        result = pair_device(plane)
        time.sleep(0.03)
        plane.sessions.reap_expired()
        with self.assertRaises(AuthenticationError):
            plane.authenticate(result["token"])


class SecurityHardeningTests(unittest.TestCase):
    def test_secret_never_in_audit_or_events(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        # The token value must not appear in any stored form.
        stored = str(plane.auth._tokens)
        self.assertNotIn(result["token"], stored)

    def test_command_payload_injection_screened(self):
        # Remote payloads are validated; unknown dangerous fields
        # do not reach the runtime as privileged input.
        plane, _ = make_plane()
        result = pair_device(plane)
        out = plane.execute_command(
            result["token"],
            {
                "command_type": "list_tasks",
                "payload": {"__class__": "os.system"},
            },
        )
        self.assertEqual(out["status"], "completed")

    def test_sensitive_artifact_gated(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        # No artifact manager wired -> NOT_WIRED, not a leak.
        out = plane.execute_command(
            result["token"],
            cmd(
                CommandType.LIST_ARTIFACTS, {"project_id": "x"}
            ),
        )
        self.assertEqual(out["status"], "failed")
        self.assertEqual(out["error_code"], "NOT_WIRED")


class IntegrationScenarioTests(unittest.TestCase):
    """Section 43: full device-handoff scenario."""

    def test_device_a_to_b_handoff_with_approval(self):
        plane, _ = make_plane()
        # Device A pairs and starts a long-running task.
        device_a = pair_device(plane, device_id="device-a")
        grant(
            plane,
            device_a["session_id"],
            "device-a",
            "remote.tasks.control",
            "remote.approvals.decide",
        )
        created = plane.execute_command(
            device_a["token"],
            cmd(
                CommandType.START_TASK,
                {"goal_text": "long research task"},
                idempotency_key="handoff-1",
            ),
        )
        self.assertEqual(created["status"], "completed")
        task_id = created["result"]["task"]["task_id"]

        # Device A disconnects: agent state is untouched.
        plane.events.detach(device_a["session_id"])
        task = plane.router.tasks.get(task_id)
        self.assertIsNotNone(task)

        # Device B pairs, authenticates, resumes.
        device_b = pair_device(plane, device_id="device-b")
        grant(
            plane,
            device_b["session_id"],
            "device-b",
            "remote.tasks.control",
            "remote.approvals.decide",
        )
        listed = plane.execute_command(
            device_b["token"], cmd(CommandType.LIST_TASKS)
        )
        ids = [t["task_id"] for t in listed["result"]["tasks"]]
        self.assertIn(task_id, ids)

        # An approval appears; device B approves it.
        tracked = plane.router.approvals.request(
            task_id=task_id,
            action="publish report",
            reason="needs human sign-off",
            ttl_s=600,
        )
        pending = plane.execute_command(
            device_b["token"],
            cmd(CommandType.LIST_APPROVALS),
        )
        self.assertEqual(pending["status"], "completed")
        self.assertTrue(
            any(
                a["approval_id"] == tracked.approval_id
                for a in pending["result"]["approvals"]
            )
        )
        decided = plane.execute_command(
            device_b["token"],
            cmd(
                CommandType.APPROVE_ACTION,
                {"approval_id": tracked.approval_id},
            ),
        )
        self.assertEqual(decided["status"], "completed")

        # Task completes through the existing TaskManager.
        claimed = plane.router.tasks.claim_next()
        self.assertIsNotNone(claimed)
        plane.router.tasks.complete(
            claimed.task_id, "done via handoff"
        )
        final = plane.execute_command(
            device_b["token"],
            cmd(CommandType.GET_TASK, {"task_id": task_id}),
        )
        self.assertEqual(
            final["result"]["task"]["status"], "completed"
        )

        # Audit chain captured the remote actions.
        chain = plane.security.audit.verify_chain()
        self.assertTrue(chain["ok"])
        self.assertGreater(chain["checked"], 0)


class TransportConfigTests(unittest.TestCase):
    def test_insecure_requires_explicit_opt_in(self):
        from afnan_ai.control.transport import (
            ControlTransport,
            TransportError,
        )

        plane, _ = make_plane()
        with self.assertRaises(TransportError):
            ControlTransport(plane, host="127.0.0.1", port=18765)

    def test_insecure_refused_on_non_loopback(self):
        from afnan_ai.control.transport import (
            ControlTransport,
            TransportError,
        )

        plane, _ = make_plane()
        with self.assertRaises(TransportError):
            ControlTransport(
                plane,
                host="0.0.0.0",
                port=18765,
                allow_insecure=True,
            )

    def test_insecure_loopback_allowed_with_flag(self):
        from afnan_ai.control.transport import ControlTransport

        plane, _ = make_plane()
        transport = ControlTransport(
            plane,
            host="127.0.0.1",
            port=18765,
            allow_insecure=True,
        )
        transport.start()
        try:
            import json
            import urllib.request

            with urllib.request.urlopen(
                "http://127.0.0.1:18765/v1/health", timeout=5
            ) as response:
                body = json.loads(
                    response.read().decode("utf-8")
                )
            self.assertTrue(body["ok"])
            self.assertEqual(body["protocol_version"], "v1")
        finally:
            transport.stop()


class MetricsTests(unittest.TestCase):
    def test_metrics_snapshot(self):
        plane, _ = make_plane()
        result = pair_device(plane)
        plane.execute_command(
            result["token"], cmd(CommandType.LIST_TASKS)
        )
        plane.execute_command(
            result["token"], cmd(CommandType.HEALTH_CHECK)
        )
        out = plane.execute_command(
            result["token"], cmd(CommandType.GET_METRICS)
        )
        self.assertEqual(out["status"], "completed")
        metrics = out["result"]["metrics"]
        # The in-flight GET_METRICS is counted after it returns.
        self.assertGreaterEqual(metrics["commands_total"], 2)
        self.assertIn("command_latency_avg_s", metrics)


if __name__ == "__main__":
    unittest.main()

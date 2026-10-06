package com.afnan.ai

import com.afnan.ai.data.ActivityItem
import com.afnan.ai.data.AgentStatusView
import com.afnan.ai.data.ApprovalView
import com.afnan.ai.data.ArtifactView
import com.afnan.ai.data.BrowserNavResult
import com.afnan.ai.data.BrowserScreenshotRef
import com.afnan.ai.data.BrowserState
import com.afnan.ai.data.ClientSession
import com.afnan.ai.data.CommandRepository
import com.afnan.ai.data.ControlEvent
import com.afnan.ai.data.ControlJson
import com.afnan.ai.data.ControlPlaneClient
import com.afnan.ai.data.DeviceInfo
import com.afnan.ai.data.GoalView
import com.afnan.ai.data.HealthResponse
import com.afnan.ai.data.HeartbeatResponse
import com.afnan.ai.data.KnownCommands
import com.afnan.ai.data.MilestoneView
import com.afnan.ai.data.PairingResponse
import com.afnan.ai.data.ProtocolException
import com.afnan.ai.data.RedeemResponse
import com.afnan.ai.data.RemoteCommand
import com.afnan.ai.data.RemoteCommandResult
import com.afnan.ai.data.TaskView
import com.afnan.ai.data.WsOpcode
import com.afnan.ai.data.decodeServerFrame
import com.afnan.ai.data.encodeClientFrame
import com.afnan.ai.data.redactTokenText
import java.io.ByteArrayInputStream
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.KSerializer
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * JVM unit tests for the control-plane data layer.
 *
 * These tests verify pure protocol logic with no network and no
 * Android framework: JSON round-trips, the command vocabulary,
 * token redaction, idempotency-key uniqueness and WebSocket frame
 * encoding/decoding.
 */
class ProtocolClientTest {

    private fun <T> roundTrip(value: T, serializer: KSerializer<T>) {
        val json = ControlJson.encodeToString(serializer, value)
        val decoded = ControlJson.decodeFromString(serializer, json)
        assertEquals(value, decoded)
    }

    // -- JSON round-trips -------------------------------------------------

    @Test
    fun `remote command round-trips`() {
        roundTrip(
            RemoteCommand(
                commandId = "cmd_abc123",
                commandType = "start_task",
                payload = buildJsonObject { put("goal_text", "hello") },
                idempotencyKey = "key_xyz",
                createdAt = 1_700_000_000.0,
                expiresAt = 1_700_000_030.0,
                correlationId = "corr1",
            ),
            RemoteCommand.serializer(),
        )
    }

    @Test
    fun `remote command result round-trips`() {
        roundTrip(
            RemoteCommandResult(
                commandId = "cmd_abc123",
                status = "completed",
                result = buildJsonObject { put("ok", true) },
                error = "",
                errorCode = "",
                completedAt = 1_700_000_001.0,
                correlationId = "corr1",
            ),
            RemoteCommandResult.serializer(),
        )
    }

    @Test
    fun `control event round-trips`() {
        roundTrip(
            ControlEvent(
                eventId = "evt_1",
                category = "task.completed",
                seq = 42L,
                at = 1_700_000_000.5,
                deviceId = "phone-1",
                sessionId = "sess_1",
                correlationId = "corr1",
                data = buildJsonObject { put("task_id", "task_1") },
            ),
            ControlEvent.serializer(),
        )
    }

    @Test
    fun `device info round-trips`() {
        roundTrip(
            DeviceInfo(
                deviceId = "phone-1",
                deviceName = "My Phone",
                platform = "android",
                trust = "paired",
                connectionState = "online",
                lastSeen = 1_700_000_000.0,
            ),
            DeviceInfo.serializer(),
        )
    }

    @Test
    fun `client session round-trips`() {
        roundTrip(
            ClientSession(
                sessionId = "sess_1",
                deviceId = "phone-1",
                state = "authorized",
                principal = "owner",
                createdAt = 1_700_000_000.0,
                lastActivity = 1_700_000_001.0,
                expiresAt = 1_700_086_400.0,
            ),
            ClientSession.serializer(),
        )
    }

    @Test
    fun `pairing and redeem responses round-trip`() {
        roundTrip(
            PairingResponse(pairingId = "pair_1", code = "537018"),
            PairingResponse.serializer(),
        )
        roundTrip(
            RedeemResponse(
                deviceId = "phone-1",
                sessionId = "sess_1",
                token = "tok_abc",
                capabilities = listOf("remote.status.read"),
                expiresAt = 1_700_086_400.0,
            ),
            RedeemResponse.serializer(),
        )
    }

    @Test
    fun `health response round-trips`() {
        roundTrip(
            HealthResponse(
                ok = true,
                protocolVersion = "v1",
                uptimeS = 123.4,
                devices = 2,
                sessionsLive = 1,
                emergencyTripped = false,
            ),
            HealthResponse.serializer(),
        )
    }

    @Test
    fun `task view round-trips and maps server goal to title`() {
        val view = TaskView(
            taskId = "task_1",
            title = "Buy milk",
            status = "running",
            progress = 0.5,
            goal = "Buy milk",
            summary = "in progress",
            goalId = "goal_1",
        )
        roundTrip(view, TaskView.serializer())

        // Real server payloads carry the text under "goal".
        val serverObj = buildJsonObject {
            put("task_id", "task_9")
            put("goal", "Write report")
            put("status", "pending")
        }
        val mapped = TaskView.fromServerObject(serverObj)
        assertEquals("task_9", mapped.taskId)
        assertEquals("Write report", mapped.title)
        assertEquals("pending", mapped.status)
    }

    @Test
    fun `goal view round-trips`() {
        roundTrip(
            GoalView(
                goalId = "goal_1",
                description = "Learn Kotlin",
                status = "active",
                progress = 0.25,
                milestones = listOf(
                    MilestoneView("Setup", true),
                    MilestoneView("Basics", false),
                ),
            ),
            GoalView.serializer(),
        )
    }

    @Test
    fun `approval view round-trips with risk_level mapping`() {
        val json = """
            {
              "approval_id": "appr_1",
              "action": "browser_navigate",
              "target": "https://example.com",
              "risk_level": "high",
              "reason": "needs navigation",
              "expires_at": 1700000030.0,
              "status": "pending"
            }
        """.trimIndent()
        val decoded = ControlJson.decodeFromString(
            ApprovalView.serializer(), json
        )
        assertEquals("appr_1", decoded.approvalId)
        assertEquals("browser_navigate", decoded.action)
        assertEquals("https://example.com", decoded.target)
        assertEquals("high", decoded.risk)
        assertEquals("needs navigation", decoded.reason)

        roundTrip(
            ApprovalView(
                approvalId = "appr_2",
                action = "computer_act",
                target = "screen",
                risk = "critical",
                reason = "click",
                expiresAt = 1.0,
            ),
            ApprovalView.serializer(),
        )
    }

    @Test
    fun `artifact view round-trips`() {
        roundTrip(
            ArtifactView(
                artifactId = "art_1",
                name = "report.md",
                artifactType = "document",
                status = "ready",
                currentVersion = 3,
                sensitive = false,
            ),
            ArtifactView.serializer(),
        )
    }

    @Test
    fun `activity item round-trips`() {
        roundTrip(
            ActivityItem(
                type = "task.completed",
                summary = "Task done",
                eventId = "e1",
                seq = 7L,
                at = "2026-10-06T12:00:00Z",
                taskId = "task_1",
                actor = "agent",
                source = "loop",
            ),
            ActivityItem.serializer(),
        )
    }

    @Test
    fun `agent status view round-trips`() {
        roundTrip(
            AgentStatusView(
                status = "working",
                waitingForApproval = true,
                pendingApprovals = 1,
                connectedDevices = 2,
                activeSessions = 1,
                runtimeVersion = "1.2",
            ),
            AgentStatusView.serializer(),
        )
    }

    @Test
    fun `heartbeat and browser models round-trip`() {
        roundTrip(
            HeartbeatResponse("sess_1", "authorized", 1.0),
            HeartbeatResponse.serializer(),
        )
        roundTrip(
            BrowserNavResult("https://example.com", "Example"),
            BrowserNavResult.serializer(),
        )
        roundTrip(
            BrowserScreenshotRef("/tmp/shot.png", 12345L, "2026-10-06"),
            BrowserScreenshotRef.serializer(),
        )
        roundTrip(
            BrowserState("https://example.com", "Example"),
            BrowserState.serializer(),
        )
    }

    @Test
    fun `unknown json keys are ignored`() {
        val json = """
            {
              "task_id": "task_1",
              "status": "running",
              "future_field": {"nested": [1, 2, 3]},
              "another": 42
            }
        """.trimIndent()
        val decoded = ControlJson.decodeFromString(
            TaskView.serializer(), json
        )
        assertEquals("task_1", decoded.taskId)
        assertEquals("running", decoded.status)
    }

    // -- command vocabulary ----------------------------------------------

    @Test
    fun `known commands match the server vocabulary exactly`() {
        val expected = setOf(
            "get_agent_status", "list_tasks", "get_task", "start_task",
            "pause_task", "resume_task", "cancel_task", "retry_task",
            "list_goals", "get_goal", "pause_goal", "resume_goal",
            "cancel_goal", "list_artifacts", "get_artifact",
            "list_approvals", "get_approval", "approve_action",
            "deny_action", "list_devices", "get_device", "list_sessions",
            "query_activity", "get_metrics", "health_check",
            "browser_navigate", "browser_screenshot", "browser_state",
            "computer_observe", "computer_act", "list_schedules",
            "manage_schedule", "emergency_stop", "heartbeat",
            "revoke_device", "revoke_session", "rotate_credentials",
        )
        assertEquals(37, expected.size)
        assertEquals(expected, KnownCommands.ALL)
        assertTrue(
            KnownCommands.ALL.all { it.matches(Regex("[a-z_]+")) }
        )
    }

    @Test
    fun `unknown command type is rejected client-side`() = runBlocking {
        val repo = CommandRepository(
            ControlPlaneClient("http://127.0.0.1:9")
        )
        try {
            repo.execute("tok", "delete_everything", JsonObject(emptyMap()))
            fail("expected ProtocolException")
        } catch (e: ProtocolException) {
            assertEquals("unknown_command_type", e.code)
        }
    }

    // -- token redaction --------------------------------------------------

    @Test
    fun `bearer token is redacted from text`() {
        val token = "tok_s3cr3t_value_12345"
        val redacted = redactTokenText("Authorization: Bearer $token")
        assertFalse(redacted.contains(token))
        assertTrue(redacted.contains("Bearer <redacted>"))
    }

    @Test
    fun `query token is redacted from text`() {
        val token = "tok_s3cr3t_value_12345"
        val redacted = redactTokenText(
            "GET /v1/stream?token=$token HTTP/1.1"
        )
        assertFalse(redacted.contains(token))
        assertTrue(redacted.contains("token=<redacted>"))
    }

    @Test
    fun `hostile server error echo is redacted`() {
        val token = "tok_s3cr3t_value_12345"
        // Bearer credentials in any casing are redacted.
        val hostile = "Authorization: Bearer $token rejected"
        val redacted = redactTokenText(hostile)
        assertFalse(redacted.contains(token))
        assertTrue(redacted.contains("Bearer <redacted>"))
        // Token query parameters are redacted too.
        val hostile2 = "GET /v1/events?token=$token&cursor=12 failed"
        val redacted2 = redactTokenText(hostile2)
        assertFalse(redacted2.contains(token))
        assertTrue(redacted2.contains("token=<redacted>"))
    }

    // -- idempotency ------------------------------------------------------

    @Test
    fun `idempotency keys are unique`() {
        val repo = CommandRepository(
            ControlPlaneClient("http://127.0.0.1:9")
        )
        val keys = (1..1000).map { repo.newIdempotencyKey() }.toSet()
        assertEquals(1000, keys.size)
        assertTrue(keys.all { it.startsWith("key_") && it.length > 10 })
    }

    // -- WebSocket framing -------------------------------------------------

    @Test
    fun `masked client frame round-trips through manual unmasking`() {
        val payload = "hello afnan".toByteArray(Charsets.UTF_8)
        val frame = encodeClientFrame(WsOpcode.TEXT, payload)

        // FIN + text opcode.
        assertEquals(0x81.toByte(), frame[0])
        // Mask bit must be set (RFC 6455: clients MUST mask).
        assertTrue((frame[1].toInt() and 0x80) != 0)

        val length = frame[1].toInt() and 0x7F
        assertEquals(payload.size, length)
        val mask = frame.copyOfRange(2, 6)
        val masked = frame.copyOfRange(6, 6 + length)
        val unmasked = ByteArray(masked.size) { i ->
            (masked[i].toInt() xor mask[i % 4].toInt()).toByte()
        }
        assertArrayEquals(payload, unmasked)
    }

    @Test
    fun `unmasked server frame is accepted`() {
        val payload = "hello".toByteArray(Charsets.UTF_8)
        val raw = byteArrayOf(0x81.toByte(), payload.size.toByte()) +
            payload
        val frame = decodeServerFrame(ByteArrayInputStream(raw))
        assertEquals(WsOpcode.TEXT, frame!!.opcode)
        assertArrayEquals(payload, frame.payload)
    }

    @Test
    fun `extended-length server frame is accepted`() {
        val payload = ByteArray(200) { 'x'.code.toByte() }
        val raw = byteArrayOf(
            0x81.toByte(), 126.toByte(), 0x00.toByte(), 200.toByte()
        ) + payload
        val frame = decodeServerFrame(ByteArrayInputStream(raw))
        assertEquals(WsOpcode.TEXT, frame!!.opcode)
        assertEquals(200, frame.payload.size)
    }

    @Test
    fun `masked server frame is rejected`() {
        val payload = "hello".toByteArray(Charsets.UTF_8)
        val mask = byteArrayOf(1, 2, 3, 4)
        val masked = ByteArray(payload.size) { i ->
            (payload[i].toInt() xor mask[i % 4].toInt()).toByte()
        }
        val raw = byteArrayOf(
            0x81.toByte(), (0x80 or payload.size).toByte()
        ) + mask + masked
        try {
            decodeServerFrame(ByteArrayInputStream(raw))
            fail("expected ProtocolException")
        } catch (e: ProtocolException) {
            assertEquals("ws_masked_server", e.code)
        }
    }

    @Test
    fun `oversized server frame is rejected before allocation`() {
        // Declared length 2,000,000 with no payload bytes following.
        val lengthBytes = ByteArray(8) { i ->
            ((2_000_000L ushr ((7 - i) * 8)) and 0xFF).toByte()
        }
        val raw = byteArrayOf(0x81.toByte(), 127.toByte()) + lengthBytes
        try {
            decodeServerFrame(ByteArrayInputStream(raw))
            fail("expected ProtocolException")
        } catch (e: ProtocolException) {
            assertEquals("ws_frame_too_big", e.code)
        }
    }

    @Test
    fun `reserved opcode is rejected`() {
        val raw = byteArrayOf(0x83.toByte(), 0x00.toByte())
        try {
            decodeServerFrame(ByteArrayInputStream(raw))
            fail("expected ProtocolException")
        } catch (e: ProtocolException) {
            assertEquals("ws_bad_opcode", e.code)
        }
    }

    @Test
    fun `empty stream decodes to clean eof`() {
        assertEquals(
            null, decodeServerFrame(ByteArrayInputStream(ByteArray(0)))
        )
    }
}

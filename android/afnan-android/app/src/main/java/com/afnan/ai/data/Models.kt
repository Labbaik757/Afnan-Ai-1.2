package com.afnan.ai.data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * Shared JSON configuration for the control-plane wire protocol.
 *
 * The server may add fields over time; unknown keys are ignored so
 * older clients keep working against newer servers.
 */
val ControlJson: Json = Json {
    ignoreUnknownKeys = true
    isLenient = true
    explicitNulls = false
    encodeDefaults = true
}

/** Protocol version spoken by the server (`v1`). */
const val PROTOCOL_VERSION = "v1"

/** One remote command, as sent in `POST /v1/commands`. */
@Serializable
data class RemoteCommand(
    @SerialName("command_id") val commandId: String,
    @SerialName("command_type") val commandType: String,
    @SerialName("payload") val payload: JsonObject = JsonObject(emptyMap()),
    @SerialName("idempotency_key") val idempotencyKey: String,
    @SerialName("created_at") val createdAt: Double = 0.0,
    @SerialName("expires_at") val expiresAt: Double = 0.0,
    @SerialName("correlation_id") val correlationId: String = "",
    @SerialName("protocol_version") val protocolVersion: String = PROTOCOL_VERSION,
)

/**
 * Result of a remote command.
 *
 * Note: the server answers HTTP 200 even when the command itself was
 * denied or failed — callers must inspect [status] instead of relying
 * on the HTTP code.
 */
@Serializable
data class RemoteCommandResult(
    @SerialName("command_id") val commandId: String = "",
    @SerialName("status") val status: String = "",
    @SerialName("result") val result: JsonObject = JsonObject(emptyMap()),
    @SerialName("error") val error: String = "",
    @SerialName("error_code") val errorCode: String = "",
    @SerialName("verification") val verification: JsonObject = JsonObject(emptyMap()),
    @SerialName("completed_at") val completedAt: Double = 0.0,
    @SerialName("correlation_id") val correlationId: String = "",
    @SerialName("protocol_version") val protocolVersion: String = PROTOCOL_VERSION,
) {
    val isCompleted: Boolean get() = status == "completed"
}

/** One event from the control-plane event stream. Never carries secrets. */
@Serializable
data class ControlEvent(
    @SerialName("event_id") val eventId: String = "",
    @SerialName("category") val category: String = "",
    @SerialName("seq") val seq: Long = 0L,
    @SerialName("at") val at: Double = 0.0,
    @SerialName("device_id") val deviceId: String = "",
    @SerialName("session_id") val sessionId: String = "",
    @SerialName("correlation_id") val correlationId: String = "",
    @SerialName("data") val data: JsonObject = JsonObject(emptyMap()),
    @SerialName("protocol_version") val protocolVersion: String = PROTOCOL_VERSION,
)

/** Sanitized device projection (no secret material). */
@Serializable
data class DeviceInfo(
    @SerialName("device_id") val deviceId: String = "",
    @SerialName("device_name") val deviceName: String = "",
    @SerialName("platform") val platform: String = "",
    @SerialName("trust") val trust: String = "",
    @SerialName("connection_state") val connectionState: String = "",
    @SerialName("last_seen") val lastSeen: Double = 0.0,
)

/** Sanitized session projection (no token material). */
@Serializable
data class ClientSession(
    @SerialName("session_id") val sessionId: String = "",
    @SerialName("device_id") val deviceId: String = "",
    @SerialName("state") val state: String = "",
    @SerialName("principal") val principal: String = "",
    @SerialName("created_at") val createdAt: Double = 0.0,
    @SerialName("last_activity") val lastActivity: Double = 0.0,
    @SerialName("expires_at") val expiresAt: Double = 0.0,
)

/** `POST /v1/pair/request` response. The code is shown once, in memory only. */
@Serializable
data class PairingResponse(
    @SerialName("pairing_id") val pairingId: String = "",
    @SerialName("code") val code: String = "",
)

/** `POST /v1/pair/redeem` response. */
@Serializable
data class RedeemResponse(
    @SerialName("device_id") val deviceId: String = "",
    @SerialName("session_id") val sessionId: String = "",
    @SerialName("token") val token: String = "",
    @SerialName("capabilities") val capabilities: List<String> = emptyList(),
    @SerialName("expires_at") val expiresAt: Double = 0.0,
)

/** `GET /v1/health` response. */
@Serializable
data class HealthResponse(
    @SerialName("ok") val ok: Boolean = false,
    @SerialName("protocol_version") val protocolVersion: String = "",
    @SerialName("uptime_s") val uptimeS: Double = 0.0,
    @SerialName("devices") val devices: Int = 0,
    @SerialName("sessions_live") val sessionsLive: Int = 0,
    @SerialName("emergency_tripped") val emergencyTripped: Boolean = false,
)

/**
 * Sanitized task projection.
 *
 * The server's task view carries the natural-language goal text under
 * `goal`; [title] is populated from it when decoding real server
 * payloads (see [fromServerObject]).
 */
@Serializable
data class TaskView(
    @SerialName("task_id") val taskId: String = "",
    @SerialName("title") val title: String = "",
    @SerialName("status") val status: String = "",
    @SerialName("progress") val progress: Double = 0.0,
    @SerialName("goal") val goal: String = "",
    @SerialName("summary") val summary: String = "",
    @SerialName("error") val error: String = "",
    @SerialName("goal_id") val goalId: String = "",
    @SerialName("priority") val priority: Int = 0,
    @SerialName("attempts") val attempts: Int = 0,
    @SerialName("created_at") val createdAt: Double = 0.0,
    @SerialName("updated_at") val updatedAt: Double = 0.0,
    @SerialName("has_checkpoint") val hasCheckpoint: Boolean = false,
) {
    companion object {
        /** Build a [TaskView] from a raw server task object. */
        fun fromServerObject(obj: JsonObject): TaskView {
            val base = ControlJson.decodeFromJsonElement(
                serializer(), obj
            )
            val goalText = obj["goal"]?.jsonPrimitive?.contentOrNull.orEmpty()
            val title = base.title.ifEmpty { goalText }
            return base.copy(title = title)
        }
    }
}

/** Milestone inside a [GoalView]. */
@Serializable
data class MilestoneView(
    @SerialName("title") val title: String = "",
    @SerialName("completed") val completed: Boolean = false,
)

/** Sanitized goal projection. */
@Serializable
data class GoalView(
    @SerialName("goal_id") val goalId: String = "",
    @SerialName("description") val description: String = "",
    @SerialName("status") val status: String = "",
    @SerialName("progress") val progress: Double = 0.0,
    @SerialName("milestones") val milestones: List<MilestoneView> = emptyList(),
    @SerialName("priority") val priority: Int = 0,
    @SerialName("created_at") val createdAt: Double = 0.0,
    @SerialName("updated_at") val updatedAt: Double = 0.0,
)

/**
 * Sanitized approval projection.
 *
 * The server names the risk field `risk_level`; it is exposed here
 * as [risk].
 */
@Serializable
data class ApprovalView(
    @SerialName("approval_id") val approvalId: String = "",
    @SerialName("action") val action: String = "",
    @SerialName("target") val target: String = "",
    @SerialName("risk_level") val risk: String = "",
    @SerialName("reason") val reason: String = "",
    @SerialName("expires_at") val expiresAt: Double = 0.0,
    @SerialName("task_id") val taskId: String = "",
    @SerialName("status") val status: String = "",
    @SerialName("requested_capability") val requestedCapability: String = "",
    @SerialName("is_expired") val isExpired: Boolean = false,
    @SerialName("created_at") val createdAt: Double = 0.0,
)

/** Sanitized artifact projection (sensitive ones need `remote.admin`). */
@Serializable
data class ArtifactView(
    @SerialName("artifact_id") val artifactId: String = "",
    @SerialName("name") val name: String = "",
    @SerialName("artifact_type") val artifactType: String = "",
    @SerialName("status") val status: String = "",
    @SerialName("current_version") val currentVersion: Int = 0,
    @SerialName("sensitive") val sensitive: Boolean = false,
    @SerialName("created_at") val createdAt: String = "",
    @SerialName("updated_at") val updatedAt: String = "",
)

/** One user-safe activity record from `query_activity`. */
@Serializable
data class ActivityItem(
    @SerialName("type") val type: String = "",
    @SerialName("summary") val summary: String = "",
    @SerialName("event_id") val eventId: String = "",
    @SerialName("seq") val seq: Long = 0L,
    @SerialName("at") val at: String = "",
    @SerialName("task_id") val taskId: String = "",
    @SerialName("goal_id") val goalId: String = "",
    @SerialName("actor") val actor: String = "",
    @SerialName("source") val source: String = "",
)

/** Concise agent snapshot from `get_agent_status`. */
@Serializable
data class AgentStatusView(
    @SerialName("status") val status: String = "",
    @SerialName("current_task") val currentTask: TaskView? = null,
    @SerialName("current_goal") val currentGoal: GoalView? = null,
    @SerialName("waiting_for_approval") val waitingForApproval: Boolean = false,
    @SerialName("pending_approvals") val pendingApprovals: Int = 0,
    @SerialName("error") val error: String = "",
    @SerialName("connected_devices") val connectedDevices: Int = 0,
    @SerialName("active_sessions") val activeSessions: Int = 0,
    @SerialName("emergency_tripped") val emergencyTripped: Boolean = false,
    @SerialName("runtime_version") val runtimeVersion: String = "",
)

/** `POST /v1/session/heartbeat` response. */
@Serializable
data class HeartbeatResponse(
    @SerialName("session_id") val sessionId: String = "",
    @SerialName("state") val state: String = "",
    @SerialName("at") val at: Double = 0.0,
)

/** `browser_navigate` result. */
@Serializable
data class BrowserNavResult(
    @SerialName("url") val url: String = "",
    @SerialName("title") val title: String = "",
)

/** `browser_screenshot` result (server-side reference, not raw bytes). */
@Serializable
data class BrowserScreenshotRef(
    @SerialName("path") val path: String = "",
    @SerialName("size_bytes") val sizeBytes: Long = 0L,
    @SerialName("captured_at") val capturedAt: String = "",
    @SerialName("data_base64") val dataBase64: String = "",
)

/** `browser_state` result. */
@Serializable
data class BrowserState(
    @SerialName("url") val url: String = "",
    @SerialName("title") val title: String = "",
)

/** Internal helper: read a string field from a [JsonObject] safely. */
internal fun JsonObject.stringOrEmpty(key: String): String =
    this[key]?.jsonPrimitive?.contentOrNull.orEmpty()

/** Internal helper: read a nested object field safely. */
internal fun JsonObject.objectOrEmpty(key: String): JsonObject =
    this[key]?.jsonObject ?: JsonObject(emptyMap())

/** Internal helper: read a double field safely. */
internal fun JsonObject.doubleOrZero(key: String): Double =
    this[key]?.jsonPrimitive?.doubleOrNull ?: 0.0

/** Internal helper: read a boolean field safely. */
internal fun JsonObject.booleanOrFalse(key: String): Boolean =
    this[key]?.jsonPrimitive?.booleanOrNull ?: false

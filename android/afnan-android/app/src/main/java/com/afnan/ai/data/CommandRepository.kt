package com.afnan.ai.data

import java.util.UUID
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Every `command_type` the server accepts, copied verbatim from the
 * protocol reference (`afnan_ai/control/models.py`, lowercase).
 * Anything not in this set is rejected client-side before it can
 * reach the network.
 */
object KnownCommands {
    val ALL: Set<String> = setOf(
        "get_agent_status",
        "list_tasks",
        "get_task",
        "start_task",
        "pause_task",
        "resume_task",
        "cancel_task",
        "retry_task",
        "list_goals",
        "get_goal",
        "pause_goal",
        "resume_goal",
        "cancel_goal",
        "list_artifacts",
        "get_artifact",
        "list_approvals",
        "get_approval",
        "approve_action",
        "deny_action",
        "list_devices",
        "get_device",
        "list_sessions",
        "query_activity",
        "get_metrics",
        "health_check",
        "browser_navigate",
        "browser_screenshot",
        "browser_state",
        "computer_observe",
        "computer_act",
        "list_schedules",
        "manage_schedule",
        "emergency_stop",
        "heartbeat",
        "revoke_device",
        "revoke_session",
        "rotate_credentials",
    )
}

/**
 * Validated, idempotent remote commands against `POST /v1/commands`.
 *
 * Mirrors `afnan_ai/control/client/commands.py` (CommandClient).
 * A command-level failure (denied/failed/expired/duplicate) is
 * returned as data in [RemoteCommandResult]; transport problems
 * raise [ProtocolException] subclasses.
 */
class CommandRepository(
    private val client: ControlPlaneClient,
) {
    /**
     * Execute one command and return the server's result.
     *
     * The result's `status` may be `completed`, `failed`, `denied`,
     * `expired` or `duplicate` — always check it.
     */
    suspend fun execute(
        token: String,
        commandType: String,
        payload: JsonObject = JsonObject(emptyMap()),
        idempotencyKey: String? = null,
        timeoutMs: Long = 30_000,
    ): RemoteCommandResult {
        if (commandType !in KnownCommands.ALL) {
            throw ProtocolException(
                "unknown command_type '$commandType'",
                "unknown_command_type",
            )
        }
        if (payload.toString().length > MAX_PAYLOAD_CHARS) {
            throw ProtocolException(
                "payload too large",
                "payload_too_large",
            )
        }
        val created = System.currentTimeMillis() / 1000.0
        val body = buildJsonObject {
            put("command_id", "cmd_${UUID.randomUUID().toString().replace("-", "").take(16)}")
            put("command_type", commandType)
            put("payload", payload)
            put("idempotency_key", idempotencyKey ?: newIdempotencyKey())
            put("created_at", created)
            put("expires_at", created + maxOf(timeoutMs / 1000.0, 1.0))
            put("protocol_version", PROTOCOL_VERSION)
        }
        val response = client.post(
            "/v1/commands", body, token, timeoutMs.toInt()
        )
        val result = ControlJson.decodeFromJsonElement(
            RemoteCommandResult.serializer(), response
        )
        if (result.status.isEmpty()) {
            throw ProtocolException(
                "server returned a malformed command result",
                "invalid_response",
            )
        }
        return result
    }

    /**
     * Request server-side cancellation of a task.
     * The authoritative TaskManager performs the state transition;
     * this client stops nothing locally.
     */
    suspend fun cancelTask(
        token: String,
        taskId: String,
        idempotencyKey: String? = null,
    ): RemoteCommandResult {
        if (taskId.isBlank()) {
            throw ProtocolException(
                "task_id is required to cancel a task",
                "invalid_target",
            )
        }
        return execute(
            token,
            "cancel_task",
            buildJsonObject { put("task_id", taskId) },
            idempotencyKey,
        )
    }

    /** Generate a fresh idempotency key. Keys are unique per call. */
    fun newIdempotencyKey(): String =
        "key_${UUID.randomUUID().toString().replace("-", "")}"

    companion object {
        /** Mirrors the server's payload size guard. */
        const val MAX_PAYLOAD_CHARS = 65_536
    }
}

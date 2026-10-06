package com.afnan.ai.data

import com.afnan.ai.security.TokenStore
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put

/**
 * Facade over the control-plane client layer.
 *
 * Combines [ControlPlaneClient], [PairingRepository],
 * [CommandRepository], [EventStream] and [TokenStore] into the
 * operations the Android UI needs. Every method performs real
 * network work against the paired Afnan runtime; nothing is mocked.
 *
 * Getters throw [ProtocolException] (with the server's own error
 * message/code) when the command result status is not `completed`.
 * Action methods that the protocol answers with an explicit decision
 * ([approve], [deny], [revokeDevice]) return the raw
 * [RemoteCommandResult] so callers can inspect `status`
 * (`completed`/`denied`/`failed`/...) themselves.
 */
class AfnanRepository(
    private val tokenStore: TokenStore,
    private val clientFactory: (String) -> ControlPlaneClient = { url ->
        ControlPlaneClient(url)
    },
) {
    private var client: ControlPlaneClient? = null

    /** Backoff driver for event streaming; owned by the UI/worker layer. */
    val reconnectManager = ReconnectManager()

    private fun clientFor(baseUrl: String): ControlPlaneClient {
        val current = client
        if (current != null && current.baseUrl == baseUrl.trimEnd('/')) {
            return current
        }
        current?.close()
        return clientFactory(baseUrl).also { client = it }
    }

    private fun storedClient(): ControlPlaneClient {
        val url = tokenStore.baseUrl
            ?: throw ConnectionException(
                "not paired: no server URL stored", "not_paired"
            )
        return clientFor(url)
    }

    private fun storedToken(): String =
        tokenStore.token ?: throw AuthenticationException(
            "no session token stored", "no_token"
        )

    private suspend fun execute(
        commandType: String,
        payload: JsonObject = JsonObject(emptyMap()),
        idempotencyKey: String? = null,
    ): RemoteCommandResult =
        CommandRepository(storedClient()).execute(
            storedToken(), commandType, payload, idempotencyKey
        )

    /**
     * Throw with the server's message when a command did not
     * complete; otherwise return its `result` object.
     */
    private fun requireCompleted(result: RemoteCommandResult): JsonObject {
        if (!result.isCompleted) {
            throw ProtocolException(
                result.error.ifEmpty { "command ${result.status}" },
                result.errorCode.ifEmpty { "command_${result.status}" },
                null,
            )
        }
        return result.result
    }

    /** Forget the pairing and drop the HTTP client. */
    fun unpair() {
        tokenStore.clear()
        client?.close()
        client = null
    }

    // -- pairing ---------------------------------------------------------

    /**
     * Start pairing against the PC at [baseUrl].
     * Returns `Pair(pairing_id, code)`; show the code to the user once.
     */
    suspend fun pair(
        deviceId: String,
        deviceName: String,
        baseUrl: String,
    ): Pair<String, String> =
        PairingRepository(clientFor(baseUrl)).requestPairing(
            deviceId, deviceName
        )

    /**
     * Redeem an owner-approved pairing code and persist the session.
     */
    suspend fun redeemAndStore(
        pairingId: String,
        code: String,
        baseUrl: String,
    ): RedeemResponse {
        val response = PairingRepository(clientFor(baseUrl)).redeem(
            pairingId, code
        )
        tokenStore.saveSession(
            baseUrl, response.token, response.sessionId,
            response.deviceId,
        )
        return response
    }

    // -- health ----------------------------------------------------------

    /** Unauthenticated liveness probe. */
    suspend fun health(baseUrl: String? = null): HealthResponse {
        val url = baseUrl ?: tokenStore.baseUrl
            ?: throw ConnectionException(
                "no server URL available", "not_paired"
            )
        val obj = clientFor(url).get("/v1/health")
        return ControlJson.decodeFromJsonElement(
            HealthResponse.serializer(), obj
        )
    }

    // -- tasks -----------------------------------------------------------

    /**
     * Submit a natural-language task. The server's `start_task`
     * handler reads `goal_text`; `title` is also sent per the
     * integration contract and ignored by the server otherwise.
     */
    suspend fun sendTask(text: String): TaskView {
        require(text.isNotBlank()) { "task text must not be blank" }
        val payload = buildJsonObject {
            put("goal_text", text)
            put("title", text)
        }
        val result = requireCompleted(execute("start_task", payload))
        return TaskView.fromServerObject(result.objectOrEmpty("task"))
    }

    suspend fun listTasks(): List<TaskView> {
        val result = requireCompleted(execute("list_tasks"))
        return result["tasks"]?.jsonArray?.map {
            TaskView.fromServerObject(it.jsonObject)
        } ?: emptyList()
    }

    suspend fun getTask(taskId: String): TaskView {
        val result = requireCompleted(
            execute("get_task", taskIdPayload(taskId))
        )
        return TaskView.fromServerObject(result.objectOrEmpty("task"))
    }

    suspend fun pauseTask(taskId: String): TaskView {
        val result = requireCompleted(
            execute("pause_task", taskIdPayload(taskId))
        )
        return TaskView.fromServerObject(result.objectOrEmpty("task"))
    }

    suspend fun resumeTask(taskId: String): TaskView {
        val result = requireCompleted(
            execute("resume_task", taskIdPayload(taskId))
        )
        return TaskView.fromServerObject(result.objectOrEmpty("task"))
    }

    suspend fun cancelTask(taskId: String): TaskView {
        val result = requireCompleted(
            execute("cancel_task", taskIdPayload(taskId))
        )
        return TaskView.fromServerObject(result.objectOrEmpty("task"))
    }

    // -- goals -----------------------------------------------------------

    suspend fun listGoals(): List<GoalView> {
        val result = requireCompleted(execute("list_goals"))
        return result["goals"]?.jsonArray?.map {
            ControlJson.decodeFromJsonElement(
                GoalView.serializer(), it.jsonObject
            )
        } ?: emptyList()
    }

    suspend fun getGoal(goalId: String): GoalView {
        val payload = buildJsonObject { put("goal_id", goalId) }
        val result = requireCompleted(execute("get_goal", payload))
        return ControlJson.decodeFromJsonElement(
            GoalView.serializer(), result.objectOrEmpty("goal")
        )
    }

    // -- approvals -------------------------------------------------------

    suspend fun listApprovals(): List<ApprovalView> {
        val result = requireCompleted(execute("list_approvals"))
        return result["approvals"]?.jsonArray?.map {
            ControlJson.decodeFromJsonElement(
                ApprovalView.serializer(), it.jsonObject
            )
        } ?: emptyList()
    }

    /**
     * Approve an action. Returns the raw result — the caller must
     * check `status` (`completed` vs `denied`/`failed`) per protocol.
     */
    suspend fun approve(approvalId: String): RemoteCommandResult =
        execute("approve_action", approvalPayload(approvalId))

    /** Deny an action. Returns the raw result; check `status`. */
    suspend fun deny(approvalId: String): RemoteCommandResult =
        execute("deny_action", approvalPayload(approvalId))

    // -- devices ---------------------------------------------------------

    suspend fun listDevices(): List<DeviceInfo> {
        val result = requireCompleted(execute("list_devices"))
        return result["devices"]?.jsonArray?.map {
            ControlJson.decodeFromJsonElement(
                DeviceInfo.serializer(), it.jsonObject
            )
        } ?: emptyList()
    }

    /**
     * Revoke a device. Returns the raw result; check `status`.
     * Revocation immediately invalidates the device's sessions and
     * tokens server-side.
     */
    suspend fun revokeDevice(deviceId: String): RemoteCommandResult {
        val payload = buildJsonObject { put("device_id", deviceId) }
        return execute("revoke_device", payload)
    }

    // -- artifacts -------------------------------------------------------

    suspend fun listArtifacts(): List<ArtifactView> {
        val result = requireCompleted(execute("list_artifacts"))
        return result["artifacts"]?.jsonArray?.map {
            ControlJson.decodeFromJsonElement(
                ArtifactView.serializer(), it.jsonObject
            )
        } ?: emptyList()
    }

    // -- agent & activity ------------------------------------------------

    suspend fun getAgentStatus(): AgentStatusView {
        val result = requireCompleted(execute("get_agent_status"))
        return ControlJson.decodeFromJsonElement(
            AgentStatusView.serializer(), result
        )
    }

    /**
     * Query the activity feed. Returns `Pair(lastSeq, items)`.
     */
    suspend fun queryActivity(
        afterSeq: Long = 0,
        limit: Int = 100,
    ): Pair<Long, List<ActivityItem>> {
        val payload = buildJsonObject {
            put("after_seq", afterSeq)
            put("limit", limit.coerceIn(1, 500))
        }
        val result = requireCompleted(execute("query_activity", payload))
        val lastSeq = result["last_seq"]?.jsonPrimitive?.contentOrNull
            ?.toLongOrNull() ?: afterSeq
        val items = result["events"]?.jsonArray?.mapNotNull {
            try {
                ControlJson.decodeFromJsonElement(
                    ActivityItem.serializer(), it.jsonObject
                )
            } catch (e: Exception) {
                null
            }
        } ?: emptyList()
        return lastSeq to items
    }

    // -- safety ----------------------------------------------------------

    /**
     * Trip the emergency stop. Throws unless the server confirms;
     * a safety action must never fail silently.
     */
    suspend fun emergencyStop(
        reason: String = "remote emergency stop",
    ): JsonObject {
        val payload = buildJsonObject { put("reason", reason) }
        return requireCompleted(execute("emergency_stop", payload))
    }

    // -- browser ---------------------------------------------------------

    suspend fun browserNavigate(url: String): BrowserNavResult {
        require(url.isNotBlank()) { "url must not be blank" }
        val payload = buildJsonObject { put("url", url) }
        val result = requireCompleted(execute("browser_navigate", payload))
        return ControlJson.decodeFromJsonElement(
            BrowserNavResult.serializer(), result
        )
    }

    suspend fun browserScreenshot(): BrowserScreenshotRef {
        // Ask for inline base64: thin clients cannot fetch the
        // server-side path directly (no download endpoint exists).
        val payload = buildJsonObject { put("inline", true) }
        val result = requireCompleted(execute("browser_screenshot", payload))
        return ControlJson.decodeFromJsonElement(
            BrowserScreenshotRef.serializer(), result
        )
    }

    suspend fun browserState(): BrowserState {
        val result = requireCompleted(execute("browser_state"))
        return ControlJson.decodeFromJsonElement(
            BrowserState.serializer(), result.objectOrEmpty("state")
        )
    }

    // -- session ---------------------------------------------------------

    /** One heartbeat; keeps the session warm. */
    suspend fun heartbeat(): HeartbeatResponse {
        val obj = storedClient().post(
            "/v1/session/heartbeat", null, storedToken()
        )
        return ControlJson.decodeFromJsonElement(
            HeartbeatResponse.serializer(), obj
        )
    }

    /**
     * Launch a background heartbeat loop. Transient network errors
     * are retried on the next tick. An [AuthenticationException]
     * (dead token) ends the loop: it is delivered to [onAuthFailure]
     * when provided so the UI can navigate to re-pair, otherwise it
     * is rethrown.
     */
    fun heartbeatLoop(
        scope: CoroutineScope,
        intervalMs: Long = 60_000,
        onAuthFailure: ((AuthenticationException) -> Unit)? = null,
    ): Job = scope.launch {
        while (isActive) {
            try {
                heartbeat()
            } catch (e: CancellationException) {
                throw e
            } catch (e: AuthenticationException) {
                if (onAuthFailure != null) {
                    onAuthFailure(e)
                    return@launch
                }
                throw e
            } catch (e: Exception) {
                // Transient failure — retry on the next tick.
            }
            delay(intervalMs)
        }
    }

    /**
     * Rotate the bearer token. The old token dies immediately on
     * the server; the session id stays stable. The new token is
     * persisted and returned.
     */
    suspend fun rotateToken(): String {
        val old = storedToken()
        val response = storedClient().post(
            "/v1/session/rotate",
            buildJsonObject { put("token", old) },
            old,
        )
        val newToken = response.stringOrEmpty("token")
        if (newToken.isEmpty()) {
            throw ProtocolException(
                "token rotation returned no token", "rotation_failed"
            )
        }
        tokenStore.updateToken(newToken)
        return newToken
    }

    // -- events ----------------------------------------------------------

    /** Create an SSE stream bound to this repository's HTTP client. */
    fun eventStream(): EventStream = EventStream(storedClient())

    /** Create a WebSocket connection to the stored server. */
    fun newWsConnection(): WsConnection =
        WsConnection(
            (tokenStore.baseUrl
                ?: throw ConnectionException(
                    "no server URL available", "not_paired"
                )),
            storedClient().timeoutMs,
        )

    /** The bearer token currently in custody (for streams). */
    fun currentToken(): String = storedToken()

    // -- helpers ---------------------------------------------------------

    private fun taskIdPayload(taskId: String): JsonObject {
        require(taskId.isNotBlank()) { "task_id must not be blank" }
        return buildJsonObject { put("task_id", taskId) }
    }

    private fun approvalPayload(approvalId: String): JsonObject {
        require(approvalId.isNotBlank()) { "approval_id must not be blank" }
        return buildJsonObject { put("approval_id", approvalId) }
    }
}

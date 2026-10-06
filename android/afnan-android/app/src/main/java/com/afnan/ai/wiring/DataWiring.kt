package com.afnan.ai.wiring

import android.content.Context
import com.afnan.ai.data.AfnanRepository as DataRepository
import com.afnan.ai.data.ActivityItem as DataActivityItem
import com.afnan.ai.data.AgentStatusView as DataAgentStatus
import com.afnan.ai.data.ApprovalView as DataApproval
import com.afnan.ai.data.ArtifactView as DataArtifact
import com.afnan.ai.data.AuthenticationException
import com.afnan.ai.data.BrowserNavResult as DataBrowserNav
import com.afnan.ai.data.BrowserState as DataBrowserState
import com.afnan.ai.data.CommandRepository
import com.afnan.ai.data.ConnectionException
import com.afnan.ai.data.ControlEvent as DataEvent
import com.afnan.ai.data.ControlPlaneClient
import com.afnan.ai.data.DeviceInfo as DataDevice
import com.afnan.ai.data.GoalView as DataGoal
import com.afnan.ai.data.ProtocolException
import com.afnan.ai.data.TaskView as DataTask
import com.afnan.ai.security.TokenStore as SecureTokenStore
import com.afnan.ai.ui.ActivityItem
import com.afnan.ai.ui.AfnanRepository as UiRepository
import com.afnan.ai.ui.AgentStatus
import com.afnan.ai.ui.ApprovalStatus
import com.afnan.ai.ui.ApprovalView
import com.afnan.ai.ui.ArtifactView
import com.afnan.ai.ui.BrowserStateView
import com.afnan.ai.ui.ConnectorView
import com.afnan.ai.ui.CredentialMeta
import com.afnan.ai.ui.DeviceView
import com.afnan.ai.ui.GoalView
import com.afnan.ai.ui.MilestoneView
import com.afnan.ai.ui.PairingInfo
import com.afnan.ai.ui.RemoteEvent
import com.afnan.ai.ui.RiskLevel
import com.afnan.ai.ui.TaskStatus
import com.afnan.ai.ui.TaskView
import com.afnan.ai.ui.TokenStore as UiTokenStore
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.time.Instant
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put

// ---------------------------------------------------------------------------
// Adapters: bind the real data layer (data/, security/) to the UI contracts
// (ui/UiContracts.kt). Every mapping is explicit; unknown enum values fail
// safe rather than crashing.
// ---------------------------------------------------------------------------

private const val SETUP_PREFS = "afnan_setup"
private const val KEY_SETUP_URL = "setup_base_url"

/** Epoch-seconds Double from the server -> epoch millis. */
private fun epochSecToMs(epochSeconds: Double): Long =
    (epochSeconds * 1000).toLong()

/** ISO-8601 or epoch (seconds or millis) string -> epoch millis. */
private fun parseAt(at: String): Long {
    if (at.isBlank()) return 0L
    try {
        return Instant.parse(at).toEpochMilli()
    } catch (_: Exception) {
        // Not ISO — fall through to numeric parsing.
    }
    val n = at.toDoubleOrNull() ?: return 0L
    return if (n > 1e12) n.toLong() else (n * 1000).toLong()
}

/** Server progress (0..1 fraction or 0..100 percent) -> 0f..1f. */
private fun progressToFraction(progress: Double): Float {
    val f = if (progress in 0.0..1.0) progress.toFloat()
    else (progress / 100).toFloat()
    return f.coerceIn(0f, 1f)
}

private fun parseTaskStatus(status: String): TaskStatus = when (status.lowercase()) {
    "pending" -> TaskStatus.PENDING
    "running" -> TaskStatus.RUNNING
    "paused" -> TaskStatus.PAUSED
    "completed" -> TaskStatus.COMPLETED
    "failed" -> TaskStatus.FAILED
    "cancelled", "canceled" -> TaskStatus.CANCELLED
    else -> TaskStatus.PENDING
}

private fun parseRiskLevel(risk: String): RiskLevel = when (risk.lowercase()) {
    "low" -> RiskLevel.LOW
    "medium", "moderate" -> RiskLevel.MEDIUM
    "high" -> RiskLevel.HIGH
    "critical" -> RiskLevel.CRITICAL
    // Fail-safe: an unknown risk level is treated as high, never low.
    else -> RiskLevel.HIGH
}

private fun parseApprovalStatus(status: String): ApprovalStatus =
    when (status.lowercase()) {
        "pending" -> ApprovalStatus.PENDING
        "approved" -> ApprovalStatus.APPROVED
        "denied" -> ApprovalStatus.DENIED
        "expired" -> ApprovalStatus.EXPIRED
        else -> ApprovalStatus.PENDING
    }

private fun mapTask(t: DataTask): TaskView = TaskView(
    id = t.taskId,
    title = t.title.ifBlank { t.goal.ifBlank { t.taskId } },
    status = parseTaskStatus(t.status),
    progress = progressToFraction(t.progress),
    createdAt = epochSecToMs(t.createdAt),
    updatedAt = epochSecToMs(t.updatedAt),
    resultSummary = t.summary.ifBlank { null },
    error = t.error.ifBlank { null },
)

private fun mapGoal(g: DataGoal): GoalView = GoalView(
    id = g.goalId,
    title = g.description.ifBlank { g.goalId },
    description = g.description,
    progress = progressToFraction(g.progress),
    status = g.status,
    milestones = g.milestones.map { m ->
        MilestoneView(id = m.title, title = m.title, done = m.completed)
    },
    taskIds = emptyList(),
)

private fun mapApproval(a: DataApproval): ApprovalView = ApprovalView(
    id = a.approvalId,
    action = a.action.ifBlank {
        a.requestedCapability.ifBlank { "approval" }
    },
    target = a.target,
    risk = parseRiskLevel(a.risk),
    reason = a.reason,
    requestedAt = epochSecToMs(a.createdAt),
    expiresAt = epochSecToMs(a.expiresAt),
    status = parseApprovalStatus(a.status),
    taskId = a.taskId.ifBlank { null },
    parameters = buildMap {
        if (a.requestedCapability.isNotBlank()) {
            put("capability", a.requestedCapability)
        }
    },
)

private fun mapDevice(d: DataDevice, currentDeviceId: String?): DeviceView =
    DeviceView(
        id = d.deviceId,
        name = d.deviceName.ifBlank { d.deviceId },
        platform = d.platform,
        online = d.connectionState.equals("online", ignoreCase = true) ||
            d.connectionState.equals("connected", ignoreCase = true),
        lastSeenAt = epochSecToMs(d.lastSeen),
        isCurrentDevice = currentDeviceId != null && d.deviceId == currentDeviceId,
        capabilities = listOfNotNull(d.trust.takeIf { it.isNotBlank() }),
    )

private fun mapArtifact(a: DataArtifact): ArtifactView = ArtifactView(
    id = a.artifactId,
    name = a.name.ifBlank { a.artifactId },
    kind = a.artifactType,
    sizeBytes = 0L,
    createdAt = parseAt(a.createdAt),
    taskId = null,
)

private fun mapActivity(a: DataActivityItem): ActivityItem {
    val detail = buildString {
        if (a.actor.isNotBlank()) append(a.actor)
        val ref = a.taskId.ifBlank { a.goalId }
        if (ref.isNotBlank()) {
            if (isNotEmpty()) append(" \u2022 ")
            append(ref)
        }
        if (a.source.isNotBlank()) {
            if (isNotEmpty()) append(" \u2022 ")
            append(a.source)
        }
    }
    return ActivityItem(
        id = a.eventId.ifBlank { a.seq.toString() },
        category = a.type,
        title = a.summary.ifBlank { a.type },
        detail = detail,
        at = parseAt(a.at),
    )
}

private fun mapEvent(e: DataEvent): RemoteEvent {
    val data = e.data
    fun str(key: String): String =
        data[key]?.jsonPrimitive?.contentOrNull.orEmpty()
    val refId = str("task_id")
        .ifBlank { str("approval_id") }
        .ifBlank { str("device_id") }
        .ifBlank { null }
    return RemoteEvent(
        id = e.eventId,
        category = e.category,
        title = str("title").ifBlank { e.category },
        detail = str("summary").ifBlank { str("detail") },
        refId = refId,
        at = (e.at * 1000).toLong(),
    )
}

/**
 * UI-facing repository backed by the real [DataRepository].
 */
class DataUiRepository(
    context: Context,
    private val real: DataRepository,
    private val secureStore: SecureTokenStore,
) : UiRepository {

    private val setupPrefs = context.applicationContext
        .getSharedPreferences(SETUP_PREFS, Context.MODE_PRIVATE)

    private fun requireBaseUrl(): String =
        secureStore.baseUrl
            ?: setupPrefs.getString(KEY_SETUP_URL, null)
            ?: throw IOException("not paired: no server URL stored")

    // -- setup / health ----------------------------------------------------

    override suspend fun checkHealth(baseUrl: String): Boolean =
        real.health(baseUrl).ok

    // -- pairing ------------------------------------------------------------

    override suspend fun requestPairing(
        host: String,
        port: Int,
        useTls: Boolean,
        deviceName: String,
    ): PairingInfo {
        val scheme = if (useTls) "https" else "http"
        val baseUrl = "$scheme://$host:$port"
        val deviceId = UUID.randomUUID().toString()
        val (pairingId, code) = real.pair(
            deviceId,
            deviceName.ifBlank { "Android device" },
            baseUrl,
        )
        return PairingInfo(pairingId = pairingId, code = code)
    }

    /**
     * Returns the bearer token once the owner approves, or null while the
     * pairing request is still pending approval. [DataRepository]
     * persists the session itself on success.
     */
    override suspend fun redeemPairing(
        pairingId: String,
        code: String,
    ): String? {
        val baseUrl = requireBaseUrl()
        try {
            val response = real.redeemAndStore(pairingId, code, baseUrl)
            return response.token.ifEmpty { null }
        } catch (e: AuthenticationException) {
            throw e
        } catch (e: ConnectionException) {
            // Transport failure is a real error, not "still pending".
            throw e
        } catch (e: ProtocolException) {
            val haystack = "${e.code} ${e.message}".lowercase()
            if ("pending" in haystack ||
                "approv" in haystack ||
                "await" in haystack
            ) {
                return null
            }
            throw e
        }
    }

    // -- agent ---------------------------------------------------------------

    override suspend fun agentStatus(): AgentStatus {
        val s: DataAgentStatus = real.getAgentStatus()
        return AgentStatus(
            state = s.status,
            currentTaskId = s.currentTask?.taskId,
            uptimeSec = 0L,
        )
    }

    override suspend fun sendTask(text: String): TaskView =
        mapTask(real.sendTask(text))

    // -- tasks ---------------------------------------------------------------

    override suspend fun listTasks(): List<TaskView> =
        real.listTasks().map(::mapTask)

    override suspend fun getTask(id: String): TaskView =
        mapTask(real.getTask(id))

    override suspend fun pauseTask(id: String): TaskView =
        mapTask(real.pauseTask(id))

    override suspend fun resumeTask(id: String): TaskView =
        mapTask(real.resumeTask(id))

    override suspend fun cancelTask(id: String): TaskView =
        mapTask(real.cancelTask(id))

    /**
     * The facade has no retry method, so `retry_task` is executed through
     * the data layer's own [CommandRepository] with a short-lived client.
     * `retry_task` is a known protocol command.
     */
    override suspend fun retryTask(id: String): TaskView {
        val baseUrl = requireBaseUrl()
        val token = secureStore.token
            ?: throw IOException("not paired: no session token stored")
        val client = ControlPlaneClient(baseUrl)
        try {
            val payload = buildJsonObject { put("task_id", id) }
            val result = CommandRepository(client).execute(
                token, "retry_task", payload
            )
            if (!result.isCompleted) {
                throw ProtocolException(
                    result.error.ifEmpty { "retry ${result.status}" },
                    result.errorCode.ifEmpty { "retry_${result.status}" },
                )
            }
            val taskObj = result.result["task"]?.jsonObject
                ?: JsonObject(emptyMap())
            return mapTask(DataTask.fromServerObject(taskObj))
        } finally {
            client.close()
        }
    }

    // -- goals ---------------------------------------------------------------

    override suspend fun listGoals(): List<GoalView> =
        real.listGoals().map(::mapGoal)

    // -- approvals -----------------------------------------------------------

    override suspend fun listApprovals(): List<ApprovalView> =
        real.listApprovals().map(::mapApproval)

    override suspend fun getApproval(id: String): ApprovalView =
        real.listApprovals()
            .firstOrNull { it.approvalId == id }
            ?.let(::mapApproval)
            ?: throw IOException("approval not found: $id")

    override suspend fun approve(id: String) {
        val result = real.approve(id)
        if (!result.isCompleted) {
            throw IOException(
                "approve ${result.status}: " +
                    result.error.ifEmpty {
                        "server did not complete the approval"
                    },
            )
        }
    }

    override suspend fun deny(id: String) {
        val result = real.deny(id)
        if (!result.isCompleted) {
            throw IOException(
                "deny ${result.status}: " +
                    result.error.ifEmpty {
                        "server did not complete the denial"
                    },
            )
        }
    }

    // -- devices -------------------------------------------------------------

    override suspend fun listDevices(): List<DeviceView> =
        real.listDevices().map { mapDevice(it, secureStore.deviceId) }

    override suspend fun revokeDevice(id: String) {
        val result = real.revokeDevice(id)
        if (!result.isCompleted) {
            throw IOException(
                "revoke ${result.status}: " +
                    result.error.ifEmpty {
                        "server did not complete the revocation"
                    },
            )
        }
    }

    // -- remote browser ------------------------------------------------------

    override suspend fun browserNavigate(url: String): BrowserStateView {
        val result: DataBrowserNav = real.browserNavigate(url)
        return BrowserStateView(
            url = result.url.ifBlank { url },
            title = result.title,
            loading = false,
        )
    }

    /**
     * Screenshot bytes for the remote browser.
     *
     * Prefers the inline base64 payload (requested from the server);
     * falls back to downloading the server-side reference path for
     * older servers. This is the only place raw image bytes are
     * produced.
     */
    override suspend fun browserScreenshot(): ByteArray =
        withContext(Dispatchers.IO) {
            val ref = real.browserScreenshot()
            if (ref.dataBase64.isNotBlank()) {
                return@withContext try {
                    android.util.Base64.decode(
                        ref.dataBase64, android.util.Base64.DEFAULT
                    )
                } catch (e: IllegalArgumentException) {
                    throw IOException("invalid screenshot data")
                }
            }
            val base = requireBaseUrl().trimEnd('/')
            val path = ref.path
            val url = when {
                path.startsWith("http://") || path.startsWith("https://") -> path
                path.startsWith("/") -> base + path
                path.isNotBlank() -> "$base/$path"
                else -> throw IOException("screenshot reference has no path")
            }
            val token = secureStore.token
                ?: throw IOException("not paired: no session token stored")
            val connection =
                (URL(url).openConnection() as HttpURLConnection).apply {
                    connectTimeout = 15_000
                    readTimeout = 15_000
                    setRequestProperty("Authorization", "Bearer $token")
                    setRequestProperty("Accept", "image/*")
                }
            try {
                val status = connection.responseCode
                if (status !in 200..299) {
                    throw IOException(
                        "screenshot download failed: HTTP $status"
                    )
                }
                val bytes = connection.inputStream.use { it.readBytes() }
                if (bytes.isEmpty()) throw IOException("empty screenshot")
                bytes
            } finally {
                connection.disconnect()
            }
        }

    override suspend fun browserState(): BrowserStateView {
        val state: DataBrowserState = real.browserState()
        return BrowserStateView(
            url = state.url,
            title = state.title,
            loading = false,
        )
    }

    // -- activity / artifacts ------------------------------------------------

    override suspend fun queryActivity(limit: Int): List<ActivityItem> =
        real.queryActivity(limit = limit).second.map(::mapActivity)

    override suspend fun listArtifacts(): List<ArtifactView> =
        real.listArtifacts().map(::mapArtifact)

    // -- connectors / credentials -------------------------------------------
    // The control plane exposes no listing endpoints for these; the UI
    // shows an empty state.

    override suspend fun listConnectors(): List<ConnectorView> = emptyList()

    override suspend fun listCredentialMeta(): List<CredentialMeta> = emptyList()

    // -- live event stream ---------------------------------------------------

    override fun observeEvents(): Flow<RemoteEvent> = callbackFlow {
        // Throws AuthenticationException when unpaired — the collector
        // treats that as a signal to stop, not to retry.
        val token = real.currentToken()
        val stream = real.eventStream()
        try {
            stream.streamEvents(token, 0) { event ->
                trySend(mapEvent(event))
            }
        } finally {
            stream.stop()
        }
        // The server closed the stream: end the flow so the collector
        // can reconnect with backoff. Cancellation still unwinds through
        // the finally block above.
        close()
    }
}

/**
 * UI-facing token store backed by the encrypted [SecureTokenStore].
 *
 * The secure store only records the server URL once a session is saved,
 * so the pre-pairing URL (entered on the setup screen) is kept in plain
 * app preferences until [DataRepository.redeemAndStore] persists the
 * real session. The URL itself is not a secret.
 */
class DataUiTokenStore(
    context: Context,
    private val real: DataRepository,
    private val secureStore: SecureTokenStore,
) : UiTokenStore {

    private val prefs = context.applicationContext
        .getSharedPreferences(SETUP_PREFS, Context.MODE_PRIVATE)

    override fun getBaseUrl(): String =
        secureStore.baseUrl ?: prefs.getString(KEY_SETUP_URL, null) ?: ""

    override fun setBaseUrl(url: String) {
        prefs.edit().putString(KEY_SETUP_URL, url).apply()
    }

    override fun getToken(): String? = secureStore.token

    override fun setToken(token: String?) {
        if (token.isNullOrEmpty()) {
            // A null token means there is no session at all.
            real.unpair()
        } else {
            secureStore.updateToken(token)
        }
    }

    override fun clear() {
        real.unpair()
        prefs.edit().remove(KEY_SETUP_URL).apply()
    }
}

package com.afnan.ai.ui

import android.content.Context
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.afnan.ai.AfnanApp
import com.afnan.ai.notifications.Notifier
import com.afnan.ai.voice.VoiceInput
import com.afnan.ai.voice.VoiceOutput
import kotlinx.coroutines.flow.Flow
import java.util.UUID

// ---------------------------------------------------------------------------
// Models. Named 1:1 with the data-layer models so the real AfnanRepository
// can back this UI without translation layers.
// ---------------------------------------------------------------------------

enum class TaskStatus {
    PENDING, RUNNING, PAUSED, COMPLETED, FAILED, CANCELLED;

    val isTerminal: Boolean
        get() = this == COMPLETED || this == FAILED || this == CANCELLED
}

data class TaskView(
    val id: String,
    val title: String,
    val status: TaskStatus,
    /** 0f..1f */
    val progress: Float,
    val createdAt: Long,
    val updatedAt: Long,
    val resultSummary: String?,
    val error: String?,
)

data class MilestoneView(
    val id: String,
    val title: String,
    val done: Boolean,
)

data class GoalView(
    val id: String,
    val title: String,
    val description: String,
    val progress: Float,
    val status: String,
    val milestones: List<MilestoneView>,
    val taskIds: List<String>,
)

enum class RiskLevel { LOW, MEDIUM, HIGH, CRITICAL }
enum class ApprovalStatus { PENDING, APPROVED, DENIED, EXPIRED }

data class ApprovalView(
    val id: String,
    /** e.g. "browser.navigate" */
    val action: String,
    /** human-readable target summary */
    val target: String,
    val risk: RiskLevel,
    val reason: String,
    val requestedAt: Long,
    val expiresAt: Long,
    val status: ApprovalStatus,
    val taskId: String?,
    /** safe structured context; never secrets */
    val parameters: Map<String, String>,
)

data class DeviceView(
    val id: String,
    val name: String,
    val platform: String,
    val online: Boolean,
    val lastSeenAt: Long,
    val isCurrentDevice: Boolean,
    val capabilities: List<String>,
)

data class ArtifactView(
    val id: String,
    val name: String,
    val kind: String,
    val sizeBytes: Long,
    val createdAt: Long,
    val taskId: String?,
)

data class ActivityItem(
    val id: String,
    /** e.g. "task.completed", "approval.requested" */
    val category: String,
    val title: String,
    val detail: String,
    val at: Long,
)

data class AgentStatus(
    val state: String,
    val currentTaskId: String?,
    val uptimeSec: Long,
)

data class BrowserStateView(
    val url: String,
    val title: String,
    val loading: Boolean,
)

data class PairingInfo(
    val pairingId: String,
    val code: String,
)

data class ConnectorView(
    val id: String,
    val name: String,
    val connected: Boolean,
    val status: String,
)

/** Credential metadata only. Values are never exposed to the UI. */
data class CredentialMeta(
    val id: String,
    val label: String,
    val kind: String,
    val createdAt: Long,
)

/** One event pushed from the control plane to the sync service / UI. */
data class RemoteEvent(
    val id: String,
    val category: String,
    val title: String,
    val detail: String,
    /** task / approval / device id the event refers to, if any */
    val refId: String?,
    val at: Long,
)

// ---------------------------------------------------------------------------
// Chat UI model (presentation-level; built from TaskView/ApprovalView).
// ---------------------------------------------------------------------------

enum class ChatRole { USER, ASSISTANT, PROGRESS, APPROVAL, ERROR, SUMMARY }

data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val role: ChatRole,
    val text: String,
    val approval: ApprovalView? = null,
    val at: Long = System.currentTimeMillis(),
)

// ---------------------------------------------------------------------------
// Narrow contracts the UI depends on. Implemented by the data layer.
// ---------------------------------------------------------------------------

/** Secure storage for the control-plane base URL and bearer token. */
interface TokenStore {
    fun getBaseUrl(): String
    fun setBaseUrl(url: String)
    fun getToken(): String?
    fun setToken(token: String?)
    fun clear()
}

/**
 * Facade over the Afnan control plane. Every function performs a real
 * network call against the paired Afnan runtime; nothing is mocked.
 */
interface AfnanRepository {
    // -- setup / health ----------------------------------------------------
    suspend fun checkHealth(baseUrl: String): Boolean

    // -- pairing ------------------------------------------------------------
    suspend fun requestPairing(
        host: String,
        port: Int,
        useTls: Boolean,
        deviceName: String,
    ): PairingInfo

    /**
     * Returns the bearer token once the owner approves, or null while the
     * pairing request is still pending approval.
     */
    suspend fun redeemPairing(pairingId: String, code: String): String?

    // -- agent ---------------------------------------------------------------
    suspend fun agentStatus(): AgentStatus
    suspend fun sendTask(text: String): TaskView

    // -- tasks ---------------------------------------------------------------
    suspend fun listTasks(): List<TaskView>
    suspend fun getTask(id: String): TaskView
    suspend fun pauseTask(id: String): TaskView
    suspend fun resumeTask(id: String): TaskView
    suspend fun cancelTask(id: String): TaskView
    suspend fun retryTask(id: String): TaskView

    // -- goals ---------------------------------------------------------------
    suspend fun listGoals(): List<GoalView>

    // -- approvals -----------------------------------------------------------
    suspend fun listApprovals(): List<ApprovalView>
    suspend fun getApproval(id: String): ApprovalView
    suspend fun approve(id: String)
    suspend fun deny(id: String)

    // -- devices -------------------------------------------------------------
    suspend fun listDevices(): List<DeviceView>
    suspend fun revokeDevice(id: String)

    // -- remote browser (runs on the paired PC, surfaced here) --------------
    suspend fun browserNavigate(url: String): BrowserStateView
    suspend fun browserScreenshot(): ByteArray
    suspend fun browserState(): BrowserStateView

    // -- activity / artifacts ------------------------------------------------
    suspend fun queryActivity(limit: Int = 50): List<ActivityItem>
    suspend fun listArtifacts(): List<ArtifactView>

    // -- connectors / credentials (metadata only) ---------------------------
    suspend fun listConnectors(): List<ConnectorView>
    suspend fun listCredentialMeta(): List<CredentialMeta>

    // -- live event stream (SSE/WebSocket managed by the data layer) -------
    fun observeEvents(): Flow<RemoteEvent>
}

/** Everything the UI needs, assembled once in [AfnanApp]. */
class AppServices(
    val context: Context,
    val tokenStore: TokenStore,
    val repository: AfnanRepository,
    val voiceInput: VoiceInput,
    val voiceOutput: VoiceOutput,
    val notifier: Notifier,
)

// ---------------------------------------------------------------------------
// Small composition helpers.
// ---------------------------------------------------------------------------

@Composable
fun rememberServices(): AppServices {
    val context = LocalContext.current
    return remember {
        (context.applicationContext as AfnanApp).services
    }
}

fun <T : ViewModel> viewModelFactory(create: () -> T): ViewModelProvider.Factory =
    object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <M : ViewModel> create(modelClass: Class<M>): M =
            create() as M
    }

fun formatBytes(bytes: Long): String = when {
    bytes < 1024 -> "$bytes B"
    bytes < 1024 * 1024 -> "${bytes / 1024} KB"
    else -> "${bytes / (1024 * 1024)} MB"
}

fun formatTimeAgo(at: Long, now: Long = System.currentTimeMillis()): String {
    val s = ((now - at) / 1000).coerceAtLeast(0)
    return when {
        s < 60 -> "just now"
        s < 3600 -> "${s / 60}m ago"
        s < 86400 -> "${s / 3600}h ago"
        else -> "${s / 86400}d ago"
    }
}

fun formatCountdown(expiresAt: Long, now: Long = System.currentTimeMillis()): String {
    val s = ((expiresAt - now) / 1000).coerceAtLeast(0)
    return "%02d:%02d".format(s / 60, s % 60)
}

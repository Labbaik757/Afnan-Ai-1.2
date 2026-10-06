package com.afnan.ai.worker

import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import com.afnan.ai.AfnanApp
import com.afnan.ai.data.AuthenticationException
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.min

/**
 * Foreground service that holds the control-plane event stream open while
 * the app is backgrounded and turns important remote events into
 * notifications (approvals, task completion/failure, device events).
 *
 * Reconnects with exponential backoff and never busy-loops. The actual
 * long-running work stays on the paired Afnan runtime; this service only
 * listens.
 */
class AfnanSyncService : Service() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onCreate() {
        super.onCreate()
        val app = application as AfnanApp
        val notification = app.services.notifier.syncNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
            )
        } else {
            @Suppress("DEPRECATION")
            startForeground(NOTIFICATION_ID, notification)
        }
        scope.launch { runEventLoop(app) }
    }

    private suspend fun runEventLoop(app: AfnanApp) {
        var backoffMs = INITIAL_BACKOFF_MS
        while (scope.isActive) {
            try {
                app.services.repository.observeEvents().collect { event ->
                    backoffMs = INITIAL_BACKOFF_MS
                    handleEvent(app, event.category, event.title, event.detail, event.refId)
                }
                // Flow completed without error (server closed stream): reconnect.
                delay(backoffMs)
            } catch (e: CancellationException) {
                throw e
            } catch (e: AuthenticationException) {
                // The token is dead — reconnecting will never heal it.
                // Stop; the UI routes back to pairing on next launch.
                stopSelf()
                break
            } catch (e: Exception) {
                if (!scope.isActive) break
                delay(backoffMs)
                backoffMs = min(backoffMs * 2, MAX_BACKOFF_MS)
            }
        }
    }

    private fun handleEvent(
        app: AfnanApp,
        category: String,
        title: String,
        detail: String,
        refId: String?,
    ) {
        val notifier = app.services.notifier
        when {
            category.startsWith("approval.requested") && refId != null ->
                notifier.showApprovalRequested(refId, title, detail)

            category.startsWith("task.completed") && refId != null ->
                notifier.showTaskEvent(refId, title, detail)

            category.startsWith("task.failed") && refId != null ->
                notifier.showTaskEvent(refId, title, detail)

            category.startsWith("security.") ->
                notifier.showTaskEvent(refId ?: "security", title, detail)
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int =
        START_STICKY

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    companion object {
        /** Start (or keep) the foreground sync service. Safe to call repeatedly. */
        fun start(context: Context) {
            val intent = Intent(context, AfnanSyncService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        private const val NOTIFICATION_ID = 1
        private const val INITIAL_BACKOFF_MS = 2_000L
        private const val MAX_BACKOFF_MS = 60_000L
    }
}

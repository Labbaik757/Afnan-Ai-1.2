package com.afnan.ai.notifications

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import com.afnan.ai.MainActivity
import com.afnan.ai.R

/**
 * Posts Afnan notifications. Tapping a notification deep-links into the
 * relevant screen (afnan://approval/{id}, afnan://task/{id}). Approve/Deny
 * quick actions are delivered to [com.afnan.ai.AfnanApp]'s dynamically
 * registered receiver — no secrets ever appear in notification content.
 */
class Notifier(private val context: Context) {

    private val manager: NotificationManager =
        context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

    fun showApprovalRequested(approvalId: String, title: String, detail: String) {
        val open = deepLinkIntent("afnan://approval/$approvalId", REQ_OPEN_APPROVAL + approvalId.hashCode())
        val approve = actionIntent(approvalId, true)
        val deny = actionIntent(approvalId, false)
        val notification = NotificationCompat.Builder(context, CHANNEL_APPROVALS)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle(title)
            .setContentText(detail)
            .setStyle(NotificationCompat.BigTextStyle().bigText(detail))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setAutoCancel(true)
            .setContentIntent(open)
            .addAction(
                android.R.drawable.ic_menu_send,
                context.getString(R.string.notif_approve),
                approve,
            )
            .addAction(
                android.R.drawable.ic_menu_close_clear_cancel,
                context.getString(R.string.notif_deny),
                deny,
            )
            .build()
        manager.notify(NOTIF_APPROVAL_BASE + approvalId.hashCode(), notification)
    }

    fun cancelApproval(approvalId: String) {
        manager.cancel(NOTIF_APPROVAL_BASE + approvalId.hashCode())
    }

    fun showTaskEvent(taskId: String, title: String, detail: String) {
        val open = deepLinkIntent("afnan://task/$taskId", REQ_OPEN_TASK + taskId.hashCode())
        val notification = NotificationCompat.Builder(context, CHANNEL_TASKS)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle(title)
            .setContentText(detail)
            .setStyle(NotificationCompat.BigTextStyle().bigText(detail))
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setAutoCancel(true)
            .setContentIntent(open)
            .build()
        manager.notify(NOTIF_TASK_BASE + taskId.hashCode(), notification)
    }

    /** Low-key ongoing notification for the foreground sync service. */
    fun syncNotification(): android.app.Notification =
        NotificationCompat.Builder(context, CHANNEL_TASKS)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle(context.getString(R.string.notif_sync_title))
            .setContentText(context.getString(R.string.notif_sync_text))
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_MIN)
            .build()

    private fun deepLinkIntent(uri: String, requestCode: Int): PendingIntent {
        val intent = Intent(Intent.ACTION_VIEW, android.net.Uri.parse(uri)).apply {
            setClass(context, MainActivity::class.java)
        }
        return PendingIntent.getActivity(
            context,
            requestCode,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
    }

    private fun actionIntent(approvalId: String, approve: Boolean): PendingIntent {
        val intent = Intent(ACTION_APPROVAL_DECISION).apply {
            setPackage(context.packageName)
            putExtra(EXTRA_APPROVAL_ID, approvalId)
            putExtra(EXTRA_APPROVE, approve)
        }
        return PendingIntent.getBroadcast(
            context,
            (if (approve) REQ_APPROVE else REQ_DENY) + approvalId.hashCode(),
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
    }

    companion object {
        const val CHANNEL_APPROVALS = "afnan_approvals"
        const val CHANNEL_TASKS = "afnan_tasks"

        const val ACTION_APPROVAL_DECISION = "com.afnan.ai.APPROVAL_DECISION"
        const val EXTRA_APPROVAL_ID = "approval_id"
        const val EXTRA_APPROVE = "approve"

        private const val NOTIF_APPROVAL_BASE = 1000
        private const val NOTIF_TASK_BASE = 2000
        private const val REQ_OPEN_APPROVAL = 3000
        private const val REQ_OPEN_TASK = 4000
        private const val REQ_APPROVE = 5000
        private const val REQ_DENY = 6000

        fun createChannels(context: Context) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
            val manager =
                context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(
                NotificationChannel(
                    CHANNEL_APPROVALS,
                    context.getString(R.string.channel_approvals),
                    NotificationManager.IMPORTANCE_HIGH,
                ).apply {
                    description = context.getString(R.string.channel_approvals_desc)
                },
            )
            manager.createNotificationChannel(
                NotificationChannel(
                    CHANNEL_TASKS,
                    context.getString(R.string.channel_tasks),
                    NotificationManager.IMPORTANCE_DEFAULT,
                ).apply {
                    description = context.getString(R.string.channel_tasks_desc)
                },
            )
        }
    }
}

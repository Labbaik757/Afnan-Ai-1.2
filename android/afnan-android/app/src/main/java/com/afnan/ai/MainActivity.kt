package com.afnan.ai

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.FragmentActivity
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.afnan.ai.ui.debug.CrashReportScreen
import com.afnan.ai.ui.lock.AppLockScreen
import com.afnan.ai.ui.navigation.AfnanScaffold
import com.afnan.ai.ui.navigation.Routes
import com.afnan.ai.ui.rememberServices
import com.afnan.ai.ui.theme.AfnanTheme

/**
 * Single-activity host. Handles afnan:// deep links (approval/{id},
 * task/{id}), the POST_NOTIFICATIONS runtime permission on Android 13+,
 * and the optional biometric app lock.
 */
class MainActivity : FragmentActivity() {

    private companion object {
        /** Must stay within the lower 16 bits (see note above). */
        const val REQUEST_NOTIFICATION_PERMISSION = 1001
    }

    private var pendingDeepLink by mutableStateOf<String?>(null)

    // NOTE: Do NOT use registerForActivityResult() for permissions here.
    // The ActivityResultRegistry generates request codes larger than 16 bits,
    // which this project's old androidx.fragment FragmentActivity rejects
    // with "Can only use lower 16 bits for requestCode" and crashes on launch.
    // The classic API with a small request code is safe.

    override fun onCreate(savedInstanceState: Bundle?) {
        // Swap the splash theme for the real app theme before drawing.
        setTheme(R.style.Theme_AfnanAI)
        super.onCreate(savedInstanceState)

        // If the last run crashed, show the saved stack trace instead
        // of starting normally — the user screenshots it for diagnosis.
        val crashReport = AfnanApp.readCrashLog(this)
        if (crashReport != null) {
            setContent {
                AfnanTheme(app = application as AfnanApp) {
                    CrashReportScreen(
                        report = crashReport,
                        onDismiss = {
                            AfnanApp.clearCrashLog(this)
                            recreate()
                        },
                    )
                }
            }
            return
        }

        handleIntent(intent)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(
                this, Manifest.permission.POST_NOTIFICATIONS,
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            // Fire-and-forget: notifications degrade gracefully without it.
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                REQUEST_NOTIFICATION_PERMISSION,
            )
        }

        setContent {
            val app = application as AfnanApp
            val lockEnabled by app.appLockEnabled.collectAsStateWithLifecycle()
            var unlocked by remember(lockEnabled) { mutableStateOf(!lockEnabled) }
            AfnanTheme(app = app) {
                if (!unlocked) {
                    AppLockScreen(onUnlocked = { unlocked = true })
                } else {
                    val services = rememberServices()
                    AfnanScaffold(
                        services = services,
                        deepLink = pendingDeepLink,
                        onDeepLinkConsumed = { pendingDeepLink = null },
                    )
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: Intent?) {
        val data: Uri = intent?.data ?: return
        if (data.scheme != "afnan") return
        val id = data.lastPathSegment ?: return
        pendingDeepLink = when (data.host) {
            "approval" -> Routes.approvalDetail(id)
            "task" -> Routes.taskDetail(id)
            else -> null
        }
    }
}

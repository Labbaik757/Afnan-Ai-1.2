package com.afnan.ai

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import androidx.fragment.app.FragmentActivity
import androidx.lifecycle.compose.collectAsStateWithLifecycle
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

    private var pendingDeepLink by mutableStateOf<String?>(null)

    private val notificationPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) {
            // Result is informational; notifications degrade gracefully.
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        // Swap the splash theme for the real app theme before drawing.
        setTheme(R.style.Theme_AfnanAI)
        super.onCreate(savedInstanceState)
        handleIntent(intent)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(
                this, Manifest.permission.POST_NOTIFICATIONS,
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
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

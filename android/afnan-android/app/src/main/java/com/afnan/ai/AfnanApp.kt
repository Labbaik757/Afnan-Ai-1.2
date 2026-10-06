package com.afnan.ai

import android.app.Application
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import com.afnan.ai.data.AfnanRepository as DataAfnanRepository
import com.afnan.ai.notifications.Notifier
import com.afnan.ai.security.TokenStore as SecureTokenStore
import com.afnan.ai.ui.AppServices
import com.afnan.ai.ui.theme.ThemeMode
import com.afnan.ai.voice.AndroidVoiceInput
import com.afnan.ai.voice.AndroidVoiceOutput
import com.afnan.ai.wiring.DataUiRepository
import com.afnan.ai.wiring.DataUiTokenStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Application wiring for Afnan AI.
 *
 * The UI depends only on the narrow interfaces in ui/UiContracts.kt.
 * The concrete data-layer components (data/, security/) are bound to
 * those interfaces by the adapters in wiring/DataWiring.kt, assembled
 * once here in [services].
 */
class AfnanApp : Application() {

    private val ioScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val secureStore by lazy { SecureTokenStore(this) }
    private val dataRepository by lazy { DataAfnanRepository(secureStore) }

    val services: AppServices by lazy {
        AppServices(
            context = this,
            tokenStore = DataUiTokenStore(this, dataRepository, secureStore),
            repository = DataUiRepository(this, dataRepository, secureStore),
            voiceInput = AndroidVoiceInput(this),
            voiceOutput = AndroidVoiceOutput(this),
            notifier = Notifier(this),
        )
    }

    private val _themeMode = MutableStateFlow(ThemeMode.SYSTEM)
    val themeMode = _themeMode.asStateFlow()

    private val _appLockEnabled = MutableStateFlow(false)
    val appLockEnabled = _appLockEnabled.asStateFlow()

    private val _voiceResponsesEnabled = MutableStateFlow(false)
    val voiceResponsesEnabled = _voiceResponsesEnabled.asStateFlow()

    override fun onCreate() {
        super.onCreate()
        installCrashReporter()
        Notifier.createChannels(this)
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        _themeMode.value = runCatching {
            ThemeMode.valueOf(prefs.getString(KEY_THEME, ThemeMode.SYSTEM.name)!!)
        }.getOrDefault(ThemeMode.SYSTEM)
        _appLockEnabled.value = prefs.getBoolean(KEY_APP_LOCK, false)
        _voiceResponsesEnabled.value = prefs.getBoolean(KEY_VOICE_RESPONSES, false)

        // Notification quick actions (Approve / Deny) without a
        // manifest-declared receiver: registered while the process lives.
        val filter = IntentFilter(Notifier.ACTION_APPROVAL_DECISION)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(
                approvalActionReceiver, filter, RECEIVER_NOT_EXPORTED,
            )
        } else {
            @Suppress("UnspecifiedRegisterReceiverFlag")
            registerReceiver(approvalActionReceiver, filter)
        }
    }

    fun setThemeMode(mode: ThemeMode) {
        _themeMode.value = mode
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
            .putString(KEY_THEME, mode.name).apply()
    }

    fun setAppLockEnabled(enabled: Boolean) {
        _appLockEnabled.value = enabled
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
            .putBoolean(KEY_APP_LOCK, enabled).apply()
    }

    fun setVoiceResponsesEnabled(enabled: Boolean) {
        _voiceResponsesEnabled.value = enabled
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
            .putBoolean(KEY_VOICE_RESPONSES, enabled).apply()
    }

    /** Signs out locally: drops token + server URL. Never touches server data. */
    fun signOutLocal() {
        runCatching { services.tokenStore.clear() }
    }

    private val approvalActionReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action != Notifier.ACTION_APPROVAL_DECISION) return
            val id = intent.getStringExtra(Notifier.EXTRA_APPROVAL_ID) ?: return
            val approve = intent.getBooleanExtra(Notifier.EXTRA_APPROVE, false)
            ioScope.launch {
                try {
                    val repo = services.repository
                    if (approve) repo.approve(id) else repo.deny(id)
                    services.notifier.cancelApproval(id)
                } catch (_: Exception) {
                    // The approval screen remains the source of truth;
                    // the user can retry there.
                }
            }
        }
    }

    companion object {
        private const val PREFS = "afnan_ui_prefs"
        private const val KEY_THEME = "theme_mode"
        private const val KEY_APP_LOCK = "app_lock"
        private const val KEY_VOICE_RESPONSES = "voice_responses"

        /** File holding the last uncaught exception's stack trace. */
        const val CRASH_LOG = "afnan_crash.log"

        fun readCrashLog(context: Context): String? {
            val file = java.io.File(context.filesDir, CRASH_LOG)
            if (!file.exists() || file.length() == 0L) return null
            return runCatching { file.readText() }.getOrNull()
        }

        fun clearCrashLog(context: Context) {
            runCatching {
                java.io.File(context.filesDir, CRASH_LOG).delete()
            }
        }
    }

    /**
     * Writes any uncaught exception's stack trace to [CRASH_LOG]
     * before the process dies, so the next launch can show what
     * actually crashed instead of a generic "keeps stopping".
     */
    private fun installCrashReporter() {
        val defaultHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            runCatching {
                val trace = StringBuilder()
                    .append(java.util.Date().toString())
                    .append('\n')
                    .append(throwable.toString())
                    .append('\n')
                var cause: Throwable? = throwable
                while (cause != null) {
                    for (element in cause.stackTrace.take(40)) {
                        trace.append("    at ").append(element).append('\n')
                    }
                    cause = cause.cause
                    if (cause != null) trace.append("Caused by: ").append(cause).append('\n')
                }
                java.io.File(filesDir, CRASH_LOG).writeText(trace.toString())
            }
            defaultHandler?.uncaughtException(thread, throwable)
            if (defaultHandler == null) {
                android.os.Process.killProcess(android.os.Process.myPid())
            }
        }
    }
}

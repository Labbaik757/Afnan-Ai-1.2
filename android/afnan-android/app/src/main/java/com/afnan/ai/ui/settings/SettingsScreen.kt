package com.afnan.ai.ui.settings

import android.content.Intent
import android.provider.Settings
import androidx.biometric.BiometricManager
import androidx.biometric.BiometricPrompt
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.DarkMode
import androidx.compose.material.icons.filled.Fingerprint
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Key
import androidx.compose.material.icons.filled.Link
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material.icons.filled.RecordVoiceOver
import androidx.compose.material.icons.filled.Security
import androidx.compose.material.icons.filled.Smartphone
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.afnan.ai.AfnanApp
import com.afnan.ai.MainActivity
import com.afnan.ai.ui.AppServices
import com.afnan.ai.ui.navigation.Routes
import com.afnan.ai.ui.theme.ThemeMode

private data class SettingEntry(
    val route: String?,
    val label: String,
    val hint: String,
    val icon: ImageVector,
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    services: AppServices,
    onNavigate: (String) -> Unit,
) {
    val context = LocalContext.current
    val app = context.applicationContext as AfnanApp
    val themeMode by app.themeMode.collectAsStateWithLifecycle()
    val appLock by app.appLockEnabled.collectAsStateWithLifecycle()
    val voiceResponses by app.voiceResponsesEnabled.collectAsStateWithLifecycle()

    var showThemeDialog by remember { mutableStateOf(false) }
    var showLockConfirm by remember { mutableStateOf(false) }

    val entries = listOf(
        SettingEntry(Routes.CONNECTORS, "Connectors", "Connected services and their status", Icons.Filled.Link),
        SettingEntry(Routes.CREDENTIALS, "Secure credentials", "Saved logins (metadata only)", Icons.Filled.Key),
        SettingEntry(Routes.PERMISSIONS, "Permissions", "Device and Afnan permissions", Icons.Filled.Security),
        SettingEntry(Routes.DEVICES, "Devices", "Paired devices and sessions", Icons.Filled.Smartphone),
        SettingEntry(null, "Notifications", "System notification settings", Icons.Filled.Notifications),
        SettingEntry(null, "Appearance", "Light / dark theme", Icons.Filled.DarkMode),
        SettingEntry(null, "App lock", "Biometric lock", Icons.Filled.Fingerprint),
        SettingEntry(null, "Voice responses", "Speak summaries in Urdu", Icons.Filled.RecordVoiceOver),
        SettingEntry(Routes.DATA_CONTROLS, "Data controls", "Local data and sign out", Icons.Filled.Storage),
        SettingEntry(Routes.ABOUT, "About", "Version and legal", Icons.Filled.Info),
    )

    Scaffold(
        topBar = { TopAppBar(title = { Text("Settings") }) },
    ) { padding ->
        LazyColumn(
            Modifier
                .fillMaxSize()
                .padding(padding),
        ) {
            items(entries) { entry ->
                SettingRow(
                    entry = entry,
                    trailing = {
                        when (entry.label) {
                            "Appearance" -> Text(
                                when (themeMode) {
                                    ThemeMode.LIGHT -> "Light"
                                    ThemeMode.DARK -> "Dark"
                                    ThemeMode.SYSTEM -> "System"
                                },
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )

                            "App lock" -> Switch(
                                checked = appLock,
                                onCheckedChange = { enabled ->
                                    if (enabled) showLockConfirm = true
                                    else app.setAppLockEnabled(false)
                                },
                            )

                            "Voice responses" -> Switch(
                                checked = voiceResponses,
                                onCheckedChange = app::setVoiceResponsesEnabled,
                            )

                            else -> Icon(
                                Icons.Filled.ChevronRight,
                                contentDescription = "Open",
                                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    },
                    onClick = {
                        when (entry.label) {
                            "Appearance" -> showThemeDialog = true
                            "Notifications" -> {
                                val intent = Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS).apply {
                                    putExtra(Settings.EXTRA_APP_PACKAGE, context.packageName)
                                }
                                context.startActivity(intent)
                            }

                            else -> entry.route?.let(onNavigate)
                        }
                    },
                )
                HorizontalDivider()
            }
        }
    }

    if (showThemeDialog) {
        AlertDialog(
            onDismissRequest = { showThemeDialog = false },
            title = { Text("Appearance") },
            text = {
                Column {
                    ThemeMode.entries.forEach { mode ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    app.setThemeMode(mode)
                                    showThemeDialog = false
                                }
                                .padding(vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            RadioButton(
                                selected = themeMode == mode,
                                onClick = {
                                    app.setThemeMode(mode)
                                    showThemeDialog = false
                                },
                            )
                            Spacer(Modifier.width(8.dp))
                            Text(
                                when (mode) {
                                    ThemeMode.SYSTEM -> "System default"
                                    ThemeMode.LIGHT -> "Light"
                                    ThemeMode.DARK -> "Dark"
                                },
                            )
                        }
                    }
                }
            },
            confirmButton = {},
            dismissButton = {
                TextButton(onClick = { showThemeDialog = false }) { Text("Close") }
            },
        )
    }

    if (showLockConfirm) {
        ConfirmBiometricThen(
            onAuthenticated = {
                app.setAppLockEnabled(true)
                showLockConfirm = false
            },
            onDismiss = { showLockConfirm = false },
        )
    }
}

/** Runs a biometric check once, then calls back. Used to confirm enabling app lock. */
@Composable
private fun ConfirmBiometricThen(
    onAuthenticated: () -> Unit,
    onDismiss: () -> Unit,
) {
    val context = LocalContext.current
    val activity = context as MainActivity
    var attempted by remember { mutableStateOf(false) }

    if (!attempted) {
        attempted = true
        val executor = ContextCompat.getMainExecutor(context)
        val prompt = BiometricPrompt(
            activity,
            executor,
            object : BiometricPrompt.AuthenticationCallback() {
                override fun onAuthenticationSucceeded(
                    result: BiometricPrompt.AuthenticationResult,
                ) {
                    onAuthenticated()
                }

                override fun onAuthenticationError(errorCode: Int, errString: CharSequence) {
                    onDismiss()
                }
            },
        )
        val canAuth = BiometricManager.from(context).canAuthenticate(
            BiometricManager.Authenticators.BIOMETRIC_STRONG or
                BiometricManager.Authenticators.DEVICE_CREDENTIAL,
        )
        if (canAuth == BiometricManager.BIOMETRIC_SUCCESS) {
            prompt.authenticate(
                BiometricPrompt.PromptInfo.Builder()
                    .setTitle("Enable app lock")
                    .setSubtitle("Confirm it's you")
                    .setAllowedAuthenticators(
                        BiometricManager.Authenticators.BIOMETRIC_STRONG or
                            BiometricManager.Authenticators.DEVICE_CREDENTIAL,
                    )
                    .build(),
            )
        } else {
            onDismiss()
        }
    }
}

@Composable
private fun SettingRow(
    entry: SettingEntry,
    trailing: @Composable () -> Unit,
    onClick: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(horizontal = 20.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            entry.icon,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary,
        )
        Spacer(Modifier.width(16.dp))
        Column(Modifier.weight(1f)) {
            Text(entry.label, style = MaterialTheme.typography.titleSmall)
            Text(
                entry.hint,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        trailing()
    }
}

@Composable
fun SettingsSectionTitle(text: String) {
    Text(
        text,
        style = MaterialTheme.typography.titleSmall,
        color = MaterialTheme.colorScheme.primary,
        modifier = Modifier.padding(horizontal = 20.dp, vertical = 8.dp),
    )
}

@Composable
fun SettingsHint(text: String) {
    Text(
        text,
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.padding(horizontal = 20.dp, vertical = 8.dp),
    )
    Spacer(Modifier.height(8.dp))
}

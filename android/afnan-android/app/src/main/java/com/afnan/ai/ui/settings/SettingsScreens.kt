package com.afnan.ai.ui.settings

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Link
import androidx.compose.material.icons.filled.LinkOff
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.afnan.ai.AfnanApp
import com.afnan.ai.ui.AppServices
import com.afnan.ai.ui.ConnectorView
import com.afnan.ai.ui.CredentialMeta
import com.afnan.ai.ui.tasks.EmptyState
import com.afnan.ai.ui.viewModelFactory
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// ---------------------------------------------------------------------------
// Connectors
// ---------------------------------------------------------------------------

class ConnectorsViewModel(private val services: AppServices) : ViewModel() {
    var connectors by mutableStateOf<List<ConnectorView>>(emptyList())
        private set
    var loading by mutableStateOf(false)
        private set
    var error by mutableStateOf<String?>(null)
        private set

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            loading = true
            error = null
            try {
                connectors = services.repository.listConnectors()
            } catch (e: Exception) {
                error = e.message ?: "Couldn't load connectors"
            } finally {
                loading = false
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConnectorsScreen(
    services: AppServices,
    vm: ConnectorsViewModel = viewModel(
        factory = viewModelFactory { ConnectorsViewModel(services) },
    ),
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Connectors") },
                actions = {
                    IconButton(onClick = vm::refresh, enabled = !vm.loading) {
                        Icon(Icons.Filled.Refresh, contentDescription = "Refresh connectors")
                    }
                },
            )
        },
    ) { padding ->
        Column(
            Modifier
                .fillMaxSize()
                .padding(padding),
        ) {
            SettingsHint(
                "Services Afnan can work with. Connecting a service never " +
                    "authorizes every action — sensitive actions still ask for approval.",
            )
            if (vm.loading && vm.connectors.isEmpty()) {
                Row(
                    Modifier.fillMaxSize(),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically,
                ) { CircularProgressIndicator() }
                return@Column
            }
            vm.error?.let {
                Text(
                    it,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(horizontal = 20.dp),
                )
            }
            if (vm.connectors.isEmpty() && !vm.loading) {
                EmptyState("No connectors", "Nothing connected yet.")
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(vm.connectors, key = { it.id }) { connector ->
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 16.dp),
                        ) {
                            Row(
                                Modifier.padding(16.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Icon(
                                    if (connector.connected) Icons.Filled.Link
                                    else Icons.Filled.LinkOff,
                                    contentDescription = if (connector.connected) "Connected" else "Not connected",
                                    tint = if (connector.connected)
                                        MaterialTheme.colorScheme.primary
                                    else MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                                Spacer(Modifier.width(12.dp))
                                Column(Modifier.weight(1f)) {
                                    Text(
                                        connector.name,
                                        fontWeight = FontWeight.SemiBold,
                                    )
                                    Text(
                                        connector.status,
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Credentials (metadata only — values are never shown)
// ---------------------------------------------------------------------------

class CredentialsViewModel(private val services: AppServices) : ViewModel() {
    var credentials by mutableStateOf<List<CredentialMeta>>(emptyList())
        private set
    var loading by mutableStateOf(false)
        private set
    var error by mutableStateOf<String?>(null)
        private set

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            loading = true
            error = null
            try {
                credentials = services.repository.listCredentialMeta()
            } catch (e: Exception) {
                error = e.message ?: "Couldn't load credentials"
            } finally {
                loading = false
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CredentialsScreen(
    services: AppServices,
    vm: CredentialsViewModel = viewModel(
        factory = viewModelFactory { CredentialsViewModel(services) },
    ),
) {
    val dateFormat = remember {
        SimpleDateFormat("MMM d, yyyy", Locale.getDefault())
    }
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Secure credentials") },
                actions = {
                    IconButton(onClick = vm::refresh, enabled = !vm.loading) {
                        Icon(Icons.Filled.Refresh, contentDescription = "Refresh credentials")
                    }
                },
            )
        },
    ) { padding ->
        Column(
            Modifier
                .fillMaxSize()
                .padding(padding),
        ) {
            SettingsHint(
                "Only metadata is shown here. Secrets stay in the secure vault " +
                    "and are never displayed or logged.",
            )
            if (vm.loading && vm.credentials.isEmpty()) {
                Row(
                    Modifier.fillMaxSize(),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically,
                ) { CircularProgressIndicator() }
                return@Column
            }
            vm.error?.let {
                Text(
                    it,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(horizontal = 20.dp),
                )
            }
            if (vm.credentials.isEmpty() && !vm.loading) {
                EmptyState("No saved credentials", "Afnan stores secrets only when you allow it.")
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(vm.credentials, key = { it.id }) { credential ->
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 16.dp),
                        ) {
                            Column(Modifier.padding(16.dp)) {
                                Text(
                                    credential.label,
                                    fontWeight = FontWeight.SemiBold,
                                )
                                Spacer(Modifier.height(4.dp))
                                Text(
                                    "${credential.kind} • saved ${
                                        dateFormat.format(Date(credential.createdAt))
                                    }",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                                Text(
                                    "Value hidden by policy",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.primary,
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Permissions: real device permission states
// ---------------------------------------------------------------------------

private data class DevicePermission(
    val label: String,
    val permission: String,
    val rationale: String,
)

private val devicePermissions = listOf(
    DevicePermission(
        "Microphone",
        Manifest.permission.RECORD_AUDIO,
        "Voice input for talking to Afnan",
    ),
    DevicePermission(
        "Camera",
        Manifest.permission.CAMERA,
        "Scanning pairing QR codes",
    ),
    DevicePermission(
        "Notifications",
        Manifest.permission.POST_NOTIFICATIONS,
        "Approvals, task completion and security alerts",
    ),
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PermissionsScreen() {
    val context = LocalContext.current
    Scaffold(
        topBar = { TopAppBar(title = { Text("Permissions") }) },
    ) { padding ->
        LazyColumn(
            Modifier
                .fillMaxSize()
                .padding(padding),
        ) {
            item {
                SettingsSectionTitle("Device permissions")
                SettingsHint(
                    "These are Android runtime permissions. Afnan's own " +
                        "capabilities (browser, tools, connectors) are authorized " +
                        "per action through the approval system.",
                )
            }
            items(devicePermissions) { permission ->
                val granted = ContextCompat.checkSelfPermission(
                    context, permission.permission,
                ) == PackageManager.PERMISSION_GRANTED
                // Notifications permission only exists on Android 13+.
                val applicable = permission.permission !=
                    Manifest.permission.POST_NOTIFICATIONS ||
                    android.os.Build.VERSION.SDK_INT >=
                    android.os.Build.VERSION_CODES.TIRAMISU
                if (applicable) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 20.dp, vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(
                            if (granted) Icons.Filled.CheckCircle
                            else Icons.Filled.LinkOff,
                            contentDescription = if (granted) "Granted" else "Not granted",
                            tint = if (granted) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Spacer(Modifier.width(12.dp))
                        Column(Modifier.weight(1f)) {
                            Text(
                                permission.label,
                                style = MaterialTheme.typography.titleSmall,
                            )
                            Text(
                                permission.rationale,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        Text(
                            if (granted) "Granted" else "Not granted",
                            style = MaterialTheme.typography.labelMedium,
                            color = if (granted) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.error,
                        )
                    }
                }
            }
            item {
                SettingsSectionTitle("Afnan capabilities")
                SettingsHint(
                    "Sensitive or irreversible operations always require your " +
                        "explicit approval — there is no 'always allow everything' " +
                        "switch. Review pending requests in the Approvals tab.",
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Data controls
// ---------------------------------------------------------------------------

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DataControlsScreen(
    services: AppServices,
    onSignedOut: () -> Unit,
) {
    val context = LocalContext.current
    val app = context.applicationContext as AfnanApp
    var showSignOut by remember { mutableStateOf(false) }

    Scaffold(
        topBar = { TopAppBar(title = { Text("Data controls") }) },
    ) { padding ->
        Column(
            Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(20.dp),
        ) {
            Text(
                "This phone stores only the server address and your session " +
                    "token, encrypted. Your tasks, memory and files live on " +
                    "the paired Afnan runtime.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(16.dp))
            Text(
                "Connected server",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                services.tokenStore.getBaseUrl().ifBlank { "Not set" },
                style = MaterialTheme.typography.bodyLarge,
            )
            Spacer(Modifier.height(24.dp))
            OutlinedButton(
                onClick = { showSignOut = true },
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Sign out this device") }
            Spacer(Modifier.height(8.dp))
            Text(
                "Signing out clears the token and server address on this phone " +
                    "only. Nothing is deleted from the Afnan runtime.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }

    if (showSignOut) {
        AlertDialog(
            onDismissRequest = { showSignOut = false },
            title = { Text("Sign out?") },
            text = {
                Text(
                    "This clears the server address and session token on this " +
                        "phone. You will need to pair again.",
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        app.signOutLocal()
                        showSignOut = false
                        onSignedOut()
                    },
                ) { Text("Sign out") }
            },
            dismissButton = {
                TextButton(onClick = { showSignOut = false }) { Text("Cancel") }
            },
        )
    }
}

// ---------------------------------------------------------------------------
// About
// ---------------------------------------------------------------------------

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AboutScreen() {
    Scaffold(
        topBar = { TopAppBar(title = { Text("About") }) },
    ) { padding ->
        Column(
            Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(24.dp),
        ) {
            Text(
                "Afnan",
                style = MaterialTheme.typography.displayMedium,
                color = MaterialTheme.colorScheme.primary,
            )
            Text(
                "AI",
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.secondary,
            )
            Spacer(Modifier.height(16.dp))
            Text("Version 1.0.0", style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.height(8.dp))
            Text(
                "Afnan AI for Android is the mobile surface for your Afnan " +
                    "runtime. Planning, execution, memory, browser automation " +
                    "and security all run in the existing Afnan architecture " +
                    "on your paired machine — this app monitors, approves " +
                    "and converses.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(16.dp))
            val context = LocalContext.current
            Button(
                onClick = {
                    // Real update channel: the project's release page.
                    val intent = Intent(
                        Intent.ACTION_VIEW,
                        android.net.Uri.parse(
                            "https://github.com/Labbaik757/Afnan-Ai-1.2/releases",
                        ),
                    )
                    runCatching { context.startActivity(intent) }
                },
            ) {
                Text("Check for updates")
            }
            Spacer(Modifier.height(4.dp))
            Text(
                "Updates are delivered through the Play Store / APK channel.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

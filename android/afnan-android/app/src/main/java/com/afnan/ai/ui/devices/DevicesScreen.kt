package com.afnan.ai.ui.devices

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Smartphone
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.afnan.ai.ui.AppServices
import com.afnan.ai.ui.DeviceView
import com.afnan.ai.ui.formatTimeAgo
import com.afnan.ai.ui.tasks.EmptyState
import com.afnan.ai.ui.viewModelFactory
import kotlinx.coroutines.launch

class DevicesViewModel(private val services: AppServices) : ViewModel() {
    var devices by mutableStateOf<List<DeviceView>>(emptyList())
        private set
    var loading by mutableStateOf(false)
        private set
    var error by mutableStateOf<String?>(null)
        private set
    var revoking by mutableStateOf(false)
        private set

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            loading = true
            error = null
            try {
                devices = services.repository.listDevices()
            } catch (e: Exception) {
                error = e.message ?: "Couldn't load devices"
            } finally {
                loading = false
            }
        }
    }

    fun revoke(device: DeviceView, onDone: () -> Unit = {}) {
        viewModelScope.launch {
            revoking = true
            error = null
            try {
                services.repository.revokeDevice(device.id)
                devices = services.repository.listDevices()
                onDone()
            } catch (e: Exception) {
                error = e.message ?: "Couldn't revoke device"
            } finally {
                revoking = false
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DevicesScreen(
    services: AppServices,
    vm: DevicesViewModel = viewModel(factory = viewModelFactory { DevicesViewModel(services) }),
) {
    var revokeTarget by remember { mutableStateOf<DeviceView?>(null) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Devices") },
                actions = {
                    IconButton(onClick = vm::refresh, enabled = !vm.loading) {
                        Icon(Icons.Filled.Refresh, contentDescription = "Refresh devices")
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
            if (vm.loading && vm.devices.isEmpty()) {
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
                    modifier = Modifier.padding(16.dp),
                )
            }
            if (vm.devices.isEmpty() && !vm.loading) {
                EmptyState("No paired devices", "Pair this phone to see it here.")
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    items(vm.devices, key = { it.id }) { device ->
                        DeviceCard(
                            device = device,
                            onRevoke = { revokeTarget = device },
                        )
                    }
                }
            }
        }
    }

    val target = revokeTarget
    if (target != null) {
        AlertDialog(
            onDismissRequest = { revokeTarget = null },
            title = { Text("Revoke ${target.name}?") },
            text = {
                Text(
                    "This immediately invalidates the device's sessions and tokens. " +
                        "The device must be paired again to reconnect.",
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        vm.revoke(target) { revokeTarget = null }
                    },
                    enabled = !vm.revoking,
                ) { Text("Revoke") }
            },
            dismissButton = {
                TextButton(onClick = { revokeTarget = null }) { Text("Cancel") }
            },
        )
    }
}

@Composable
private fun DeviceCard(device: DeviceView, onRevoke: () -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (device.isCurrentDevice)
                MaterialTheme.colorScheme.primaryContainer
            else MaterialTheme.colorScheme.surface,
        ),
    ) {
        Row(
            Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Surface(
                shape = CircleShape,
                color = if (device.online) Color(0xFF2E7D32) else Color(0xFF9E9E9E),
                modifier = Modifier.size(12.dp),
            ) {}
            Spacer(Modifier.width(12.dp))
            Icon(
                Icons.Filled.Smartphone,
                contentDescription = "Device",
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        device.name,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.weight(1f),
                    )
                    if (device.isCurrentDevice) {
                        Text(
                            "This device",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary,
                            fontWeight = FontWeight.Bold,
                        )
                    }
                }
                Text(
                    "${device.platform} • ${if (device.online) "Online" else "Offline"}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    "Last seen ${formatTimeAgo(device.lastSeenAt)}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                if (device.capabilities.isNotEmpty()) {
                    Text(
                        device.capabilities.take(4).joinToString(", "),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                    )
                }
            }
            if (!device.isCurrentDevice) {
                OutlinedButton(onClick = onRevoke) { Text("Revoke") }
            }
        }
    }
    Spacer(Modifier.height(0.dp))
}

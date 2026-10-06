package com.afnan.ai.ui.setup

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.afnan.ai.ui.AppServices
import kotlinx.coroutines.launch

/**
 * First-run setup: point the app at the Afnan runtime on the PC.
 * "Test connection" performs a real GET /v1/health before continuing.
 */
@Composable
fun SetupScreen(
    services: AppServices,
    onDone: () -> Unit,
) {
    var host by remember { mutableStateOf("") }
    var port by remember { mutableStateOf("8765") }
    var useTls by remember { mutableStateOf(true) }
    var testing by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    val scope = rememberCoroutineScope()

    fun baseUrl(): String {
        val scheme = if (useTls) "https" else "http"
        return "$scheme://${host.trim()}:${port.trim()}"
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            "Connect to your Afnan",
            style = MaterialTheme.typography.headlineMedium,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            "Afnan runs on your PC. Enter its address so this phone can pair with it.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(24.dp))

        OutlinedTextField(
            value = host,
            onValueChange = { host = it; error = null },
            label = { Text("PC host or IP address") },
            placeholder = { Text("192.168.1.10") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(12.dp))
        OutlinedTextField(
            value = port,
            onValueChange = { port = it.filter(Char::isDigit); error = null },
            label = { Text("Port") },
            singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(12.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Column(Modifier.weight(1f)) {
                Text("Use TLS (recommended)")
                Text(
                    "Plain HTTP is only allowed for development on loopback.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Switch(checked = useTls, onCheckedChange = { useTls = it })
        }

        error?.let {
            Spacer(Modifier.height(12.dp))
            Text(it, color = MaterialTheme.colorScheme.error)
        }

        Spacer(Modifier.height(24.dp))
        Button(
            onClick = {
                val h = host.trim()
                val p = port.trim().toIntOrNull()
                if (h.isEmpty() || p == null || p !in 1..65535) {
                    error = "Enter a valid host and port."
                    return@Button
                }
                testing = true
                error = null
                scope.launch {
                    try {
                        val ok = services.repository.checkHealth(baseUrl())
                        if (ok) {
                            services.tokenStore.setBaseUrl(baseUrl())
                            onDone()
                        } else {
                            error = "The server answered but is not healthy. Check the address."
                        }
                    } catch (e: Exception) {
                        error = "Couldn't reach Afnan: ${e.message ?: "connection failed"}"
                    } finally {
                        testing = false
                    }
                }
            },
            enabled = !testing,
            modifier = Modifier.fillMaxWidth(),
        ) {
            if (testing) {
                CircularProgressIndicator(
                    modifier = Modifier.padding(end = 8.dp),
                    strokeWidth = 2.dp,
                )
            }
            Text(if (testing) "Testing…" else "Test connection & continue")
        }
    }
}

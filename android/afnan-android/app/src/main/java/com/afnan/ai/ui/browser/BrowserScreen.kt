package com.afnan.ai.ui.browser

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowForward
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.afnan.ai.ui.AppServices
import com.afnan.ai.ui.BrowserStateView
import com.afnan.ai.ui.viewModelFactory
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class BrowserViewModel(private val services: AppServices) : ViewModel() {
    var state by mutableStateOf<BrowserStateView?>(null)
        private set
    var bitmap by mutableStateOf<android.graphics.Bitmap?>(null)
        private set
    var error by mutableStateOf<String?>(null)
        private set
    var navigating by mutableStateOf(false)
        private set

    /** Live polling loop; started/stopped with the screen lifecycle. */
    fun startLiveView() {
        viewModelScope.launch {
            while (isActive) {
                try {
                    state = services.repository.browserState()
                    val bytes = services.repository.browserScreenshot()
                    if (bytes.isNotEmpty()) {
                        BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                            ?.let { bitmap = it }
                    }
                    error = null
                } catch (e: Exception) {
                    error = e.message ?: "Browser unavailable"
                }
                delay(REFRESH_MS)
            }
        }.also { liveJob = it }
    }

    private var liveJob: kotlinx.coroutines.Job? = null

    fun stopLiveView() {
        liveJob?.cancel()
        liveJob = null
    }

    fun navigate(url: String) {
        var target = url.trim()
        if (target.isEmpty()) return
        if (!target.startsWith("http://") && !target.startsWith("https://")) {
            target = "https://$target"
        }
        viewModelScope.launch {
            navigating = true
            error = null
            try {
                state = services.repository.browserNavigate(target)
            } catch (e: Exception) {
                error = e.message ?: "Navigation failed"
            } finally {
                navigating = false
            }
        }
    }

    companion object {
        private const val REFRESH_MS = 3_000L
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BrowserScreen(
    services: AppServices,
    vm: BrowserViewModel = viewModel(factory = viewModelFactory { BrowserViewModel(services) }),
) {
    var urlInput by remember { mutableStateOf("") }
    val scope = rememberCoroutineScope()

    DisposableEffect(Unit) {
        vm.startLiveView()
        onDispose { vm.stopLiveView() }
    }

    // Keep the address bar in sync with the live page.
    LaunchedEffect(vm.state?.url) {
        val live = vm.state?.url.orEmpty()
        if (live.isNotBlank() && urlInput.isEmpty()) urlInput = live
    }

    Scaffold(
        topBar = {
            TopAppBar(title = { Text("Browser") })
        },
    ) { padding ->
        Column(
            Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(12.dp),
        ) {
            // Execution-context label: this browser runs on the PC.
            Surface(
                color = MaterialTheme.colorScheme.primaryContainer,
                shape = MaterialTheme.shapes.small,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    "Running on your PC — live view",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                )
            }
            Spacer(Modifier.height(8.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                OutlinedTextField(
                    value = urlInput,
                    onValueChange = { urlInput = it },
                    label = { Text("Address") },
                    singleLine = true,
                    modifier = Modifier.weight(1f),
                )
                Spacer(Modifier.width(8.dp))
                IconButton(
                    onClick = { scope.launch { vm.navigate(urlInput) } },
                    enabled = !vm.navigating,
                ) {
                    if (vm.navigating) {
                        CircularProgressIndicator(
                            strokeWidth = 2.dp,
                            modifier = Modifier.width(24.dp),
                        )
                    } else {
                        Icon(Icons.Filled.ArrowForward, contentDescription = "Navigate")
                    }
                }
            }

            val live = vm.state
            if (live != null) {
                Spacer(Modifier.height(8.dp))
                Text(
                    live.title.ifBlank { live.url },
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    maxLines = 1,
                )
                Text(
                    live.url,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                )
                if (live.loading) {
                    Spacer(Modifier.height(4.dp))
                    LinearProgressIndicator(Modifier.fillMaxWidth())
                }
            }

            Spacer(Modifier.height(8.dp))
            Card(modifier = Modifier.fillMaxWidth()) {
                val bmp = vm.bitmap
                if (bmp != null) {
                    Image(
                        bitmap = bmp.asImageBitmap(),
                        contentDescription = "Live screenshot of the remote browser on your PC",
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(16f / 10f),
                        contentScale = ContentScale.Fit,
                    )
                } else {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(220.dp),
                        horizontalArrangement = Arrangement.Center,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        CircularProgressIndicator()
                        Spacer(Modifier.width(12.dp))
                        Text("Waiting for the browser…")
                    }
                }
            }

            vm.error?.let {
                Spacer(Modifier.height(8.dp))
                Text(it, color = MaterialTheme.colorScheme.error)
                Spacer(Modifier.height(4.dp))
                IconButton(onClick = { vm.stopLiveView(); vm.startLiveView() }) {
                    Icon(Icons.Filled.Refresh, contentDescription = "Reconnect browser view")
                }
            }

            Spacer(Modifier.height(8.dp))
            Text(
                "Only navigate / screenshot / state are exposed — the same " +
                    "capabilities the control-plane protocol allows. " +
                    "Sensitive pages still require your approval.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

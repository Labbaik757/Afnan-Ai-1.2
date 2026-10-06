package com.afnan.ai.ui.pairing

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.afnan.ai.ui.AppServices
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.common.InputImage
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.concurrent.Executors

// ---------------------------------------------------------------------------
// State machine: Idle -> Requesting -> WaitingApproval(code shown) -> Done.
// Redeem is polled until the owner approves on the PC or it times out.
// ---------------------------------------------------------------------------

sealed interface PairingUiState {
    data object Idle : PairingUiState
    data object Requesting : PairingUiState
    data class WaitingApproval(val code: String) : PairingUiState
    data object Done : PairingUiState
    data class Error(val message: String) : PairingUiState
}

class PairingViewModel(private val services: AppServices) : ViewModel() {

    var state by mutableStateOf<PairingUiState>(PairingUiState.Idle)
        private set

    /**
     * Starts pairing against [host]:[port]. When [qrCode]/[qrPairingId] are
     * supplied (scanned from the PC), we redeem directly; otherwise a fresh
     * pairing request is issued and its 6-digit code is shown for the owner
     * to approve on the PC.
     */
    fun startPairing(
        host: String,
        port: Int,
        useTls: Boolean,
        qrCode: String? = null,
        qrPairingId: String? = null,
    ) {
        if (state is PairingUiState.Requesting ||
            state is PairingUiState.WaitingApproval
        ) {
            return
        }
        viewModelScope.launch {
            state = PairingUiState.Requesting
            try {
                val scheme = if (useTls) "https" else "http"
                services.tokenStore.setBaseUrl("$scheme://$host:$port")
                val (pairingId, code) =
                    if (qrCode != null && qrPairingId != null) {
                        qrPairingId to qrCode
                    } else {
                        val info = services.repository.requestPairing(
                            host = host,
                            port = port,
                            useTls = useTls,
                            deviceName = "${Build.MANUFACTURER} ${Build.MODEL}".trim(),
                        )
                        state = PairingUiState.WaitingApproval(info.code)
                        info.pairingId to info.code
                    }
                pollRedeem(pairingId, code)
            } catch (e: Exception) {
                state = PairingUiState.Error(
                    e.message ?: "Pairing failed. Check the server address.",
                )
            }
        }
    }

    private suspend fun pollRedeem(pairingId: String, code: String) {
        val deadline = System.currentTimeMillis() + REDEEM_TIMEOUT_MS
        while (viewModelScope.isActive && System.currentTimeMillis() < deadline) {
            val token = try {
                services.repository.redeemPairing(pairingId, code)
            } catch (e: Exception) {
                state = PairingUiState.Error(
                    e.message ?: "Pairing failed during approval.",
                )
                return
            }
            if (token != null) {
                services.tokenStore.setToken(token)
                state = PairingUiState.Done
                return
            }
            delay(POLL_INTERVAL_MS)
        }
        if (viewModelScope.isActive) {
            state = PairingUiState.Error(
                "Approval timed out. Ask the PC owner to approve, then try again.",
            )
        }
    }

    fun reset() {
        state = PairingUiState.Idle
    }

    companion object {
        private const val POLL_INTERVAL_MS = 3_000L
        private const val REDEEM_TIMEOUT_MS = 5 * 60 * 1000L
    }
}

private data class PairQr(
    val host: String,
    val port: Int,
    val useTls: Boolean,
    val pairingId: String?,
    val code: String?,
)

/** Parses afnan://pair?host=&port=[&tls=][&pid=][&code=] */
private fun parsePairQr(raw: String): PairQr? {
    val uri = runCatching { Uri.parse(raw) }.getOrNull() ?: return null
    if (uri.scheme != "afnan" || uri.host != "pair") return null
    val host = uri.getQueryParameter("host")?.trim().orEmpty()
    val port = uri.getQueryParameter("port")?.toIntOrNull() ?: return null
    if (host.isEmpty() || port !in 1..65535) return null
    return PairQr(
        host = host,
        port = port,
        useTls = uri.getQueryParameter("tls") == "1",
        pairingId = uri.getQueryParameter("pid")?.trim()?.ifEmpty { null },
        code = uri.getQueryParameter("code")?.trim()?.ifEmpty { null },
    )
}

// ---------------------------------------------------------------------------
// Screen
// ---------------------------------------------------------------------------

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PairingScreen(
    services: AppServices,
    vm: PairingViewModel = androidx.lifecycle.viewmodel.compose.viewModel(
        factory = com.afnan.ai.ui.viewModelFactory { PairingViewModel(services) },
    ),
    onPaired: () -> Unit,
) {
    var tab by remember { mutableIntStateOf(0) }
    val state = vm.state

    LaunchedEffect(state) {
        if (state is PairingUiState.Done) onPaired()
    }

    Column(Modifier.fillMaxSize()) {
        Column(Modifier.padding(24.dp)) {
            Text("Pair this device", style = MaterialTheme.typography.headlineMedium)
            Spacer(Modifier.height(8.dp))
            Text(
                "Link this phone to the Afnan runtime on your PC. " +
                    "The PC owner approves every new device.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        TabRow(selectedTabIndex = tab) {
            Tab(selected = tab == 0, onClick = { tab = 0 }, text = { Text("Scan QR") })
            Tab(selected = tab == 1, onClick = { tab = 1 }, text = { Text("Enter code") })
        }
        when (tab) {
            0 -> QrPairTab(vm = vm, state = state)
            1 -> ManualPairTab(vm = vm, state = state, services = services)
        }
    }
}

@Composable
private fun PairingStatusBlock(
    state: PairingUiState,
    onRetry: () -> Unit,
) {
    when (state) {
        is PairingUiState.Requesting -> {
            Spacer(Modifier.height(16.dp))
            CircularProgressIndicator()
            Spacer(Modifier.height(8.dp))
            Text("Contacting Afnan…")
        }

        is PairingUiState.WaitingApproval -> {
            Spacer(Modifier.height(16.dp))
            Text(
                "Approve on your PC",
                style = MaterialTheme.typography.titleMedium,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                state.code,
                style = MaterialTheme.typography.displayLarge.copy(
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 8.sp,
                ),
                color = MaterialTheme.colorScheme.primary,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(8.dp))
            Text(
                "Show this code to the PC owner for approval. " +
                    "This screen continues automatically once approved.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(12.dp))
            CircularProgressIndicator()
        }

        is PairingUiState.Error -> {
            Spacer(Modifier.height(16.dp))
            Text(state.message, color = MaterialTheme.colorScheme.error)
            Spacer(Modifier.height(8.dp))
            Button(onClick = onRetry) { Text("Try again") }
        }

        else -> Unit
    }
}

// ---------------------------------------------------------------------------
// Tab 1: QR scan (CameraX + ML Kit)
// ---------------------------------------------------------------------------

@Composable
private fun QrPairTab(
    vm: PairingViewModel,
    state: PairingUiState,
) {
    val context = LocalContext.current
    var hasCamera by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context, Manifest.permission.CAMERA,
            ) == PackageManager.PERMISSION_GRANTED,
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted -> hasCamera = granted }
    var scanned by remember { mutableStateOf<PairQr?>(null) }
    var scanError by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) {
        if (!hasCamera) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        when {
            !hasCamera -> {
                Text("Camera access is needed to scan the pairing QR code.")
                Spacer(Modifier.height(12.dp))
                Button(onClick = {
                    permissionLauncher.launch(Manifest.permission.CAMERA)
                }) { Text("Grant camera access") }
            }

            scanned == null &&
                state is PairingUiState.Idle -> {
                Text(
                    "Point the camera at the QR code shown by Afnan on your PC.",
                    textAlign = TextAlign.Center,
                )
                Spacer(Modifier.height(12.dp))
                QrCameraView(
                    onCode = { raw ->
                        val parsed = parsePairQr(raw)
                        if (parsed == null) {
                            scanError = "That QR code is not an Afnan pairing code."
                        } else {
                            scanned = parsed
                        }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(320.dp),
                )
                scanError?.let {
                    Spacer(Modifier.height(8.dp))
                    Text(it, color = MaterialTheme.colorScheme.error)
                }
            }

            else -> {
                val qr = scanned
                if (qr != null && state is PairingUiState.Idle) {
                    Text("Found Afnan at ${qr.host}:${qr.port}")
                    Spacer(Modifier.height(12.dp))
                    Button(
                        onClick = {
                            vm.startPairing(
                                host = qr.host,
                                port = qr.port,
                                useTls = qr.useTls,
                                qrCode = qr.code,
                                qrPairingId = qr.pairingId,
                            )
                        },
                        modifier = Modifier.fillMaxWidth(),
                    ) { Text("Pair with this server") }
                }
                PairingStatusBlock(state = state, onRetry = {
                    scanned = null
                    vm.reset()
                })
            }
        }
    }
}

@Composable
private fun QrCameraView(
    onCode: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val executor = remember { Executors.newSingleThreadExecutor() }
    var lastResult by remember { mutableStateOf("") }

    DisposableEffect(Unit) {
        onDispose { executor.shutdown() }
    }

    AndroidView(
        factory = { ctx ->
            val previewView = PreviewView(ctx)
            val future = ProcessCameraProvider.getInstance(ctx)
            future.addListener(
                {
                    val provider = future.get()
                    val preview = androidx.camera.core.Preview.Builder()
                        .build()
                        .also { it.setSurfaceProvider(previewView.surfaceProvider) }
                    val analysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(
                            ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST,
                        )
                        .build()
                    val scanner = BarcodeScanning.getClient()
                    analysis.setAnalyzer(executor) { imageProxy ->
                        val mediaImage = imageProxy.image
                        if (mediaImage != null) {
                            val image = InputImage.fromMediaImage(
                                mediaImage, imageProxy.imageInfo.rotationDegrees,
                            )
                            scanner.process(image)
                                .addOnSuccessListener { barcodes ->
                                    val raw = barcodes.firstOrNull()?.rawValue
                                    if (!raw.isNullOrEmpty() && raw != lastResult) {
                                        lastResult = raw
                                        onCode(raw)
                                    }
                                }
                                .addOnCompleteListener { imageProxy.close() }
                        } else {
                            imageProxy.close()
                        }
                    }
                    runCatching {
                        provider.unbindAll()
                        provider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_BACK_CAMERA,
                            preview,
                            analysis,
                        )
                    }
                },
                ContextCompat.getMainExecutor(ctx),
            )
            previewView
        },
        modifier = modifier,
    )
}

// ---------------------------------------------------------------------------
// Tab 2: manual 6-digit code
// ---------------------------------------------------------------------------

@Composable
private fun ManualPairTab(
    vm: PairingViewModel,
    state: PairingUiState,
    services: AppServices,
) {
    var host by remember {
        mutableStateOf(
            services.tokenStore.getBaseUrl()
                .removePrefix("https://").removePrefix("http://")
                .substringBefore(":"),
        )
    }
    var port by remember {
        mutableStateOf(
            services.tokenStore.getBaseUrl().substringAfterLast(":")
                .filter(Char::isDigit).ifEmpty { "8765" },
        )
    }
    var useTls by remember {
        mutableStateOf(services.tokenStore.getBaseUrl().startsWith("https"))
    }
    var code by remember { mutableStateOf("") }
    var pairingId by remember { mutableStateOf("") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.Top,
    ) {
        Text(
            "If your PC shows a pairing code, enter it here. " +
                "Otherwise request a fresh code below.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(16.dp))
        OutlinedTextField(
            value = host,
            onValueChange = { host = it },
            label = { Text("PC host or IP") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(8.dp))
        OutlinedTextField(
            value = port,
            onValueChange = { port = it.filter(Char::isDigit) },
            label = { Text("Port") },
            singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(8.dp))
        OutlinedTextField(
            value = code,
            onValueChange = { code = it.filter(Char::isDigit).take(6) },
            label = { Text("6-digit code (if you have one)") },
            singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.NumberPassword),
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(8.dp))
        OutlinedTextField(
            value = pairingId,
            onValueChange = { pairingId = it },
            label = { Text("Pairing ID (only with a PC-provided code)") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(16.dp))
        Button(
            onClick = {
                val p = port.toIntOrNull()
                if (host.isBlank() || p == null) return@Button
                vm.startPairing(
                    host = host.trim(),
                    port = p,
                    useTls = useTls,
                    qrCode = code.ifBlank { null },
                    qrPairingId = pairingId.ifBlank { null },
                )
            },
            enabled = state is PairingUiState.Idle || state is PairingUiState.Error,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(
                if (code.isNotBlank() && pairingId.isNotBlank()) "Redeem code"
                else "Request pairing code",
            )
        }
        PairingStatusBlock(state = state, onRetry = { vm.reset() })
    }
}

package com.afnan.ai.ui.approvals

import androidx.compose.foundation.clickable
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
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.afnan.ai.ui.AppServices
import com.afnan.ai.ui.ApprovalStatus
import com.afnan.ai.ui.ApprovalView
import com.afnan.ai.ui.formatCountdown
import com.afnan.ai.ui.formatTimeAgo
import com.afnan.ai.ui.theme.riskColor
import com.afnan.ai.ui.tasks.EmptyState
import com.afnan.ai.ui.viewModelFactory
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class ApprovalsViewModel(private val repository: com.afnan.ai.ui.AfnanRepository) : ViewModel() {
    var approvals by mutableStateOf<List<ApprovalView>>(emptyList())
        private set
    var loading by mutableStateOf(false)
        private set
    var error by mutableStateOf<String?>(null)
        private set

    val pendingCount: Int
        get() = approvals.count { it.status == ApprovalStatus.PENDING }

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            loading = true
            error = null
            try {
                approvals = repository.listApprovals()
            } catch (e: Exception) {
                error = e.message ?: "Couldn't load approvals"
            } finally {
                loading = false
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ApprovalsScreen(
    services: AppServices,
    onApprovalClick: (String) -> Unit,
    vm: ApprovalsViewModel = viewModel(
        factory = viewModelFactory { ApprovalsViewModel(services.repository) },
    ),
) {
    // Tick every second so expiry countdowns stay live.
    var now by remember { mutableLongStateOf(System.currentTimeMillis()) }
    LaunchedEffect(Unit) {
        while (true) {
            delay(1000)
            now = System.currentTimeMillis()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Approvals") },
                actions = {
                    IconButton(onClick = vm::refresh, enabled = !vm.loading) {
                        Icon(Icons.Filled.Refresh, contentDescription = "Refresh approvals")
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
            if (vm.loading && vm.approvals.isEmpty()) {
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
            val pending = vm.approvals.filter { it.status == ApprovalStatus.PENDING }
            val decided = vm.approvals.filter { it.status != ApprovalStatus.PENDING }
            if (vm.approvals.isEmpty() && !vm.loading) {
                EmptyState("No approvals", "Afnan will ask here before sensitive actions.")
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    if (pending.isNotEmpty()) {
                        item {
                            Text(
                                "Pending (${pending.size})",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
                            )
                        }
                        items(pending, key = { it.id }) { approval ->
                            ApprovalRow(
                                approval = approval,
                                now = now,
                                onClick = { onApprovalClick(approval.id) },
                            )
                        }
                    }
                    if (decided.isNotEmpty()) {
                        item {
                            Text(
                                "Decided",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
                            )
                        }
                        items(decided, key = { it.id }) { approval ->
                            ApprovalRow(
                                approval = approval,
                                now = now,
                                onClick = { onApprovalClick(approval.id) },
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ApprovalRow(
    approval: ApprovalView,
    now: Long,
    onClick: () -> Unit,
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp)
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(
            containerColor = if (approval.status == ApprovalStatus.PENDING)
                MaterialTheme.colorScheme.surfaceVariant
            else MaterialTheme.colorScheme.surface,
        ),
    ) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    Icons.Filled.Warning,
                    contentDescription = "Approval",
                    tint = riskColor(approval.risk),
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    approval.action,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.weight(1f),
                )
                RiskChip(approval)
            }
            Spacer(Modifier.height(4.dp))
            Text(
                approval.target,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(4.dp))
            Row {
                Text(
                    "Requested ${formatTimeAgo(approval.requestedAt, now)}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f),
                )
                if (approval.status == ApprovalStatus.PENDING) {
                    Text(
                        "Expires in ${formatCountdown(approval.expiresAt, now)}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                        fontWeight = FontWeight.Bold,
                    )
                } else {
                    Text(
                        approval.status.name.lowercase()
                            .replaceFirstChar { it.uppercase() },
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun RiskChip(approval: ApprovalView) {
    val color = riskColor(approval.risk)
    Surface(
        color = color.copy(alpha = 0.15f),
        shape = MaterialTheme.shapes.small,
    ) {
        Text(
            approval.risk.name,
            color = color,
            style = MaterialTheme.typography.labelMedium,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
        )
    }
}

// ---------------------------------------------------------------------------
// Detail: full action context + Approve / Deny. Decisions go through the
// existing ApprovalCenter via the repository — never a parallel mechanism.
// ---------------------------------------------------------------------------

class ApprovalDetailViewModel(private val services: AppServices) : ViewModel() {
    var approval by mutableStateOf<ApprovalView?>(null)
        private set
    var loading by mutableStateOf(false)
        private set
    var deciding by mutableStateOf(false)
        private set
    var error by mutableStateOf<String?>(null)
        private set

    fun load(id: String) {
        viewModelScope.launch {
            loading = true
            error = null
            try {
                approval = services.repository.getApproval(id)
            } catch (e: Exception) {
                error = e.message ?: "Couldn't load approval"
            } finally {
                loading = false
            }
        }
    }

    fun decide(id: String, approve: Boolean, onDone: () -> Unit) {
        viewModelScope.launch {
            deciding = true
            error = null
            try {
                if (approve) services.repository.approve(id)
                else services.repository.deny(id)
                services.notifier.cancelApproval(id)
                onDone()
            } catch (e: Exception) {
                error = e.message ?: "Couldn't record decision"
            } finally {
                deciding = false
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ApprovalDetailScreen(
    services: AppServices,
    approvalId: String,
    onBack: () -> Unit,
    onDecided: () -> Unit,
    vm: ApprovalDetailViewModel = viewModel(
        factory = viewModelFactory { ApprovalDetailViewModel(services) },
    ),
) {
    LaunchedEffect(approvalId) { vm.load(approvalId) }
    var now by remember { mutableLongStateOf(System.currentTimeMillis()) }
    LaunchedEffect(Unit) {
        while (true) {
            delay(1000)
            now = System.currentTimeMillis()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Approval") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
            )
        },
    ) { padding ->
        val approval = vm.approval
        Column(
            Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
        ) {
            when {
                vm.loading && approval == null -> {
                    Row(
                        Modifier.fillMaxSize(),
                        horizontalArrangement = Arrangement.Center,
                        verticalAlignment = Alignment.CenterVertically,
                    ) { CircularProgressIndicator() }
                }

                approval != null -> {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            "Afnan is requesting permission",
                            style = MaterialTheme.typography.titleMedium,
                            modifier = Modifier.weight(1f),
                        )
                        RiskChip(approval)
                    }
                    Spacer(Modifier.height(16.dp))
                    DetailRow("Action", approval.action)
                    DetailRow("Target", approval.target)
                    if (approval.reason.isNotBlank()) {
                        DetailRow("Reason", approval.reason)
                    }
                    DetailRow(
                        "Requested",
                        "${formatTimeAgo(approval.requestedAt, now)}",
                    )
                    if (approval.taskId != null) {
                        DetailRow("Task", approval.taskId)
                    }
                    if (approval.parameters.isNotEmpty()) {
                        Spacer(Modifier.height(8.dp))
                        Text(
                            "Details",
                            fontWeight = FontWeight.Bold,
                            style = MaterialTheme.typography.titleSmall,
                        )
                        approval.parameters.forEach { (key, value) ->
                            DetailRow(key, value)
                        }
                    }
                    Spacer(Modifier.height(16.dp))
                    if (approval.status == ApprovalStatus.PENDING) {
                        Text(
                            "Expires in ${formatCountdown(approval.expiresAt, now)}",
                            color = MaterialTheme.colorScheme.error,
                            fontWeight = FontWeight.Bold,
                        )
                        Spacer(Modifier.height(12.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                            Button(
                                onClick = {
                                    vm.decide(approvalId, true) { onDecided() }
                                },
                                enabled = !vm.deciding,
                                modifier = Modifier.weight(1f),
                            ) { Text("Approve") }
                            OutlinedButton(
                                onClick = {
                                    vm.decide(approvalId, false) { onDecided() }
                                },
                                enabled = !vm.deciding,
                                modifier = Modifier.weight(1f),
                            ) { Text("Deny") }
                        }
                        if (vm.deciding) {
                            Spacer(Modifier.height(12.dp))
                            LinearProgressIndicator(Modifier.fillMaxWidth())
                        }
                    } else {
                        Text(
                            "This approval was ${approval.status.name.lowercase()}.",
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
            vm.error?.let {
                Spacer(Modifier.height(8.dp))
                Text(it, color = MaterialTheme.colorScheme.error)
            }
        }
    }
}

@Composable
private fun DetailRow(label: String, value: String) {
    Column(Modifier.padding(vertical = 4.dp)) {
        Text(
            label,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(value, style = MaterialTheme.typography.bodyMedium)
    }
}

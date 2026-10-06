package com.afnan.ai.ui.chat

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
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
import com.afnan.ai.AfnanApp
import com.afnan.ai.ui.AppServices
import com.afnan.ai.ui.ApprovalStatus
import com.afnan.ai.ui.ChatMessage
import com.afnan.ai.ui.ChatRole
import com.afnan.ai.ui.TaskStatus
import com.afnan.ai.ui.formatCountdown
import com.afnan.ai.ui.theme.riskColor
import com.afnan.ai.worker.AfnanSyncService
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class ChatViewModel(private val services: AppServices) : ViewModel() {

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    var input by mutableStateOf("")
        private set
    var sending by mutableStateOf(false)
        private set
    var listening by mutableStateOf(false)
        private set
    var voiceError by mutableStateOf<String?>(null)
        private set

    private val trackJobs = mutableMapOf<String, Job>()
    private val shownApprovals = mutableSetOf<String>()

    fun onInputChange(value: String) {
        input = value
    }

    fun send() {
        val text = input.trim()
        if (text.isBlank() || sending) return
        input = ""
        voiceError = null
        viewModelScope.launch {
            sending = true
            addMessage(ChatMessage(role = ChatRole.USER, text = text))
            val progress = addMessage(
                ChatMessage(role = ChatRole.PROGRESS, text = "Planning…"),
            )
            try {
                val task = services.repository.sendTask(text)
                updateMessage(progress.id) {
                    it.copy(text = "Task started: ${task.title}")
                }
                trackTask(task.id, progress.id)
            } catch (e: Exception) {
                updateMessage(progress.id) {
                    it.copy(
                        role = ChatRole.ERROR,
                        text = "Couldn't start the task: ${e.message ?: "unknown error"}",
                    )
                }
            } finally {
                sending = false
            }
        }
    }

    /** Polls task progress + surfaces pending approvals inline until done. */
    private fun trackTask(taskId: String, progressId: String) {
        trackJobs[taskId]?.cancel()
        trackJobs[taskId] = viewModelScope.launch {
            try {
                while (isActive) {
                    delay(2000)
                    val task = try {
                        services.repository.getTask(taskId)
                    } catch (e: Exception) {
                        addMessage(
                            ChatMessage(
                                role = ChatRole.ERROR,
                                text = "Lost contact with the task: ${e.message ?: "network error"}",
                            ),
                        )
                        return@launch
                    }
                    updateMessage(progressId) {
                        it.copy(
                            text = "Working on \"${task.title}\" — " +
                                "${(task.progress * 100).toInt()}%",
                        )
                    }
                    // Surface approvals inline as cards.
                    try {
                        services.repository.listApprovals()
                            .filter {
                                it.taskId == taskId &&
                                    it.status == ApprovalStatus.PENDING &&
                                    shownApprovals.add(it.id)
                            }
                            .forEach { approval ->
                                addMessage(
                                    ChatMessage(
                                        role = ChatRole.APPROVAL,
                                        text = "Afnan needs your approval",
                                        approval = approval,
                                    ),
                                )
                                services.notifier.showApprovalRequested(
                                    approval.id,
                                    "Afnan needs your approval",
                                    "${approval.action} — ${approval.target}",
                                )
                            }
                    } catch (_: Exception) {
                        // Approvals are best-effort here; the Approvals tab is authoritative.
                    }
                    when (task.status) {
                        TaskStatus.COMPLETED -> {
                            val summary = task.resultSummary ?: task.title
                            updateMessage(progressId) {
                                it.copy(
                                    role = ChatRole.SUMMARY,
                                    text = "Done — $summary",
                                )
                            }
                            speakSummary("Task completed. $summary")
                            return@launch
                        }

                        TaskStatus.FAILED -> {
                            updateMessage(progressId) {
                                it.copy(
                                    role = ChatRole.ERROR,
                                    text = "Task failed: ${task.error ?: task.title}",
                                )
                            }
                            return@launch
                        }

                        TaskStatus.CANCELLED -> {
                            updateMessage(progressId) {
                                it.copy(
                                    role = ChatRole.ERROR,
                                    text = "Task was cancelled.",
                                )
                            }
                            return@launch
                        }

                        else -> Unit
                    }
                }
            } finally {
                trackJobs.remove(taskId)
            }
        }
    }

    private fun speakSummary(text: String) {
        val app = services.context.applicationContext as AfnanApp
        if (!app.voiceResponsesEnabled.value) return
        services.voiceOutput.speak(text.take(400), "ur-PK")
    }

    fun decideApproval(message: ChatMessage, approve: Boolean) {
        val approval = message.approval ?: return
        viewModelScope.launch {
            try {
                if (approve) services.repository.approve(approval.id)
                else services.repository.deny(approval.id)
                services.notifier.cancelApproval(approval.id)
                updateMessage(message.id) {
                    it.copy(
                        role = ChatRole.ASSISTANT,
                        text = if (approve) "Approved — continuing." else "Denied.",
                        approval = null,
                    )
                }
            } catch (e: Exception) {
                updateMessage(message.id) {
                    it.copy(
                        text = "Couldn't record the decision: ${e.message ?: "error"}. " +
                            "Please use the Approvals tab.",
                    )
                }
            }
        }
    }

    fun toggleListening() {
        if (listening) {
            services.voiceInput.stopListening()
            listening = false
            return
        }
        voiceError = null
        listening = true
        services.voiceInput.startListening(
            languageTag = "ur-PK",
            onResult = { text ->
                listening = false
                if (text.isNotBlank()) {
                    input = text
                }
            },
            onError = { err ->
                listening = false
                voiceError = err
            },
        )
    }

    fun stopVoice() {
        services.voiceInput.stopListening()
        services.voiceOutput.stop()
        listening = false
    }

    private fun addMessage(message: ChatMessage): ChatMessage {
        _messages.value = _messages.value + message
        return message
    }

    private fun updateMessage(id: String, transform: (ChatMessage) -> ChatMessage) {
        _messages.value = _messages.value.map {
            if (it.id == id) transform(it) else it
        }
    }

    override fun onCleared() {
        stopVoice()
        super.onCleared()
    }
}

@Composable
fun ChatScreen(vm: ChatViewModel) {
    val messages by vm.messages.collectAsState()
    val listState = rememberLazyListState()
    val context = LocalContext.current

    // Chat is only reachable when paired: keep the background event
    // stream alive so approval/task notifications arrive.
    LaunchedEffect(Unit) {
        runCatching { AfnanSyncService.start(context) }
    }

    var micAllowed by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context, Manifest.permission.RECORD_AUDIO,
            ) == PackageManager.PERMISSION_GRANTED,
        )
    }
    val micPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        micAllowed = granted
        if (granted) vm.toggleListening()
    }

    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) listState.animateScrollToItem(messages.size - 1)
    }

    Column(Modifier.fillMaxSize()) {
        // Wordmark header.
        Surface(
            tonalElevation = 2.dp,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(
                "Afnan",
                style = MaterialTheme.typography.titleLarge.copy(
                    fontWeight = FontWeight.Bold,
                ),
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.padding(horizontal = 20.dp, vertical = 12.dp),
            )
        }

        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(horizontal = 12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            item { Spacer(Modifier.height(4.dp)) }
            if (messages.isEmpty()) {
                item {
                    EmptyChatHint()
                }
            }
            items(messages, key = { it.id }) { message ->
                ChatBubble(message = message, vm = vm)
            }
            item { Spacer(Modifier.height(4.dp)) }
        }

        vm.voiceError?.let { err ->
            Text(
                err,
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(horizontal = 16.dp),
            )
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            OutlinedTextField(
                value = vm.input,
                onValueChange = vm::onInputChange,
                placeholder = { Text("Ask Afnan… (اردو / English)") },
                modifier = Modifier.weight(1f),
                singleLine = false,
                maxLines = 4,
            )
            Spacer(Modifier.width(8.dp))
            IconButton(
                onClick = {
                    if (micAllowed) {
                        vm.toggleListening()
                    } else {
                        micPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                    }
                },
                enabled = !vm.sending,
            ) {
                Icon(
                    Icons.Filled.Mic,
                    contentDescription = if (vm.listening) "Stop listening" else "Voice input",
                    tint = if (vm.listening) MaterialTheme.colorScheme.error
                    else MaterialTheme.colorScheme.primary,
                )
            }
            IconButton(
                onClick = vm::send,
                enabled = !vm.sending && vm.input.isNotBlank(),
            ) {
                if (vm.sending) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(24.dp),
                        strokeWidth = 2.dp,
                    )
                } else {
                    Icon(
                        Icons.Filled.Send,
                        contentDescription = "Send",
                        tint = MaterialTheme.colorScheme.primary,
                    )
                }
            }
        }
    }
}

@Composable
private fun EmptyChatHint() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer,
        ),
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(
                "Give Afnan a task in your own words.",
                style = MaterialTheme.typography.titleSmall,
            )
            Spacer(Modifier.height(4.dp))
            Text(
                "“Research the latest PTA device tax rules and summarize them.”\n" +
                    "Afnan will plan, execute, ask for approval when needed, " +
                    "and report back with verified results.",
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}

@Composable
private fun ChatBubble(message: ChatMessage, vm: ChatViewModel) {
    when (message.role) {
        ChatRole.USER -> {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                Surface(
                    shape = RoundedCornerShape(16.dp, 4.dp, 16.dp, 16.dp),
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.fillMaxWidth(0.85f),
                ) {
                    Text(
                        message.text,
                        color = MaterialTheme.colorScheme.onPrimary,
                        modifier = Modifier.padding(12.dp),
                    )
                }
            }
        }

        ChatRole.APPROVAL -> {
            ApprovalInlineCard(message = message, vm = vm)
        }

        ChatRole.ERROR -> {
            Card(
                modifier = Modifier.fillMaxWidth(0.95f),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.errorContainer,
                ),
            ) {
                Row(Modifier.padding(12.dp), verticalAlignment = Alignment.Top) {
                    Icon(
                        Icons.Filled.Warning,
                        contentDescription = "Error",
                        tint = MaterialTheme.colorScheme.error,
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(message.text)
                }
            }
        }

        ChatRole.PROGRESS -> {
            Card(modifier = Modifier.fillMaxWidth(0.95f)) {
                Row(
                    Modifier.padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        strokeWidth = 2.dp,
                    )
                    Spacer(Modifier.width(12.dp))
                    Text(message.text, style = MaterialTheme.typography.bodyMedium)
                }
            }
        }

        ChatRole.SUMMARY -> {
            Card(
                modifier = Modifier.fillMaxWidth(0.95f),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.secondaryContainer,
                ),
            ) {
                Row(Modifier.padding(12.dp), verticalAlignment = Alignment.Top) {
                    Icon(
                        Icons.Filled.Check,
                        contentDescription = "Completed",
                        tint = MaterialTheme.colorScheme.secondary,
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(message.text)
                }
            }
        }

        ChatRole.ASSISTANT -> {
            Card(modifier = Modifier.fillMaxWidth(0.95f)) {
                Text(message.text, modifier = Modifier.padding(12.dp))
            }
        }
    }
}

@Composable
private fun ApprovalInlineCard(message: ChatMessage, vm: ChatViewModel) {
    val approval = message.approval ?: return
    var decided by remember { mutableStateOf(false) }
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
        ),
        elevation = CardDefaults.cardElevation(4.dp),
    ) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    Icons.Filled.Warning,
                    contentDescription = "Approval required",
                    tint = riskColor(approval.risk),
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    "Approval needed",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                )
                Spacer(Modifier.weight(1f))
                Text(
                    approval.risk.name,
                    color = riskColor(approval.risk),
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                )
            }
            Spacer(Modifier.height(8.dp))
            Text(approval.action, fontWeight = FontWeight.SemiBold)
            Text(approval.target, style = MaterialTheme.typography.bodyMedium)
            if (approval.reason.isNotBlank()) {
                Text(
                    approval.reason,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Text(
                "Expires in ${formatCountdown(approval.expiresAt)}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error,
            )
            if (!decided) {
                Spacer(Modifier.height(12.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Button(
                        onClick = {
                            decided = true
                            vm.decideApproval(message, true)
                        },
                        modifier = Modifier.weight(1f),
                    ) { Text("Approve") }
                    OutlinedButton(
                        onClick = {
                            decided = true
                            vm.decideApproval(message, false)
                        },
                        modifier = Modifier.weight(1f),
                    ) { Text("Deny") }
                }
            } else {
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }
        }
    }
}

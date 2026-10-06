package com.afnan.ai.ui.tasks

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
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Replay
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.Card
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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.afnan.ai.ui.AppServices
import com.afnan.ai.ui.TaskStatus
import com.afnan.ai.ui.TaskView
import com.afnan.ai.ui.formatTimeAgo
import com.afnan.ai.ui.theme.taskStatusColor
import com.afnan.ai.ui.viewModelFactory
import kotlinx.coroutines.launch

class TasksViewModel(private val services: AppServices) : ViewModel() {
    var tasks by mutableStateOf<List<TaskView>>(emptyList())
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
                tasks = services.repository.listTasks()
            } catch (e: Exception) {
                error = e.message ?: "Couldn't load tasks"
            } finally {
                loading = false
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TasksScreen(
    services: AppServices,
    onTaskClick: (String) -> Unit,
    vm: TasksViewModel = viewModel(factory = viewModelFactory { TasksViewModel(services) }),
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Tasks") },
                actions = {
                    IconButton(onClick = vm::refresh, enabled = !vm.loading) {
                        Icon(Icons.Filled.Refresh, contentDescription = "Refresh tasks")
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
            if (vm.loading && vm.tasks.isEmpty()) {
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
            if (vm.tasks.isEmpty() && !vm.loading) {
                EmptyState("No tasks yet", "Send Afnan a task from the Chat tab.")
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    items(vm.tasks, key = { it.id }) { task ->
                        TaskRow(task = task, onClick = { onTaskClick(task.id) })
                    }
                }
            }
        }
    }
}

@Composable
fun TaskRow(task: TaskView, onClick: () -> Unit) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp)
            .clickable(onClick = onClick),
    ) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    task.title,
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.weight(1f),
                )
                StatusChip(task.status)
            }
            Spacer(Modifier.height(8.dp))
            LinearProgressIndicator(
                progress = task.progress.coerceIn(0f, 1f),
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(4.dp))
            Text(
                "${(task.progress * 100).toInt()}% • ${formatTimeAgo(task.updatedAt)}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
fun StatusChip(status: TaskStatus) {
    val color = taskStatusColor(status)
    Surface(
        color = color.copy(alpha = 0.15f),
        shape = MaterialTheme.shapes.small,
    ) {
        Text(
            status.name.lowercase().replaceFirstChar { it.uppercase() },
            color = color,
            style = MaterialTheme.typography.labelMedium,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
        )
    }
}

@Composable
fun EmptyState(title: String, hint: String) {
    Column(
        Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(title, style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(8.dp))
        Text(
            hint,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

// ---------------------------------------------------------------------------
// Detail
// ---------------------------------------------------------------------------

class TaskDetailViewModel(private val services: AppServices) : ViewModel() {
    var task by mutableStateOf<TaskView?>(null)
        private set
    var loading by mutableStateOf(false)
        private set
    var error by mutableStateOf<String?>(null)
        private set
    var acting by mutableStateOf(false)
        private set

    fun load(taskId: String) {
        viewModelScope.launch {
            loading = true
            error = null
            try {
                task = services.repository.getTask(taskId)
            } catch (e: Exception) {
                error = e.message ?: "Couldn't load task"
            } finally {
                loading = false
            }
        }
    }

    private fun act(block: suspend (String) -> TaskView) {
        val id = task?.id ?: return
        viewModelScope.launch {
            acting = true
            error = null
            try {
                task = block(id)
            } catch (e: Exception) {
                error = e.message ?: "Action failed"
            } finally {
                acting = false
            }
        }
    }

    fun pause() = act(services.repository::pauseTask)
    fun resume() = act(services.repository::resumeTask)
    fun cancel() = act(services.repository::cancelTask)
    fun retry() = act(services.repository::retryTask)
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TaskDetailScreen(
    services: AppServices,
    taskId: String,
    onBack: () -> Unit,
    vm: TaskDetailViewModel = viewModel(factory = viewModelFactory { TaskDetailViewModel(services) }),
) {
    LaunchedEffect(taskId) { vm.load(taskId) }
    val scope = rememberCoroutineScope()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Task") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    IconButton(onClick = { vm.load(taskId) }) {
                        Icon(Icons.Filled.Refresh, contentDescription = "Refresh task")
                    }
                },
            )
        },
    ) { padding ->
        val task = vm.task
        Column(
            Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
        ) {
            when {
                vm.loading && task == null -> {
                    Row(
                        Modifier.fillMaxSize(),
                        horizontalArrangement = Arrangement.Center,
                        verticalAlignment = Alignment.CenterVertically,
                    ) { CircularProgressIndicator() }
                }

                task != null -> {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            task.title,
                            style = MaterialTheme.typography.headlineSmall,
                            modifier = Modifier.weight(1f),
                        )
                        StatusChip(task.status)
                    }
                    Spacer(Modifier.height(12.dp))
                    LinearProgressIndicator(
                        progress = task.progress.coerceIn(0f, 1f),
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Spacer(Modifier.height(4.dp))
                    Text("${(task.progress * 100).toInt()}% complete")
                    Spacer(Modifier.height(12.dp))
                    task.resultSummary?.let {
                        Text("Result", fontWeight = FontWeight.Bold)
                        Text(it)
                        Spacer(Modifier.height(8.dp))
                    }
                    task.error?.let {
                        Text(
                            "Error: $it",
                            color = MaterialTheme.colorScheme.error,
                        )
                        Spacer(Modifier.height(8.dp))
                    }
                    Text(
                        "Updated ${formatTimeAgo(task.updatedAt)}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(Modifier.height(24.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        when (task.status) {
                            TaskStatus.RUNNING -> {
                                OutlinedButton(
                                    onClick = vm::pause,
                                    enabled = !vm.acting,
                                ) {
                                    Icon(Icons.Filled.Pause, contentDescription = null)
                                    Spacer(Modifier.width(4.dp))
                                    Text("Pause")
                                }
                                OutlinedButton(
                                    onClick = vm::cancel,
                                    enabled = !vm.acting,
                                ) {
                                    Icon(Icons.Filled.Stop, contentDescription = null)
                                    Spacer(Modifier.width(4.dp))
                                    Text("Cancel")
                                }
                            }

                            TaskStatus.PAUSED -> {
                                Button(onClick = vm::resume, enabled = !vm.acting) {
                                    Icon(Icons.Filled.PlayArrow, contentDescription = null)
                                    Spacer(Modifier.width(4.dp))
                                    Text("Resume")
                                }
                                OutlinedButton(
                                    onClick = vm::cancel,
                                    enabled = !vm.acting,
                                ) { Text("Cancel") }
                            }

                            TaskStatus.FAILED, TaskStatus.CANCELLED -> {
                                Button(onClick = vm::retry, enabled = !vm.acting) {
                                    Icon(Icons.Filled.Replay, contentDescription = null)
                                    Spacer(Modifier.width(4.dp))
                                    Text("Retry")
                                }
                            }

                            else -> Unit
                        }
                    }
                    if (vm.acting) {
                        Spacer(Modifier.height(12.dp))
                        LinearProgressIndicator(Modifier.fillMaxWidth())
                    }
                }
            }
            vm.error?.let {
                Spacer(Modifier.height(8.dp))
                Text(it, color = MaterialTheme.colorScheme.error)
                Spacer(Modifier.height(4.dp))
                Button(onClick = { scope.launch { vm.load(taskId) } }) {
                    Text("Retry")
                }
            }
        }
    }
}

package com.afnan.ai.ui.feed

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
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.afnan.ai.ui.ActivityItem
import com.afnan.ai.ui.AppServices
import com.afnan.ai.ui.formatTimeAgo
import com.afnan.ai.ui.tasks.EmptyState
import com.afnan.ai.ui.viewModelFactory
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class FeedViewModel(private val services: AppServices) : ViewModel() {
    var items by mutableStateOf<List<ActivityItem>>(emptyList())
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
                items = services.repository.queryActivity(100)
            } catch (e: Exception) {
                error = e.message ?: "Couldn't load activity"
            } finally {
                loading = false
            }
        }
    }
}

private fun iconFor(category: String): Pair<ImageVector, String> = when {
    category.startsWith("task.completed") -> Icons.Filled.CheckCircle to "Completed"
    category.startsWith("task.failed") -> Icons.Filled.Error to "Failed"
    category.startsWith("task.started") -> Icons.Filled.PlayArrow to "Started"
    category.startsWith("approval") -> Icons.Filled.Warning to "Approval"
    else -> Icons.Filled.Info to "Activity"
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FeedScreen(
    services: AppServices,
    vm: FeedViewModel = viewModel(factory = viewModelFactory { FeedViewModel(services) }),
) {
    val timeFormat = rememberTimeFormat()
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Activity") },
                actions = {
                    IconButton(onClick = vm::refresh, enabled = !vm.loading) {
                        Icon(Icons.Filled.Refresh, contentDescription = "Refresh activity")
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
            if (vm.loading && vm.items.isEmpty()) {
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
            if (vm.items.isEmpty() && !vm.loading) {
                EmptyState("Nothing yet", "Afnan's activity will appear here.")
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    items(vm.items, key = { it.id }) { item ->
                        val (icon, desc) = iconFor(item.category)
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 12.dp),
                        ) {
                            Row(
                                Modifier.padding(16.dp),
                                verticalAlignment = Alignment.Top,
                            ) {
                                Icon(
                                    icon,
                                    contentDescription = desc,
                                    tint = MaterialTheme.colorScheme.primary,
                                )
                                Spacer(Modifier.width(12.dp))
                                Column(Modifier.weight(1f)) {
                                    Text(
                                        item.title,
                                        style = MaterialTheme.typography.titleSmall,
                                    )
                                    if (item.detail.isNotBlank()) {
                                        Spacer(Modifier.height(2.dp))
                                        Text(
                                            item.detail,
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        )
                                    }
                                    Spacer(Modifier.height(4.dp))
                                    Text(
                                        "${item.category} • ${
                                            timeFormat.format(Date(item.at))
                                        } (${formatTimeAgo(item.at)})",
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

@Composable
private fun rememberTimeFormat(): SimpleDateFormat {
    return androidx.compose.runtime.remember {
        SimpleDateFormat("MMM d, HH:mm", Locale.getDefault())
    }
}

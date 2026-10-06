package com.afnan.ai.ui.navigation

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Approval
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.MoreHoriz
import androidx.compose.material.icons.filled.Public
import androidx.compose.material.icons.filled.TaskAlt
import androidx.compose.material3.Badge
import androidx.compose.material3.BadgedBox
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.afnan.ai.ui.AppServices
import com.afnan.ai.ui.approvals.ApprovalDetailScreen
import com.afnan.ai.ui.approvals.ApprovalsScreen
import com.afnan.ai.ui.approvals.ApprovalsViewModel
import com.afnan.ai.ui.browser.BrowserScreen
import com.afnan.ai.ui.chat.ChatScreen
import com.afnan.ai.ui.chat.ChatViewModel
import com.afnan.ai.ui.devices.DevicesScreen
import com.afnan.ai.ui.feed.FeedScreen
import com.afnan.ai.ui.goals.GoalsScreen
import com.afnan.ai.ui.library.LibraryScreen
import com.afnan.ai.ui.more.MoreScreen
import com.afnan.ai.ui.pairing.PairingScreen
import com.afnan.ai.ui.settings.AboutScreen
import com.afnan.ai.ui.settings.ConnectorsScreen
import com.afnan.ai.ui.settings.CredentialsScreen
import com.afnan.ai.ui.settings.DataControlsScreen
import com.afnan.ai.ui.settings.PermissionsScreen
import com.afnan.ai.ui.settings.SettingsScreen
import com.afnan.ai.ui.setup.SetupScreen
import com.afnan.ai.ui.splash.SplashScreen
import com.afnan.ai.ui.tasks.TaskDetailScreen
import com.afnan.ai.ui.tasks.TasksScreen
import com.afnan.ai.ui.viewModelFactory

object Routes {
    const val SPLASH = "splash"
    const val SETUP = "setup"
    const val PAIRING = "pairing"
    const val CHAT = "chat"
    const val TASKS = "tasks"
    const val TASK_DETAIL = "task_detail/{taskId}"
    const val GOALS = "goals"
    const val APPROVALS = "approvals"
    const val APPROVAL_DETAIL = "approval_detail/{approvalId}"
    const val BROWSER = "browser"
    const val DEVICES = "devices"
    const val FEED = "feed"
    const val LIBRARY = "library"
    const val MORE = "more"
    const val SETTINGS = "settings"
    const val CONNECTORS = "connectors"
    const val CREDENTIALS = "credentials"
    const val PERMISSIONS = "permissions"
    const val DATA_CONTROLS = "data_controls"
    const val ABOUT = "about"

    fun taskDetail(taskId: String) = "task_detail/$taskId"
    fun approvalDetail(approvalId: String) = "approval_detail/$approvalId"
}

private data class BottomItem(
    val route: String,
    val label: String,
    val icon: ImageVector,
)

private val bottomItems = listOf(
    BottomItem(Routes.CHAT, "Chat", Icons.Filled.Chat),
    BottomItem(Routes.TASKS, "Tasks", Icons.Filled.TaskAlt),
    BottomItem(Routes.APPROVALS, "Approvals", Icons.Filled.Approval),
    BottomItem(Routes.BROWSER, "Browser", Icons.Filled.Public),
    BottomItem(Routes.MORE, "More", Icons.Filled.MoreHoriz),
)

/**
 * App scaffold: bottom navigation on top-level destinations, full-screen
 * NavHost everywhere else.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AfnanScaffold(
    services: AppServices,
    deepLink: String?,
    onDeepLinkConsumed: () -> Unit,
) {
    val navController = rememberNavController()
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = backStackEntry?.destination?.route
    val showBottomBar = currentRoute in bottomItems.map { it.route }

    LaunchedEffect(deepLink) {
        if (deepLink != null) {
            // Deep links only make sense once paired; otherwise the
            // splash flow (setup/pairing) takes precedence.
            if (!services.tokenStore.getToken().isNullOrBlank()) {
                navController.navigate(deepLink) {
                    launchSingleTop = true
                }
            }
            onDeepLinkConsumed()
        }
    }

    Scaffold(
        bottomBar = {
            if (showBottomBar) {
                NavigationBar {
                    bottomItems.forEach { item ->
                        val selected = currentRoute == item.route
                        NavigationBarItem(
                            selected = selected,
                            onClick = {
                                navController.navigate(item.route) {
                                    popUpTo(Routes.CHAT) { saveState = true }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            },
                            icon = {
                                if (item.route == Routes.APPROVALS) {
                                    val vm: ApprovalsViewModel = viewModel(
                                        factory = viewModelFactory {
                                            ApprovalsViewModel(services.repository)
                                        },
                                    )
                                    BadgedBox(
                                        badge = {
                                            if (vm.pendingCount > 0) {
                                                Badge { Text("${vm.pendingCount}") }
                                            }
                                        },
                                    ) {
                                        Icon(item.icon, contentDescription = item.label)
                                    }
                                } else {
                                    Icon(item.icon, contentDescription = item.label)
                                }
                            },
                            label = { Text(item.label) },
                        )
                    }
                }
            }
        },
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Routes.SPLASH,
            modifier = Modifier.padding(innerPadding),
        ) {
            composable(Routes.SPLASH) {
                SplashScreen(
                    services = services,
                    onNavigate = { route ->
                        navController.navigate(route) {
                            popUpTo(Routes.SPLASH) { inclusive = true }
                        }
                    },
                )
            }
            composable(Routes.SETUP) {
                SetupScreen(
                    services = services,
                    onDone = {
                        navController.navigate(Routes.PAIRING) {
                            popUpTo(Routes.SETUP) { inclusive = true }
                        }
                    },
                )
            }
            composable(Routes.PAIRING) {
                PairingScreen(
                    services = services,
                    onPaired = {
                        navController.navigate(Routes.CHAT) {
                            popUpTo(Routes.PAIRING) { inclusive = true }
                        }
                    },
                )
            }
            composable(Routes.CHAT) {
                val vm: ChatViewModel = viewModel(
                    factory = viewModelFactory { ChatViewModel(services) },
                )
                ChatScreen(vm = vm)
            }
            composable(Routes.TASKS) {
                TasksScreen(
                    services = services,
                    onTaskClick = { navController.navigate(Routes.taskDetail(it)) },
                )
            }
            composable(
                Routes.TASK_DETAIL,
                arguments = listOf(navArgument("taskId") { type = NavType.StringType }),
            ) { entry ->
                TaskDetailScreen(
                    services = services,
                    taskId = entry.arguments?.getString("taskId").orEmpty(),
                    onBack = { navController.popBackStack() },
                )
            }
            composable(Routes.GOALS) {
                GoalsScreen(services = services)
            }
            composable(Routes.APPROVALS) {
                ApprovalsScreen(
                    services = services,
                    onApprovalClick = { navController.navigate(Routes.approvalDetail(it)) },
                )
            }
            composable(
                Routes.APPROVAL_DETAIL,
                arguments = listOf(navArgument("approvalId") { type = NavType.StringType }),
            ) { entry ->
                ApprovalDetailScreen(
                    services = services,
                    approvalId = entry.arguments?.getString("approvalId").orEmpty(),
                    onBack = { navController.popBackStack() },
                    onDecided = { navController.popBackStack() },
                )
            }
            composable(Routes.BROWSER) {
                BrowserScreen(services = services)
            }
            composable(Routes.DEVICES) {
                DevicesScreen(services = services)
            }
            composable(Routes.FEED) {
                FeedScreen(services = services)
            }
            composable(Routes.LIBRARY) {
                LibraryScreen(services = services)
            }
            composable(Routes.MORE) {
                MoreScreen(onNavigate = { navController.navigate(it) })
            }
            composable(Routes.SETTINGS) {
                SettingsScreen(
                    services = services,
                    onNavigate = { navController.navigate(it) },
                )
            }
            composable(Routes.CONNECTORS) { ConnectorsScreen(services = services) }
            composable(Routes.CREDENTIALS) { CredentialsScreen(services = services) }
            composable(Routes.PERMISSIONS) { PermissionsScreen() }
            composable(Routes.DATA_CONTROLS) {
                DataControlsScreen(
                    services = services,
                    onSignedOut = {
                        navController.navigate(Routes.SETUP) {
                            popUpTo(Routes.CHAT) { inclusive = true }
                        }
                    },
                )
            }
            composable(Routes.ABOUT) { AboutScreen() }
        }
    }
}

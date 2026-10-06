package com.afnan.ai.ui.theme

import android.app.Activity
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.afnan.ai.AfnanApp

// Deep teal + warm amber — the "Afnan" wordmark feel.
private val TealPrimary = Color(0xFF00695C)
private val TealOnPrimary = Color(0xFFFFFFFF)
private val TealContainer = Color(0xFF9CF2E3)
private val TealOnContainer = Color(0xFF00201C)

private val AmberSecondary = Color(0xFF8A5A00)
private val AmberOnSecondary = Color(0xFFFFFFFF)
private val AmberContainer = Color(0xFFFFDDB0)
private val AmberOnContainer = Color(0xFF2B1700)

private val TealPrimaryDark = Color(0xFF7EDBCA)
private val TealOnPrimaryDark = Color(0xFF00332D)
private val TealContainerDark = Color(0xFF005047)
private val TealOnContainerDark = Color(0xFF9CF2E3)

private val AmberSecondaryDark = Color(0xFFFFB300)
private val AmberOnSecondaryDark = Color(0xFF2B1700)
private val AmberContainerDark = Color(0xFF5A3A00)
private val AmberOnContainerDark = Color(0xFFFFDDB0)

private val LightColors = lightColorScheme(
    primary = TealPrimary,
    onPrimary = TealOnPrimary,
    primaryContainer = TealContainer,
    onPrimaryContainer = TealOnContainer,
    secondary = AmberSecondary,
    onSecondary = AmberOnSecondary,
    secondaryContainer = AmberContainer,
    onSecondaryContainer = AmberOnContainer,
    tertiary = AmberSecondary,
)

private val DarkColors = darkColorScheme(
    primary = TealPrimaryDark,
    onPrimary = TealOnPrimaryDark,
    primaryContainer = TealContainerDark,
    onPrimaryContainer = TealOnContainerDark,
    secondary = AmberSecondaryDark,
    onSecondary = AmberOnSecondaryDark,
    secondaryContainer = AmberContainerDark,
    onSecondaryContainer = AmberOnContainerDark,
    tertiary = AmberSecondaryDark,
)

enum class ThemeMode { SYSTEM, LIGHT, DARK }

@Composable
fun AfnanTheme(
    app: AfnanApp,
    content: @Composable () -> Unit,
) {
    val mode by app.themeMode.collectAsStateWithLifecycle()
    val darkTheme = when (mode) {
        ThemeMode.LIGHT -> false
        ThemeMode.DARK -> true
        ThemeMode.SYSTEM -> isSystemInDarkTheme()
    }
    val colorScheme = if (darkTheme) DarkColors else LightColors
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = colorScheme.primary.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = !darkTheme
        }
    }
    MaterialTheme(
        colorScheme = colorScheme,
        content = content,
    )
}

/** Risk chip color used across approvals / tasks. */
@Composable
fun riskColor(risk: com.afnan.ai.ui.RiskLevel): Color = when (risk) {
    com.afnan.ai.ui.RiskLevel.LOW -> Color(0xFF2E7D32)
    com.afnan.ai.ui.RiskLevel.MEDIUM -> Color(0xFFEF6C00)
    com.afnan.ai.ui.RiskLevel.HIGH -> Color(0xFFC62828)
    com.afnan.ai.ui.RiskLevel.CRITICAL -> Color(0xFF6A1B9A)
}

/** Risk chip color from the server's risk string ("low"/"medium"/"high"/"critical"). */
@Composable
fun riskColor(risk: String): Color = when (risk.lowercase()) {
    "low" -> Color(0xFF2E7D32)
    "medium" -> Color(0xFFEF6C00)
    "high" -> Color(0xFFC62828)
    "critical" -> Color(0xFF6A1B9A)
    else -> Color(0xFF757575)
}

@Composable
fun taskStatusColor(status: com.afnan.ai.ui.TaskStatus): Color = when (status) {
    com.afnan.ai.ui.TaskStatus.PENDING -> Color(0xFF757575)
    com.afnan.ai.ui.TaskStatus.RUNNING -> Color(0xFF1565C0)
    com.afnan.ai.ui.TaskStatus.PAUSED -> Color(0xFFEF6C00)
    com.afnan.ai.ui.TaskStatus.COMPLETED -> Color(0xFF2E7D32)
    com.afnan.ai.ui.TaskStatus.FAILED -> Color(0xFFC62828)
    com.afnan.ai.ui.TaskStatus.CANCELLED -> Color(0xFF616161)
}

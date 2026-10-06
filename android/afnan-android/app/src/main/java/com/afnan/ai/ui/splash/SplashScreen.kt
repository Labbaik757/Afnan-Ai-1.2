package com.afnan.ai.ui.splash

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.afnan.ai.ui.AppServices
import com.afnan.ai.ui.navigation.Routes
import kotlinx.coroutines.delay

/**
 * First screen. Routes based on real local state:
 * no server URL -> setup, no token -> pairing, else chat.
 */
@Composable
fun SplashScreen(
    services: AppServices,
    onNavigate: (String) -> Unit,
) {
    LaunchedEffect(Unit) {
        delay(700)
        val destination = when {
            services.tokenStore.getBaseUrl().isBlank() -> Routes.SETUP
            services.tokenStore.getToken().isNullOrBlank() -> Routes.PAIRING
            else -> Routes.CHAT
        }
        onNavigate(destination)
    }
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center,
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = "Afnan",
                style = MaterialTheme.typography.displayLarge.copy(
                    fontWeight = FontWeight.Bold,
                    fontSize = 56.sp,
                ),
                color = MaterialTheme.colorScheme.primary,
            )
            Text(
                text = "AI",
                style = MaterialTheme.typography.headlineMedium.copy(
                    fontWeight = FontWeight.Light,
                    letterSpacing = 12.sp,
                ),
                color = MaterialTheme.colorScheme.secondary,
            )
            Spacer(Modifier.height(32.dp))
            CircularProgressIndicator()
        }
    }
}

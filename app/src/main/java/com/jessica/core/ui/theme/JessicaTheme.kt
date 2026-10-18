package com.jessica.core.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val JessicaLight = lightColorScheme(
    primary = Color(0xFF6952A4),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFECE1FF),
    onPrimaryContainer = Color(0xFF271749),
    background = Color(0xFFFDF8FF),
    onBackground = Color(0xFF211D27),
    surface = Color(0xFFFDF8FF),
    onSurface = Color(0xFF211D27),
    surfaceVariant = Color(0xFFE9E3ED),
    onSurfaceVariant = Color(0xFF5C5663)
)

private val JessicaDark = darkColorScheme(
    primary = Color(0xFFD3BBFF),
    onPrimary = Color(0xFF3B286C),
    primaryContainer = Color(0xFF554084),
    onPrimaryContainer = Color(0xFFF0E5FF),
    background = Color(0xFF17131E),
    onBackground = Color(0xFFF1EBF5),
    surface = Color(0xFF17131E),
    onSurface = Color(0xFFF1EBF5),
    surfaceVariant = Color(0xFF383140),
    onSurfaceVariant = Color(0xFFCCC2D1)
)

@Composable
fun JessicaTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = if (isSystemInDarkTheme()) JessicaDark else JessicaLight,
        content = content
    )
}

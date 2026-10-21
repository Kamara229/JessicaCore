package com.jessica.core.ui.language

import androidx.compose.runtime.Composable
import androidx.compose.runtime.compositionLocalOf

val LocalAppLanguage = compositionLocalOf { AppLanguage.RUSSIAN }

fun localeText(language: AppLanguage, russian: String, english: String): String =
    if (language == AppLanguage.ENGLISH) english else russian

@Composable
fun uiText(russian: String, english: String): String =
    localeText(LocalAppLanguage.current, russian, english)

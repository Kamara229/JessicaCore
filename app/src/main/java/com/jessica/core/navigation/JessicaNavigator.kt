package com.jessica.core.navigation

import androidx.compose.runtime.mutableStateOf

/** The conversation is the home of Jessica, not the old diagnostics screen. */
class JessicaNavigator {
    val currentPage = mutableStateOf(JessicaPage.CHAT)

    fun navigateTo(page: JessicaPage) {
        currentPage.value = page
    }

    fun backHome() {
        currentPage.value = JessicaPage.CHAT
    }
}

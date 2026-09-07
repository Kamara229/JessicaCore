package com.jessica.core.modules.chat


/*
 * =========================================================
 * JESSICA CHAT STATE
 * =========================================================
 *
 * Текущее состояние одного диалога.
 *
 * Здесь нет UI-логики.
 *
 * ChatScreen только отображает это состояние.
 */


data class ChatState(

    val messages: List<ChatMessage> =
        emptyList(),

    val inputText: String =
        "",

    val isRunning: Boolean =
        false,

    val error: String? =
        null

)

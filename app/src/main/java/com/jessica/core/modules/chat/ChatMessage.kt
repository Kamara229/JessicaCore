package com.jessica.core.modules.chat


/*
 * =========================================================
 * JESSICA CHAT MESSAGE
 * =========================================================
 *
 * Одно сообщение в диалоге Jessica.
 *
 * UI не должен хранить сообщения
 * в виде отдельных случайных String.
 *
 * Эта модель станет общей основой:
 *
 * - отображения чата;
 * - истории;
 * - копирования;
 * - повторных запросов;
 * - сохранения диалогов.
 */


enum class ChatMessageRole {

    USER,
    JESSICA,
    SYSTEM

}


data class ChatMessage(

    val id: String,

    val role: ChatMessageRole,

    val text: String,

    val timestamp: Long =
        System.currentTimeMillis(),

    val isError: Boolean =
        false

)

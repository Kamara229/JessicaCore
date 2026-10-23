package com.jessica.core.modules.codex.chat

/**
 * Privacy-safe reason codes: never send user prompts, ChatGPT text, device IDs,
 * exception messages or draft text to diagnostics.
 */
internal object CodexStopReason {
    fun classify(message: String): String {
        val x = message.lowercase()
        return when {
            x.contains("пользовател") || x.contains("немедленно остановлен") -> "owner_stop"
            x.contains("переписка завершена") || x.contains("план заверш") -> "plan_finished"
            x.contains("предел длины") -> "chat_limit"
            x.contains("предел сообщений") || x.contains("время сессии") -> "session_limit"
            x.contains("отключил службу") -> "service_disconnected"
            x.contains("черновик") || x.contains("ввод") || x.contains("отправ") ||
                x.contains("нажатие") -> "send_guard"
            x.contains("ошиб") || x.contains("приостановлен") -> "unhandled_error"
            else -> "other"
        }
    }
}

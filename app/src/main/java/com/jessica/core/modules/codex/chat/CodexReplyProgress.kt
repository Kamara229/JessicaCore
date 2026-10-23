package com.jessica.core.modules.codex.chat

/**
 * Diagnostic-only response phases. A visible marker does not authorize a new
 * message. The separate ChatRunState verifier still requires an exact completed
 * answer, no generating indication, and a stable snapshot.
 */
internal enum class CodexReplyProgress(val phase: String, val status: String) {
    WAITING("awaiting_markers",
        "Жду ответ ChatGPT. Проверяю чат без автоматической прокрутки."),
    STREAMING("reply_streaming",
        "Вижу, что ChatGPT формирует ответ. Читаю поступающий текст."),
    PARTIAL("reply_partial",
        "Вижу ответ ChatGPT. Проверяю его завершение и наличие конечной метки."),
    ANALYZING("reply_detected",
        "Вижу ответ ChatGPT и анализирую."),
    ACCEPTED("reply_verified",
        "Вижу ответ ChatGPT и анализирую. Готовлю следующий шаг.")
}

internal object CodexReplyProgressPolicy {
    fun classify(started: Boolean, completed: Boolean, generating: Boolean,
                 accepted: Boolean): CodexReplyProgress = when {
        accepted -> CodexReplyProgress.ACCEPTED
        completed && !generating -> CodexReplyProgress.ANALYZING
        generating -> CodexReplyProgress.STREAMING
        started -> CodexReplyProgress.PARTIAL
        else -> CodexReplyProgress.WAITING
    }
}

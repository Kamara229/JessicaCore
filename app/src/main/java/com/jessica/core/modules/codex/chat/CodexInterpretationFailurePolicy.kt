package com.jessica.core.modules.codex.chat

import com.jessica.core.modules.codex.relay.RelayInterpretationUnavailable

/**
 * Never retry an inaccessible analysis forever. Report one confirmed outage to
 * ChatGPT without executing a new Jessica test or leaving the selected chat.
 * Technical causes are classified locally; raw HTTP/model errors stay private.
 */
internal object CodexInterpretationFailurePolicy {
    fun isQuota(error: Throwable): Boolean {
        val cause = error.cause ?: error
        val reason = StringBuilder().append(cause.message.orEmpty()).toString()
        return Regex("""\b429\b|rate[_ -]?limit|quota|лимит|квот""",
            RegexOption.IGNORE_CASE).containsMatchIn(reason)
    }

    fun handoffNow(error: Throwable, attempts: Int): Boolean =
        attempts >= 2 || isQuota(error)

    fun report(error: Throwable): String {
        val reason = if (isQuota(error)) "лимит AI-провайдера (HTTP 429)"
            else "ошибка или недоступность сервиса интерпретации"
        return "Кодекс Jessica: НЕ смогла проанализировать последнее сообщение ChatGPT: $reason. " +
            "Если там запрошен тест, он НЕ выполнялся; автоматического повтора теста не было. " +
            "Это сообщение о сбое, а не результат теста. Передаю его в этот же выбранный чат. " +
            "Проверь backend, Render и Supabase через свои подключения и устрани причину. " +
            "Тест не повторяй без явного запроса."
    }
}

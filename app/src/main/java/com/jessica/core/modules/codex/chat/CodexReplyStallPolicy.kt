package com.jessica.core.modules.codex.chat

/**
 * A ChatGPT answer without the exact JESSICA_REPLY_<nonce> / JESSICA_END_<nonce>
 * pair must NEVER be interpreted as a successful relay turn. Previously this
 * left a 24-hour selected-chat session at "awaiting_markers" after one message.
 *
 * Distinguish a genuinely slow/streaming answer and a hidden ChatGPT window
 * from a verifiably stalled visible turn. This policy only returns a safe
 * terminal DIAGNOSTIC; it never clicks, retries, navigates, executes a Jessica
 * test or accepts an unmarked answer.
 */
internal object CodexReplyStallPolicy {
    const val NO_MARKERS_MS = 120_000L
    const val INCOMPLETE_END_MS = 180_000L
    const val GENERATING_MS = 10L * 60_000L

    enum class Failure {
        NONE,
        REPLY_WITHOUT_VERIFIABLE_MARKERS,
        PARTIAL_MARKED_REPLY,
        GENERATION_OVERDUE
    }

    fun classify(
        waiting: Boolean,
        selectedChatVisible: Boolean,
        connected: Boolean,
        generating: Boolean,
        startMarkerObserved: Boolean,
        completed: Boolean,
        elapsedSinceSendMs: Long
    ): Failure {
        if (!waiting || !selectedChatVisible || !connected ||
            completed || elapsedSinceSendMs < 0L) return Failure.NONE

        // Never treat an ordinary slow stream as a malformed response.
        if (generating) return if (elapsedSinceSendMs >= GENERATING_MS)
            Failure.GENERATION_OVERDUE else Failure.NONE

        return when {
            startMarkerObserved && elapsedSinceSendMs >= INCOMPLETE_END_MS ->
                Failure.PARTIAL_MARKED_REPLY
            !startMarkerObserved && elapsedSinceSendMs >= NO_MARKERS_MS ->
                Failure.REPLY_WITHOUT_VERIFIABLE_MARKERS
            else -> Failure.NONE
        }
    }

    fun message(failure: Failure): String = when (failure) {
        Failure.NONE -> ""
        Failure.REPLY_WITHOUT_VERIFIABLE_MARKERS ->
            "Кодекс остановил ожидание: в выбранном чате нет подтверждённого ответа " +
                "с парой меток JESSICA_REPLY / JESSICA_END. ChatGPT мог ответить " +
                "обычным текстом или сообщение не дошло. Проверь чат и запусти " +
                "сессию снова вручную. Повторной отправки и теста не было."
        Failure.PARTIAL_MARKED_REPLY ->
            "Кодекс остановил ожидание: начало ответа JESSICA_REPLY видно, " +
                "но конечная метка JESSICA_END недоступна или ответ обрезан. " +
                "Проверь последний ответ в чате. Новых сообщений и тестов нет."
        Failure.GENERATION_OVERDUE ->
            "Кодекс остановил ожидание: ChatGPT слишком долго показывает " +
                "формирование ответа. Состояние сохранено; проверь чат. " +
                "Повторной отправки и теста нет."
    }
}

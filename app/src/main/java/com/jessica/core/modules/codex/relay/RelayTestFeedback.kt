package com.jessica.core.modules.codex.relay

import com.jessica.core.modules.AIResult

/**
 * Advisory, deterministic completeness check. A backend success flag only
 * means a response arrived; this module never certifies factual correctness
 * and never changes the response returned to ChatGPT.
 */
internal object RelayTestFeedback {
    fun note(task: String, result: AIResult): String {
        if (!result.success) return "Тест не подтверждён: сервер вернул ошибку или не завершил выполнение."
        val q = task.lowercase()
        val a = result.text.lowercase()
        val requestsExplanation = Regex("объясни|поясни|почему|обоснуй|объяснение").containsMatchIn(q)
        if (requestsExplanation && a.isNotBlank()) {
            val actualExplanation = Regex("потому что|поскольку|так как|начальн|конечн|не учитыва|исключен|исключён|включен|включён|с 10 по 23|из молчания текста|в первом содержательном абзаце|далее текст|по тексту лицензии").containsMatchIn(a)
            if (!actualExplanation) return "Ответ получен, но требуемое объяснение явно не обнаружено. Проверь полноту ответа."
        }
        return "Ответ получен; фактическую правильность и полноту оценивает ChatGPT."
    }
}

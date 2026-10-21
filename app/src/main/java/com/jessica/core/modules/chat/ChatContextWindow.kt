package com.jessica.core.modules.chat

/** Only a bounded recent excerpt is sent to Jessica's backend.
 *  The complete archive is kept on the phone.
 */
object ChatContextWindow {
    /** Keep the full previous reply for explicit follow-up translation.
     * Do not silently translate only the final 800 characters.
     */
    fun selectForTask(messages: List<ChatMessage>, task: String): List<ChatMessage> {
        val translate = Regex(
            """перевед|перевести|перевод|translate|tradu|übersetz""",
            RegexOption.IGNORE_CASE
        ).containsMatchIn(task)
        val previous = Regex(
            """свой ответ|твой ответ|последний ответ|предыдущий ответ|ответ выше|your answer|your reply|last answer|previous answer|previous reply|ta réponse|votre réponse|deine antwort""",
            RegexOption.IGNORE_CASE
        ).containsMatchIn(task)
        if (!translate || !previous) return select(messages)
        val latest = messages.lastOrNull { !it.isUser && it.text.isNotBlank() }
            ?: return select(messages)
        // A 10,001-character payload signals overflow to the backend so it
        // asks for a shorter passage instead of returning a partial translation.
        return listOf(latest.copy(text = latest.text.trim().take(10_001)))
    }

    fun select(messages: List<ChatMessage>): List<ChatMessage> {
        val window = mutableListOf<ChatMessage>()
        var budget = 4000
        for (message in messages.asReversed().take(12)) {
            val text = message.text.trim().takeLast(800)
            if (text.isEmpty()) continue
            if (text.length > budget) break
            window.add(message.copy(text = text))
            budget -= text.length
        }
        return window.asReversed()
    }
}

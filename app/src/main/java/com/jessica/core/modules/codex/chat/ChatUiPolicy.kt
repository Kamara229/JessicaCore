package com.jessica.core.modules.codex.chat

/** Sending requires explicit owner selection of the visible chat, regardless of its mode. */
object ChatUiPolicy {
    fun canSend(packageName: String, confirmed: Boolean, locked: Boolean,
                composerText: String, pending: String, sending: Boolean): Boolean =
        packageName == "com.openai.chatgpt" && confirmed && !locked &&
            !sending && (composerText.isBlank() || composerText == pending)

    /**
     * Text-only advisory banner detection. A length notice is NOT an owner
     * stop request and is NOT proof the selected composer cannot still send.
     * Never create a new chat automatically; the sending UI is checked
     * separately for every turn.
     */
    fun conversationAtLimit(content: String): Boolean {
        val normalized = content.replace(Regex("\\s+"), " ").lowercase()
        val russian = normalized.contains("достигли") &&
            normalized.contains("максимальн") &&
            normalized.contains("длин") &&
            normalized.contains("обсуждени")
        val english = normalized.contains("maximum length") &&
            normalized.contains("conversation") &&
            normalized.contains("reached")
        return russian || english
    }

    /** Read-only conversation text only; the composer and its descendants must be excluded. */
    fun alreadyPosted(content: String, pending: String, replyId: String): Boolean {
        if (reply(content.takeLast(60000), replyId) != null) return true
        fun normalized(value: String) = value.replace(Regex("\\s+"), " ").trim()
        return pending.isNotBlank() && normalized(content).contains(normalized(pending))
    }

    fun repairPrompt(sessionId: String, replyId: String, reason: String): String =
        "Исправь формат предыдущего ответа: $reason. Сохрани исходный baseSha, nonce $sessionId и разрешённые пути. Верни полный JSON. " +
            "Для этого нового ответа используй новые внешние маркеры; nonce внутри JSON не меняй. " +
            "Первая отдельная строка: JESSICA_REPLY_$replyId\nПоследняя отдельная строка: JESSICA_END_$replyId\n" +
            "Если корректное исправление невозможно, верни status no_change и files []. Продолжай в этом же выбранном пользователем чате."

    fun reply(text: String, id: String): String? {
        if (text.length > 60000 || !Regex("[a-f0-9-]{36}").matches(id)) return null
        val start = "JESSICA_REPLY_$id"
        val end = "JESSICA_END_$id"
        val lines = text.replace("\r\n", "\n").lines()
        val first = lines.indexOfFirst { it.trim() == start }
        if (first < 0) return null
        val last = (first + 1 until lines.size).firstOrNull { lines[it].trim() == end } ?: return null
        val body = lines.subList(first + 1, last).joinToString("\n").trim()
            .removePrefix("```json").removePrefix("```").removeSuffix("```").trim()
        return body.takeIf { it.startsWith("{") && it.endsWith("}") }
    }
}

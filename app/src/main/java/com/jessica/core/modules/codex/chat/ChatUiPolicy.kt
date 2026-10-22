package com.jessica.core.modules.codex.chat

data class ChatControl(val label: String, val selected: Boolean, val clickable: Boolean)
enum class OrdinaryChatMode { CHAT, WORK, UNKNOWN }

/** Pure, testable rules. Message text is never used as proof of the selected mode. */
object ChatUiPolicy {
    private val chat = setOf("chat", "чат", "ordinary chat", "обычный чат")
    private val work = setOf("work", "работа", "codex", "chatgpt work")
    fun mode(controls: List<ChatControl>): OrdinaryChatMode {
        val selected = controls.filter { it.selected }.map { it.label.trim().lowercase() }
        if (selected.any { it in work }) return OrdinaryChatMode.WORK
        if (selected.any { it in chat }) return OrdinaryChatMode.CHAT
        if (controls.any { it.clickable && it.label.trim().lowercase() in work }) return OrdinaryChatMode.WORK
        return OrdinaryChatMode.UNKNOWN
    }
    fun canSend(packageName: String, mode: OrdinaryChatMode, locked: Boolean,
                composerText: String, pending: String, sending: Boolean): Boolean =
        packageName == "com.openai.chatgpt" && mode == OrdinaryChatMode.CHAT && !locked &&
            !sending && (composerText.isBlank() || composerText == pending)

    fun repairPrompt(sessionId: String, replyId: String, reason: String): String =
        "Исправь формат предыдущего ответа: $reason. Сохрани исходный baseSha, nonce $sessionId и разрешённые пути. Верни полный JSON. " +
            "Для этого нового ответа используй новые внешние маркеры; nonce внутри JSON не меняй. " +
            "Первая отдельная строка: JESSICA_REPLY_$replyId\nПоследняя отдельная строка: JESSICA_END_$replyId\n" +
            "Если корректное исправление невозможно, верни status no_change и files []. Оставайся в обычном чате."

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

package com.jessica.core.modules.chat

/** Only a bounded recent excerpt is sent to Jessica's backend.
 *  The complete archive is kept on the phone.
 */
object ChatContextWindow {
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

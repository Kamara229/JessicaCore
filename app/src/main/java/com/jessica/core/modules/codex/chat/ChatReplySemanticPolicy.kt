package com.jessica.core.modules.codex.chat

import com.jessica.core.modules.codex.relay.ChatReplyExtraction

/**
 * Read-only alternate source for long ChatGPT answers already present in
 * Android's accessibility tree but not visually rendered in the viewport.
 * Never navigates, clicks, scrolls or accepts incomplete/quoted delimiters.
 */
internal object ChatReplySemanticPolicy {
    private val idPattern = Regex("JESSICA_REPLY_([a-f0-9-]{36})")

    fun expectedReplyId(prompt: String?): String? =
        idPattern.find(prompt.orEmpty())?.groupValues?.get(1)

    fun completedNodeText(text: String?, expectedReplyId: String?): String? {
        val raw = text.orEmpty()
        val id = expectedReplyId ?: return null
        if (raw.isBlank() || raw.length > 55_000) return null
        return raw.takeIf { ChatReplyExtraction.read(it,id) != null }
    }
}

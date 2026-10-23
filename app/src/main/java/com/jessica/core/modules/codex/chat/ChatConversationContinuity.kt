package com.jessica.core.modules.codex.chat

import com.jessica.core.modules.codex.relay.ChatReplyExtraction

/** Bound each follow-up message to an already verified reply in the same visible chat. */
object ChatConversationContinuity {
    fun permits(screen: String, previousReplyId: String, acceptedId: String, replyTail: String): Boolean {
        if (previousReplyId.isBlank() || previousReplyId != acceptedId) return false
        if (ChatReplyExtraction.hasStandaloneEnd(screen,previousReplyId)) return true
        val normalized = replyTail.replace(Regex("\\s+"), " ").trim()
        // A short but distinctive verified reply remains usable after
        // Android removes the marker nodes during a notification overlay.
        // IDs must still match; generic acknowledgements are insufficient.
        return normalized.length >= 28 &&
            screen.replace(Regex("\\s+"), " ").contains(normalized)
    }
}

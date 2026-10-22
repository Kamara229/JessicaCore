package com.jessica.core.modules.codex.relay

/**
 * Accumulates an answer only after its unique marker has been observed.
 * ChatGPT may scroll the START marker out of the accessibility tree before
 * the END marker appears. We never infer missing text or reuse old replies.
 */
class ChatReplyCollector {
    private var pendingId = ""
    private var observed = ""

    fun reset() {
        pendingId = ""
        observed = ""
    }

    fun observe(screen: String, id: String): String? {
        if (id != pendingId) { pendingId = id; observed = "" }
        ChatReplyExtraction.read(screen, id)?.let { complete ->
            return complete
        }
        val tail = ChatReplyExtraction.openingTail(screen,id)
        if (tail != null) {
            if (observed.isBlank() || tail.length >= observed.length && tail.startsWith(observed.take(80))) {
                observed = tail.take(55_000)
            } else {
                extend(tail)
            }
        } else if (observed.isNotBlank()) {
            extend(screen)
        }
        return ChatReplyExtraction.read(observed,id)
    }

    private fun extend(snapshot: String) {
        if (observed.length > 54_000) return
        // Long, exact overlap must be present. Otherwise do not concatenate
        // unrelated screen content or guess text lost during a scroll.
        val max = minOf(5000, observed.length, snapshot.length)
        for (overlap in max downTo 60) {
            val part = observed.substring(observed.length - overlap)
            val start = snapshot.indexOf(part)
            if (start >= 0) {
                observed = (observed + snapshot.substring(start + overlap)).take(55_000)
                return
            }
        }
    }
}

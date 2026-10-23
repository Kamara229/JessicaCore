package com.jessica.core.modules.codex.relay

/**
 * Accumulates a reply only after its unique START marker is observed.
 *
 * ChatGPT virtualizes long responses: later accessibility snapshots may show
 * only the middle or end of the answer. Join only on a long, exact overlap;
 * NEVER infer missing text or join unrelated UI content.
 */
class ChatReplyCollector {
    private var pendingId = ""
    private var observed = ""
    private var completed: String? = null

    /** True only after the unique response start marker was observed in this session. */
    fun hasStarted(id: String): Boolean = id == pendingId && observed.isNotBlank()

    fun reset() {
        pendingId = ""
        observed = ""
        completed = null
    }

    fun observe(screen: String, id: String): String? {
        if (id != pendingId) {
            pendingId = id
            observed = ""
            completed = null
        }
        // A response verified once remains available to ChatRunState's
        // six-second stability check even if ChatGPT virtualizes it away.
        completed?.let { return it }

        ChatReplyExtraction.read(screen, id)?.let { complete ->
            completed = complete
            return complete
        }

        val tail = ChatReplyExtraction.openingTail(screen, id)
        if (tail != null) {
            if (observed.isBlank()) {
                observed = tail.take(MAX_OBSERVED)
            } else if (tail.startsWith(observed)) {
                // The visible start grew while ChatGPT was streaming.
                observed = tail.take(MAX_OBSERVED)
            } else {
                extend(tail)
            }
        } else if (observed.isNotBlank()) {
            extend(screen)
        }

        val joined = ChatReplyExtraction.read(observed, id)
        if (joined != null) completed = joined
        return joined
    }

    private fun extend(snapshot: String) {
        if (observed.length >= MAX_OBSERVED || snapshot.isBlank()) return
        val max = minOf(5000, observed.length, snapshot.length)
        if (max < MIN_OVERLAP) return

        // A match at length N guarantees a match at N-1. Binary-search the
        // longest overlap rather than trying only 60/100/240/500/...:
        // UI virtualization may expose an arbitrary 83- or 317-char overlap.
        var low = MIN_OVERLAP
        var high = max
        var overlap = 0
        var matchIndex = -1
        while (low <= high) {
            val mid = (low + high) ushr 1
            val index = snapshot.lastIndexOf(observed.takeLast(mid))
            if (index >= 0) {
                overlap = mid
                matchIndex = index
                low = mid + 1
            } else {
                high = mid - 1
            }
        }
        if (overlap == 0 || matchIndex < 0) return
        observed = (observed + snapshot.substring(matchIndex + overlap)).take(MAX_OBSERVED)
    }

    private companion object {
        const val MAX_OBSERVED = 55_000
        const val MIN_OVERLAP = 80
    }
}

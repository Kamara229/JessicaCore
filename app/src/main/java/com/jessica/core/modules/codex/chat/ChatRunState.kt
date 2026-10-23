package com.jessica.core.modules.codex.chat

/** One explicitly started, bounded session. No automatic retry of an ambiguous send. */
class ChatRunState(val startedAt: Long, val maxTurns: Int = 3, val durationMs: Long = 60 * 60 * 1000L) {
    var sends = 0; private set
    var waiting = false; private set
    var stopped = false; private set
    var sendAt = 0L; private set
    private var stable = ""
    private var stableAt = 0L

    fun beginSend(now: Long): Boolean {
        if (stopped || waiting || sends >= maxTurns || expired(now)) return false
        sends++
        waiting = true
        sendAt = now
        stable = ""
        return true
    }
    fun acceptReply(text: String?, generating: Boolean, now: Long): String? {
        // Exact START/END markers plus a disappeared generation control are
        // stronger completion evidence than a fixed eight-second delay. Keep
        // a short settle interval for Android's asynchronous UI updates.
        if (!waiting || stopped || generating || text == null || now - sendAt < 1500) {
            stable = ""; return null
        }
        if (text != stable) { stable = text; stableAt = now; return null }
        if (now - stableAt < 2500) return null
        waiting = false
        stable = ""
        return text
    }
    // A delayed response does not silently terminate a long-running Codex session.
    // Never retry the corresponding uncertain send.
    fun replyOverdue(now: Long) = waiting && now - sendAt >= 45_000L
    fun expired(now: Long) = now - startedAt >= durationMs
    fun stop() { stopped = true; waiting = false }
}

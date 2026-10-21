package com.jessica.core.modules.codex.chat

/** One user launch, at most three messages. No automatic retry of an ambiguous send. */
class ChatRunState(val startedAt: Long, val maxTurns: Int = 3) {
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
        if (!waiting || stopped || generating || text == null || now - sendAt < 8000) {
            stable = ""; return null
        }
        if (text != stable) { stable = text; stableAt = now; return null }
        if (now - stableAt < 6000) return null
        waiting = false
        stable = ""
        return text
    }
    fun expired(now: Long) = now - startedAt >= 60 * 60 * 1000L || (waiting && now - sendAt >= 20 * 60 * 1000L)
    fun stop() { stopped = true; waiting = false }
}

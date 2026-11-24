package com.jessica.core.modules.codex.chat

/**
 * The phone relay must NEVER treat an ordinary ChatGPT reply as a verified
 * instruction: only this turn's exact START/END markers can do that.
 *
 * If ChatGPT was demonstrably generating a reply but its unique envelope is
 * still absent after generation ended, fail closed with a visible status
 * instead of appearing to hang for the full 24-hour session. No UI action,
 * message resend, inferred instruction, Jessica test, or network access is
 * permitted by this policy.
 *
 * All timestamps use Android's monotonic elapsedRealtime clock.
 */
internal class CodexMissingEnvelopePolicy {
    enum class Decision { WAIT, STOP_UNMARKED, STOP_PARTIAL }

    private var sendAt: Long = -1L
    private var sawGeneration = false
    private var idleSince: Long = -1L

    fun reset() {
        sendAt = -1L
        sawGeneration = false
        idleSince = -1L
    }

    fun observe(
        now: Long,
        currentSendAt: Long,
        generating: Boolean,
        startedMarker: Boolean,
        completedEnvelope: Boolean
    ): Decision {
        if (currentSendAt <= 0L) { reset(); return Decision.WAIT }
        if (sendAt != currentSendAt) {
            reset()
            sendAt = currentSendAt
        }
        if (completedEnvelope) return Decision.WAIT
        if (generating) {
            sawGeneration = true
            idleSince = -1L
            return Decision.WAIT
        }
        // Without seeing a ChatGPT generation indicator, there is no trusted
        // evidence that the assistant has answered this particular turn.
        if (!sawGeneration) return Decision.WAIT
        if (idleSince < 0L) {
            idleSince = now
            return Decision.WAIT
        }
        if (now - currentSendAt < MIN_AGE_AFTER_SEND_MS ||
            now - idleSince < MIN_IDLE_AFTER_STREAM_MS) return Decision.WAIT

        return if (startedMarker) Decision.STOP_PARTIAL
               else Decision.STOP_UNMARKED
    }

    private companion object {
        const val MIN_AGE_AFTER_SEND_MS = 120_000L
        const val MIN_IDLE_AFTER_STREAM_MS = 45_000L
    }
}

package com.jessica.core.modules.codex.chat

/**
 * Accessibility ACTION_CLICK can return false even if ChatGPT posted the
 * message. Once an attempt is counted, neither another click nor a resend is
 * authorized until the owner deliberately starts a new session.
 */
internal object CodexSendConfirmationPolicy {
    enum class Decision { WAIT_FOR_EVIDENCE, PROCESS_REPLY, NONE }
    fun decide(waiting: Boolean, clickAcknowledged: Boolean,
               completedReply: Boolean): Decision = when {
        !waiting -> Decision.NONE
        completedReply -> Decision.PROCESS_REPLY
        !clickAcknowledged -> Decision.WAIT_FOR_EVIDENCE
        else -> Decision.WAIT_FOR_EVIDENCE
    }
}

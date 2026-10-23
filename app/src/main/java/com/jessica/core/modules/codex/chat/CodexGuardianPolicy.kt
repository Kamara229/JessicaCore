package com.jessica.core.modules.codex.chat

/** Pure state classification. This policy NEVER authorizes another send or test. */
internal object CodexGuardianPolicy {
    enum class Alert { NONE, WAITING_REPLY, INTERPRETING, PREPARING, WINDOW_HIDDEN }
    fun evaluate(waiting: Boolean, phase: String, unchangedMs: Long,
                 chatHidden: Boolean, confirmed: Boolean): Alert {
        if (!confirmed || unchangedMs < 30_000L) return Alert.NONE
        if (chatHidden) return Alert.WINDOW_HIDDEN
        return when {
            phase == "interpret_started" || phase == "interpret_retry" -> Alert.INTERPRETING
            phase == "next_ready" || phase == "draft_ready" ||
                phase == "editor_missing" || phase == "send_control_missing" -> Alert.PREPARING
            waiting || phase.startsWith("reply_") || phase == "awaiting_markers" -> Alert.WAITING_REPLY
            else -> Alert.NONE
        }
    }
}

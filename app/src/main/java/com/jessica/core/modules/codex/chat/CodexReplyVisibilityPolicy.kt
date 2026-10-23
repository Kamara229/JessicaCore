package com.jessica.core.modules.codex.chat

/**
 * Diagnostic-only: missing END is not success and does not permit sending.
 * When scrolling is disabled for safety, a long reply may be only partially
 * readable through the visible Android accessibility tree.
 */
internal object CodexReplyVisibilityPolicy {
    fun incompleteOutsideViewport(started: Boolean, completed: Boolean,
                                  generating: Boolean, elapsedMs: Long): Boolean =
        started && !completed && !generating && elapsedMs >= 20_000L
}

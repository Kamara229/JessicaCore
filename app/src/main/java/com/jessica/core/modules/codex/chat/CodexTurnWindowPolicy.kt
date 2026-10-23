package com.jessica.core.modules.codex.chat

/**
 * Window policy separating observation from mutation.
 * Waiting for a reply does not require a composer, which may disappear
 * while ChatGPT generates, shifts focus or renders the keyboard.
 */
internal object CodexTurnWindowPolicy {
    enum class Next { READ_REPLY, WAIT_COMPOSER, PREPARE_SEND }
    fun next(waiting: Boolean, composerVisible: Boolean): Next = when {
        waiting -> Next.READ_REPLY
        !composerVisible -> Next.WAIT_COMPOSER
        else -> Next.PREPARE_SEND
    }
}

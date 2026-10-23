package com.jessica.core.modules.codex.chat

/**
 * Safety lock installed after the real-device sidebar/new-chat incident.
 * A scroll target cannot be proven to be the transcript by geometry alone.
 * Until content-based verification and real-device Stop tests exist, Codex
 * must only observe ChatGPT. Both gestures and accessibility scrolls are off.
 */
internal object CodexScrollSafetyPolicy {
    fun mayScrollAutomatically(): Boolean = false
}

package com.jessica.core.modules.codex.chat

/**
 * The original unrestricted scroll follower and gesture scroller stay locked.
 * A separate tiny idle pulse is allowed ONLY after owner opt-in AND a current
 * turn nonce has been found inside a validated central transcript node.
 * Unknown nodes, stale chats and default/unattended sessions are denied.
 */
internal object CodexScrollSafetyPolicy {
    fun mayScrollAutomatically(
        explicitOwnerOptIn: Boolean = false,
        verifiedTranscriptNode: Boolean = false
    ): Boolean = explicitOwnerOptIn && verifiedTranscriptNode
}

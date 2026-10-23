package com.jessica.core.modules.codex.chat

/**
 * Android may reject or fail to acknowledge ACTION_SET_TEXT during a redraw.
 * Neither result authorizes a send: the next snapshot must verify the draft.
 */
internal object CodexDraftWritePolicy {
    enum class Next { CHECK_SNAPSHOT, RETRY_SNAPSHOT }

    fun next(verifiedExistingDraft: Boolean, setTextAcknowledged: Boolean): Next =
        if (verifiedExistingDraft || setTextAcknowledged) Next.CHECK_SNAPSHOT
        else Next.RETRY_SNAPSHOT
}

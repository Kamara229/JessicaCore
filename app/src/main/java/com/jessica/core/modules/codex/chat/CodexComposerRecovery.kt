package com.jessica.core.modules.codex.chat

/** Recover a single already-filled editor if ChatGPT temporarily reports unusual bounds.
 * Recovering a node is not permission to send. The sender must still compare
 * its full text to the exact pending draft before taking any action.
 */
internal object CodexComposerRecovery {
    fun choose(
        candidates: List<ComposerCandidate>,
        frame: UiBox,
        density: Float,
        preparedDraftPresent: Boolean
    ): Int? {
        val normal = ChatComposerPolicy.choose(candidates, frame, density)
        if (normal != null || !preparedDraftPresent) return normal
        if (candidates.size != 1) return null
        val candidate = candidates.single()
        return candidate.index.takeIf { candidate.editable || candidate.setText }
    }
}

package com.jessica.core.modules.codex.chat

/**
 * The owner-selected conversation must never be changed by Codex.
 * Recognize only the explicitly titled, non-destructive share-link prompt.
 * Generic mentions of sharing inside the transcript are NOT sufficient.
 */
internal object ChatModalRecoveryPolicy {
    fun isShareLinkPrompt(visibleText: String): Boolean {
        val lines = visibleText.lines().map { it.trim().replace(Regex("\\s+"), " ") }
        val title = lines.any {
            it == "Поделиться ссылкой на чат?" ||
                it == "Share a link to this chat?"
        }
        if (!title) return false
        val description = visibleText.lowercase()
        return (description.contains("при этом соз") && description.contains("которой друг")) ||
            (description.contains("anyone with the link") && description.contains("copy"))
    }
}

package com.jessica.core.modules.codex.chat

/** Foreground assessment is metadata-only; this does not inspect other apps. */
internal object ChatForegroundPolicy {
    private const val CHAT = "com.openai.chatgpt"

    fun permits(
        activePackage: String?,
        activeVisible: Boolean,
        chatVisible: Boolean,
        otherApplicationActive: Boolean,
        systemShadeFocused: Boolean,
        inputMethodActive: Boolean,
        chatFocused: Boolean = false
    ): Boolean {
        if (systemShadeFocused || !chatVisible) return false
        if (activePackage == CHAT && activeVisible) return true
        if (otherApplicationActive) return false
        // A notification can leave rootInActiveWindow pointing to SystemUI
        // briefly even after the owner has returned to focused ChatGPT.
        if (chatFocused) return true
        // Accessibility sometimes returns the keyboard as its active root.
        // Only use the ChatGPT application viewport if no other app is active.
        if (inputMethodActive) return true
        return activePackage.isNullOrBlank()
    }
}

package com.jessica.core.modules.codex.chat

/** SAFETY LOCK: never dispatch a swipe into the ChatGPT interface. */
internal object CodexGestureScroller {
    fun scroll(service: ChatGptAccessibilityService, window: ChatWindow,
               backwards: Boolean): Boolean = false
}

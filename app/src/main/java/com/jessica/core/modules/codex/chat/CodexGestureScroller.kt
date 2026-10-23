package com.jessica.core.modules.codex.chat

import android.accessibilityservice.GestureDescription
import android.graphics.Path

/** Fallback only when ChatGPT does not expose accessibility scroll actions. */
internal object CodexGestureScroller {
    fun scroll(service: ChatGptAccessibilityService, window: ChatWindow,
               backwards: Boolean): Boolean {
        // Independent defense in depth, even if a future caller bypasses
        // CodexScrollFollower and invokes gesture fallback directly.
        if (!CodexScrollSafetyPolicy.mayScrollAutomatically()) return false
        if (window.root.packageName?.toString() != "com.openai.chatgpt") return false
        val active = service.rootInActiveWindow ?: return false
        val inChat = active.packageName?.toString() == "com.openai.chatgpt"
        @Suppress("DEPRECATION")
        active.recycle()
        if (!inChat) return false
        val swipe=CodexScrollGesturePolicy.calculate(
            window.root.screenBox(),window.composer?.screenBox(),backwards,
            service.resources.displayMetrics.density
        ) ?: return false
        val path=Path().apply {
            moveTo(swipe.x,swipe.startY)
            lineTo(swipe.x,swipe.endY)
        }
        val gesture=GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path,0L,380L)).build()
        return runCatching { service.dispatchGesture(gesture,null,null) }.getOrDefault(false)
    }
}

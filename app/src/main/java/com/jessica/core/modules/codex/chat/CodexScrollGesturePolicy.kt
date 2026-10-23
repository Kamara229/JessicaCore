package com.jessica.core.modules.codex.chat

internal data class CodexSwipe(val x: Float, val startY: Float, val endY: Float)

/** Avoid toolbars, composer, notifications and system-navigation gesture zones. */
internal object CodexScrollGesturePolicy {
    fun calculate(frame: UiBox, composer: UiBox?, backwards: Boolean, density: Float): CodexSwipe? {
        if (frame.width <= 0 || frame.height <= 0 || density <= 0f) return null
        val top = frame.top + 90f * density
        val bottom = minOf(frame.bottom - 135f * density,
            (composer?.top ?: frame.bottom).toFloat() - 32f * density)
        if (bottom - top < 240f * density) return null
        val upper = top + (bottom-top)*0.22f
        val lower = top + (bottom-top)*0.78f
        return CodexSwipe(frame.left+frame.width*0.53f,
            if (backwards) upper else lower,
            if (backwards) lower else upper)
    }
}

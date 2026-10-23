package com.jessica.core.modules.codex.chat

import android.view.accessibility.AccessibilityNodeInfo
import kotlin.math.max

/**
 * Candidate metadata only. No text, chat title or user identifier is stored.
 * Scrollable elements are not presumed to be the current conversation:
 * candidate must cover the central reading area of the selected ChatGPT.
 */
internal data class CodexScrollCandidate(
    val index: Int,
    val box: UiBox,
    val visible: Boolean,
    val enabled: Boolean,
    val scrollDown: Boolean,
    val scrollForward: Boolean
)

internal object CodexScrollPolicy {
    fun choose(candidates: List<CodexScrollCandidate>, frame: UiBox,
               editor: UiBox?, density: Float): CodexScrollCandidate? {
        if (!density.isFinite() || density <= 0f || frame.width <= 0 || frame.height <= 0) return null
        val minHeight = max(112f * density, frame.height * .22f)
        return candidates.filter { candidate ->
            val b = candidate.box
            val centerX = frame.left + frame.width * .5f
            candidate.visible && candidate.enabled &&
                (candidate.scrollDown || candidate.scrollForward) &&
                b.width >= frame.width * .58f &&
                b.height >= minHeight &&
                b.left <= centerX && b.right >= centerX &&
                b.top < frame.top + frame.height * .6f &&
                b.bottom > frame.top + frame.height * .38f &&
                (editor == null || (
                    // Same horizontal conversation column as the composer.
                    b.right > editor.left + editor.width * .55f &&
                    b.left < editor.right - editor.width * .55f &&
                    b.top < editor.top - 18f * density))
        }.sortedWith(compareByDescending<CodexScrollCandidate> { it.scrollDown }
            // Prefer the innermost substantial conversation viewport, not
            // an outer root that might navigate unrelated UI.
            .thenBy { it.box.width.toLong() * it.box.height.toLong() }
            .thenByDescending { it.index })
            .firstOrNull()
    }
}

/**
 * SAFETY LOCK: automatic scrolling is disabled. Accessibility scroll actions
 * and synthetic gestures could escape the selected chat and block Stop.
 * Restore only after isolating and device-testing the message-list container.
 */
internal class CodexScrollFollower(private val density: Float,
    private val gestureFallback: (ChatWindow, Boolean) -> Boolean = { _, _ -> false }) {
    fun reset() = Unit
    fun recoverStart(window: ChatWindow, now: Long, expectedId: String,
                     hasStarted: Boolean, endVisible: Boolean): Boolean = false
    fun follow(window: ChatWindow, now: Long, expectedReplyId: String): Boolean = false
}

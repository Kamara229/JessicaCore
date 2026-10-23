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
 * Read-only conversation tracking + one accessibility scroll per interval.
 * Never clicks an arrow, edits a composer, changes chats or retries a send.
 * A user swipe may move the viewport; it must not terminate this session.
 */
internal class CodexScrollFollower(private val density: Float,
    private val gestureFallback: (ChatWindow, Boolean) -> Boolean = { _, _ -> false }) {
    private var replyId = ""
    private var lastAttemptAt = 0L
    private var seekingStart = false

    fun reset() { replyId = ""; lastAttemptAt = 0L; seekingStart = false }

    /**
     * When the END marker is already at the bottom before the first poll,
     * the start may have been virtualized out of view. Briefly seek upward
     * through the SAME selected ChatGPT scroll container until the unique
     * START marker has been observed. Then normal downward following resumes.
     *
     * Returns true while start recovery is in progress (even if the current
     * scroll action was unavailable), so the caller does not scroll down again.
     */
    fun recoverStart(window: ChatWindow, now: Long, expectedId: String,
                     hasStarted: Boolean, endVisible: Boolean): Boolean {
        if (expectedId.isBlank() ||
            window.root.packageName?.toString() != "com.openai.chatgpt") return false
        ensureReply(expectedId)
        if (hasStarted) {
            seekingStart = false
            lastAttemptAt = 0L
            return false
        }
        if (endVisible) seekingStart = true
        if (!seekingStart) return false
        performScroll(window, now, backwards = true)
        return true
    }

    private fun ensureReply(expectedId: String) {
        if (replyId != expectedId) {
            replyId = expectedId
            lastAttemptAt = 0L
            seekingStart = false
        }
    }

    fun follow(window: ChatWindow, now: Long, expectedReplyId: String): Boolean {
        if (expectedReplyId.isBlank() ||
            window.root.packageName?.toString() != "com.openai.chatgpt") return false
        ensureReply(expectedReplyId)
        if (seekingStart) return false
        return performScroll(window, now, backwards = false)
    }

    private fun performScroll(window: ChatWindow, now: Long, backwards: Boolean): Boolean {
        if (lastAttemptAt != 0L && now - lastAttemptAt < 2500L) return false
        val nodes = window.nodes
        val direct = if (backwards)
            AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_UP.id
        else AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_DOWN.id
        val generic = if (backwards)
            AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD
        else AccessibilityNodeInfo.ACTION_SCROLL_FORWARD
        val candidates = nodes.mapIndexedNotNull { index, node ->
            if (!node.isVisibleToUser || !node.isEnabled) return@mapIndexedNotNull null
            val actions = node.actionList
            val primary = actions.any { it.id == direct }
            val secondary = actions.any { it.id == generic }
            if (!primary && !secondary) return@mapIndexedNotNull null
            CodexScrollCandidate(index, node.screenBox(), true, true, primary, secondary)
        }
        val selected = CodexScrollPolicy.choose(candidates,window.root.screenBox(),
            window.composer?.screenBox(), density)
        if (selected == null) {
            lastAttemptAt = now
            return gestureFallback(window,backwards)
        }
        lastAttemptAt = now
        val action = if (selected.scrollDown) direct else generic
        return runCatching { nodes[selected.index].performAction(action) }.getOrDefault(false) ||
            gestureFallback(window,backwards)
    }
}

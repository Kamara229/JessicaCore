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
internal class CodexScrollFollower(private val density: Float) {
    private var replyId = ""
    private var lastAttemptAt = 0L

    fun reset() { replyId = ""; lastAttemptAt = 0L }

    fun follow(window: ChatWindow, now: Long, expectedReplyId: String): Boolean {
        if (expectedReplyId.isBlank() ||
            window.root.packageName?.toString() != "com.openai.chatgpt") return false
        if (replyId != expectedReplyId) {
            replyId = expectedReplyId
            lastAttemptAt = 0L
        }
        if (lastAttemptAt != 0L && now - lastAttemptAt < 2500L) return false
        val nodes = window.nodes
        val downAction = AccessibilityNodeInfo.AccessibilityAction.ACTION_SCROLL_DOWN.id
        val forwardAction = AccessibilityNodeInfo.ACTION_SCROLL_FORWARD
        val candidates = nodes.mapIndexedNotNull { index, node ->
            if (!node.isVisibleToUser || !node.isEnabled) return@mapIndexedNotNull null
            val actions = node.actionList
            val down = actions.any { it.id == downAction }
            val forward = actions.any { it.id == forwardAction }
            if (!down && !forward) return@mapIndexedNotNull null
            CodexScrollCandidate(index, node.screenBox(), true, true, down, forward)
        }
        val selected = CodexScrollPolicy.choose(candidates,window.root.screenBox(),
            window.composer?.screenBox(), density) ?: return false
        lastAttemptAt = now
        val action = if (selected.scrollDown) downAction else forwardAction
        return runCatching { nodes[selected.index].performAction(action) }.getOrDefault(false)
    }
}

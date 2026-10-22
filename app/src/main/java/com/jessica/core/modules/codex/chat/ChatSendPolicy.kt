package com.jessica.core.modules.codex.chat

import java.util.Locale
import kotlin.math.abs

data class UiBox(val left: Int, val top: Int, val right: Int, val bottom: Int) {
    val width get() = right - left
    val height get() = bottom - top
    val centerX get() = (left + right) / 2f
    val centerY get() = (top + bottom) / 2f
    fun contains(x: Float, y: Float) = x >= left && x < right && y >= top && y < bottom
}

data class ChatControl(val parent: Int, val labels: List<String>, val resourceId: String,
    val clickable: Boolean, val enabled: Boolean, val visible: Boolean,
    val insideEditor: Boolean, val bounds: UiBox)

/** Resolve semantics across child labels and clickable ancestors, never by screen position alone. */
object ChatSendPolicy {
    private val sends = setOf("send", "send message", "send prompt", "отправить", "отправить сообщение", "отправить запрос")
    private val stops = setOf("stop", "stop generating", "stop response", "остановить", "остановить генерацию", "остановить ответ")
    private val sendIds = setOf("send", "sendbutton", "sendmessagebutton", "sendpromptbutton", "composersendbutton")
    private fun label(value: String) = value.lowercase(Locale.ROOT).trim()
        .replace(Regex("[\\s\\u00a0]+"), " ")
        .replace(Regex("(?:[,;:]?\\s+)(?:button|кнопка)$"), "").trim()
    private fun matches(node: ChatControl, stop: Boolean): Boolean =
        node.labels.any { label(it) in if (stop) stops else sends } || (!stop &&
            node.resourceId.substringAfterLast('/').lowercase(Locale.ROOT).replace(Regex("[-_]"), "") in sendIds)

    fun targets(nodes: List<ChatControl>, editor: Int?, density: Float, stop: Boolean = false): Set<Int> {
        val box = editor?.let { nodes.getOrNull(it)?.bounds }
        if (!stop && box == null) return emptySet()
        fun eligible(index: Int): Boolean {
            val n = nodes[index]; val b = n.bounds
            return n.visible && n.enabled && n.clickable && !n.insideEditor &&
                b.width in 1..(128 * density).toInt() && b.height in 1..(128 * density).toInt() &&
                (stop || (b.centerY >= box!!.top - 12 * density && b.centerY <= box.bottom + 128 * density))
        }
        val found = mutableSetOf<Int>()
        nodes.forEachIndexed { index, node ->
            if (node.visible && node.enabled && !node.insideEditor && matches(node, stop)) {
                var candidate = index
                var depth = 0
                while (candidate in nodes.indices && depth++ < 6) {
                    if (eligible(candidate)) { found.add(candidate); break }
                    candidate = nodes[candidate].parent
                }
            }
        }
        // A labelled button and its labelled clickable icon are one control, not two sends.
        return found.filterTo(mutableSetOf()) { index ->
            var parent = nodes[index].parent
            var nested = false
            var depth = 0
            while (parent in nodes.indices && depth++ < 50) {
                if (parent in found) { nested = true; break }
                parent = nodes[parent].parent
            }
            !nested
        }
    }
}

/** Owner-selected fallback, held only for this session and anchored to the current composer. */
data class CalibratedSendTarget private constructor(
    private val frameWidth: Int, private val editorLeft: Int, private val editorRight: Int,
    private val dx: Float, private val dy: Float
) {
    companion object {
        fun select(frame: UiBox, editor: UiBox, x: Float, y: Float, density: Float): CalibratedSendTarget? {
            if (!inSendArea(frame, editor, x, y, density)) return null
            return CalibratedSendTarget(frame.width, editor.left, editor.right, x - editor.right, y - editor.bottom)
        }
        private fun inSendArea(frame: UiBox, editor: UiBox, x: Float, y: Float, density: Float) =
            editor.width > 0 && editor.height > 0 && frame.contains(x,y) &&
                x >= frame.left + frame.width * .7f &&
                y >= editor.bottom - 96 * density && y <= editor.bottom + 128 * density
    }
    fun point(frame: UiBox, editor: UiBox, density: Float): Pair<Float,Float>? {
        if (abs(frame.width - frameWidth) > 8 * density ||
            abs(editor.left - editorLeft) > 8 * density || abs(editor.right - editorRight) > 8 * density) return null
        val x = editor.right + dx; val y = editor.bottom + dy
        return if (inSendArea(frame,editor,x,y,density)) x to y else null
    }
}

package com.jessica.core.modules.codex.chat

import android.view.accessibility.AccessibilityNodeInfo

/**
 * Requires the CURRENT turn's exact session marker inside the *same*
 * scrollable accessibility subtree. Geometry alone is NEVER sufficient:
 * after an earlier sidebar/new-chat incident, scrolling an unknown node is
 * prohibited even when it visually resembles the transcript.
 */
internal object CodexIdlePulseTarget {
    fun find(window: ChatWindow, replyId: String, density: Float): UiBox? {
        if (!Regex("[a-f0-9-]{36}").matches(replyId) || !density.isFinite() || density <= 0f)
            return null
        val composer = window.composer?.screenBox() ?: return null
        val frame = window.root.screenBox()
        val marker = "JESSICA_END_$replyId"
        // The editor is already excluded by ChatWindowInspector. The same
        // nonce must also be discoverable inside an actual transcript node.
        if (!window.content.contains(marker)) return null
        val centerX = frame.left + frame.width / 2f
        var best: UiBox? = null
        var bestArea = Long.MAX_VALUE
        var visited = 0

        fun eligible(node: AccessibilityNodeInfo): Boolean {
            if (!node.isVisibleToUser || !node.isEnabled || !node.isScrollable) return false
            val box = node.screenBox()
            return box.width >= frame.width * .72f &&
                box.height >= 260f * density &&
                box.left <= centerX && box.right >= centerX &&
                box.top >= frame.top + 48f * density &&
                box.bottom <= composer.top - 12f * density &&
                box.bottom > box.top &&
                box.right > box.left &&
                CodexIdlePulsePolicy.coordinates(box, composer, density) != null
        }

        fun walk(node: AccessibilityNodeInfo, depth: Int): Boolean {
            if (visited++ >= 4000 || depth > 48 || node.isEditable) return false
            var anchored = node.text?.toString()?.contains(marker) == true ||
                node.contentDescription?.toString()?.contains(marker) == true
            for (i in 0 until node.childCount) {
                val child = node.getChild(i) ?: continue
                try {
                    if (walk(child, depth + 1)) anchored = true
                } finally {
                    @Suppress("DEPRECATION")
                    child.recycle()
                }
            }
            if (anchored && eligible(node)) {
                val box = node.screenBox()
                val area = box.width.toLong() * box.height.toLong()
                if (area < bestArea) {
                    bestArea = area
                    best = box
                }
            }
            return anchored
        }

        walk(window.root, 0)
        return best
    }
}

package com.jessica.core.modules.codex.chat

import android.accessibilityservice.AccessibilityService
import android.graphics.Rect
import android.view.accessibility.AccessibilityNodeInfo
import android.view.accessibility.AccessibilityWindowInfo

data class ChatWindow(val root: AccessibilityNodeInfo, val nodes: List<AccessibilityNodeInfo>,
                      val mode: OrdinaryChatMode, val composer: AccessibilityNodeInfo?,
                      val send: AccessibilityNodeInfo?, val generating: Boolean, val content: String) {
    fun release() { nodes.asReversed().forEach { @Suppress("DEPRECATION") it.recycle() } }
}

object ChatWindowReader {
    private val sendLabels = setOf("send", "send message", "send prompt", "отправить", "отправить сообщение", "отправить запрос")
    private val stopLabels = setOf("stop", "stop generating", "stop response", "остановить", "остановить генерацию", "остановить ответ")
    private fun label(node: AccessibilityNodeInfo): String =
        (node.contentDescription?.toString()?.takeIf { it.isNotBlank() } ?: node.text?.toString().orEmpty()).trim()

    fun read(service: AccessibilityService): ChatWindow? {
        val root = service.windows.firstNotNullOfOrNull { window ->
            if (window.type != AccessibilityWindowInfo.TYPE_APPLICATION || !window.isActive) null
            else window.root?.let { node ->
                if (node.packageName?.toString() == "com.openai.chatgpt") node
                else { @Suppress("DEPRECATION") node.recycle(); null }
            }
        } ?: return null
        val nodes = mutableListOf<AccessibilityNodeInfo>()
        fun walk(node: AccessibilityNodeInfo, depth: Int) {
            nodes.add(node)
            if (depth > 40 || nodes.size > 1500) return
            for (i in 0 until node.childCount) node.getChild(i)?.let { walk(it, depth + 1) }
        }
        walk(root, 0)
        val visible = nodes.filter { it.isVisibleToUser }
        val rootBounds = Rect().also { root.getBoundsInScreen(it) }
        // Only top-of-window UI controls establish mode. Words inside a message cannot authorize sends.
        val header = visible.filter { node ->
            val bounds = Rect().also { node.getBoundsInScreen(it) }
            bounds.bottom <= rootBounds.top + rootBounds.height() / 3 && !node.isEditable &&
                (node.isClickable || node.isSelected || node.isCheckable)
        }
        val controls = header.map { ChatControl(label(it), it.isSelected || it.isChecked, it.isClickable) }
        val editors = visible.filter { it.isEditable && it.isEnabled }
        val buttons = visible.filter { it.isClickable && it.isEnabled }
        val content = visible.filter { !it.isEditable && it.text != null }.map { it.text.toString() }
            .distinct().joinToString("\n").takeLast(120000)
        return ChatWindow(root, nodes, ChatUiPolicy.mode(controls), editors.singleOrNull(),
            buttons.filter { label(it).lowercase() in sendLabels }.singleOrNull(),
            buttons.any { label(it).lowercase() in stopLabels }, content)
    }
}

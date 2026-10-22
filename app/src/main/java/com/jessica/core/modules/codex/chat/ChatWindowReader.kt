package com.jessica.core.modules.codex.chat

import android.accessibilityservice.AccessibilityService
import android.view.accessibility.AccessibilityNodeInfo
import android.view.accessibility.AccessibilityWindowInfo

data class ChatWindow(val root: AccessibilityNodeInfo, val nodes: List<AccessibilityNodeInfo>,
                      val composer: AccessibilityNodeInfo?,
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
        val conversation = mutableListOf<String>()
        fun walk(node: AccessibilityNodeInfo, depth: Int, insideEditor: Boolean = false) {
            nodes.add(node)
            val editor = insideEditor || node.isEditable
            if (!editor && node.isVisibleToUser && node.text != null) conversation.add(node.text.toString())
            if (depth > 40 || nodes.size > 1500) return
            for (i in 0 until node.childCount) node.getChild(i)?.let { walk(it, depth + 1, editor) }
        }
        walk(root, 0)
        val visible = nodes.filter { it.isVisibleToUser }
        val editors = visible.filter { it.isEditable && it.isEnabled }
        val buttons = visible.filter { it.isClickable && it.isEnabled }
        val content = conversation.distinct().joinToString("\n").takeLast(120000)
        return ChatWindow(root, nodes, editors.singleOrNull(),
            buttons.filter { label(it).lowercase() in sendLabels }.singleOrNull(),
            buttons.any { label(it).lowercase() in stopLabels }, content)
    }
}

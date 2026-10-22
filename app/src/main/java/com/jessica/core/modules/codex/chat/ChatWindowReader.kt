package com.jessica.core.modules.codex.chat

import android.accessibilityservice.AccessibilityService
import android.graphics.Rect
import android.view.accessibility.AccessibilityNodeInfo
import android.view.accessibility.AccessibilityWindowInfo

fun AccessibilityNodeInfo.screenBox(): UiBox {
    val rect = Rect(); getBoundsInScreen(rect)
    return UiBox(rect.left,rect.top,rect.right,rect.bottom)
}

data class ChatWindow(val root: AccessibilityNodeInfo, val nodes: List<AccessibilityNodeInfo>,
                      val composer: AccessibilityNodeInfo?,
                      val send: AccessibilityNodeInfo?, val generating: Boolean, val content: String) {
    fun release() { nodes.asReversed().forEach { @Suppress("DEPRECATION") it.recycle() } }
}

object ChatWindowReader {
    fun read(service: AccessibilityService): ChatWindow? {
        val root = service.windows.firstNotNullOfOrNull { window ->
            if (window.type != AccessibilityWindowInfo.TYPE_APPLICATION || !window.isActive) null
            else window.root?.let { node ->
                if (node.packageName?.toString() == "com.openai.chatgpt") node
                else { @Suppress("DEPRECATION") node.recycle(); null }
            }
        } ?: return null
        return inspect(root, service.resources.displayMetrics.density)
    }

    /** Root ownership is checked by read(); separated to test actual Android view hierarchies. */
    internal fun inspect(root: AccessibilityNodeInfo, density: Float): ChatWindow {
        val nodes = mutableListOf<AccessibilityNodeInfo>()
        val controls = mutableListOf<ChatControl>()
        val conversation = mutableListOf<String>()
        var truncated = false
        fun walk(node: AccessibilityNodeInfo, parent: Int, depth: Int, insideEditor: Boolean = false) {
            val index = nodes.size
            nodes.add(node)
            val editor = insideEditor || node.isEditable
            val labels = if (editor) emptyList() else listOfNotNull(node.contentDescription?.toString(), node.text?.toString()) +
                node.actionList.mapNotNull { it.label?.toString() }
            controls.add(ChatControl(parent, labels, node.viewIdResourceName.orEmpty(), node.isClickable,
                node.isEnabled, node.isVisibleToUser, editor, node.screenBox()))
            if (!editor && node.isVisibleToUser && node.text != null) conversation.add(node.text.toString())
            if (depth >= 40 && node.childCount > 0) { truncated = true; return }
            for (i in 0 until node.childCount) {
                if (nodes.size >= 1500) { truncated = true; break }
                node.getChild(i)?.let { walk(it, index, depth + 1, editor) }
            }
        }
        walk(root, -1, 0)
        val editor = nodes.indices.filter { nodes[it].isVisibleToUser && nodes[it].isEditable && nodes[it].isEnabled }.singleOrNull()
        val send = if (truncated) null else ChatSendPolicy.targets(controls,editor,density).singleOrNull()
        val content = conversation.distinct().joinToString("\n").takeLast(120000)
        return ChatWindow(root, nodes, if (truncated) null else editor?.let { nodes[it] }, send?.let { nodes[it] },
            ChatSendPolicy.targets(controls,editor,density,stop=true).isNotEmpty(), content)
    }
}

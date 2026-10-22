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
                      val send: AccessibilityNodeInfo?, val generating: Boolean, val content: String,
                      val editorCandidates: Int = 0, val sendCandidates: Int = 0,
                      val composerDiagnostic: String = "") {
    fun release() { nodes.asReversed().forEach { @Suppress("DEPRECATION") it.recycle() } }
}

object ChatWindowReader {
    fun read(service: AccessibilityService, preparedDraft: String? = null): ChatWindow? {
        val applications = service.windows.filter { it.type == AccessibilityWindowInfo.TYPE_APPLICATION }
        // The keyboard or Jessica's own accessibility overlay may hold focus.
        // Do not require ChatGPT to be the `active` window if it is the only
        // visible application. Conversely, never read hidden ChatGPT content
        // while another app is actively in the foreground.
        val roots = applications.mapNotNull { window ->
            window.root?.let { root -> window to root }
        }
        var selected: AccessibilityNodeInfo? = null
        try {
            if (roots.any { (window, root) ->
                window.isActive && root.packageName?.toString() != "com.openai.chatgpt"
            }) return null
            val chosen = roots.filter { (_, root) ->
                root.packageName?.toString() == "com.openai.chatgpt" && root.isVisibleToUser
            }.maxWithOrNull(compareBy<Pair<AccessibilityWindowInfo, AccessibilityNodeInfo>> { it.first.isActive }
                .thenBy { it.first.isFocused })?.second
                ?: service.rootInActiveWindow?.takeIf { it.packageName?.toString() == "com.openai.chatgpt" }
                ?: return null
            selected = chosen
            return inspect(chosen, service.resources.displayMetrics.density, preparedDraft)
        } finally {
            // The returned ChatWindow owns and recycles its selected tree.
            // Every other root snapshot must be recycled even if it belongs
            // to a second ChatGPT application window.
            roots.filter { (_, root) -> root !== selected }
                .forEach { (_, root) -> @Suppress("DEPRECATION") root.recycle() }
        }
    }

    /** Root ownership is checked by read(); separated to test actual Android view hierarchies. */
    internal fun inspect(root: AccessibilityNodeInfo, density: Float, preparedDraft: String? = null): ChatWindow {
        val nodes = mutableListOf<AccessibilityNodeInfo>()
        val controls = mutableListOf<ChatControl>()
        val composers = mutableListOf<ComposerCandidate>()
        val conversation = mutableListOf<String>()
        var truncated = false
        fun walk(node: AccessibilityNodeInfo, parent: Int, depth: Int, insideEditor: Boolean = false) {
            val index = nodes.size
            nodes.add(node)
            val actions = node.actionList
            val canSetText = actions.any { it.id == AccessibilityNodeInfo.ACTION_SET_TEXT }
            val editor = insideEditor || node.isEditable || canSetText
            if (node.isVisibleToUser && node.isEnabled && (node.isEditable || canSetText)) {
                composers.add(ComposerCandidate(index, node.screenBox(), node.isFocused,
                    node.isEditable, canSetText,
                    preparedDraft != null && preparedDraft.isNotBlank() && node.text?.toString() == preparedDraft))
            }
            val descriptions = listOfNotNull(node.contentDescription?.toString(), node.text?.toString(),
                (if (android.os.Build.VERSION.SDK_INT >= 28) node.tooltipText?.toString() else null), node.hintText?.toString())
            val labels = if (editor) emptyList() else descriptions +
                actions.mapNotNull { it.label?.toString() }
            controls.add(ChatControl(parent, labels, node.viewIdResourceName.orEmpty(),
                node.isClickable || actions.any { it.id == AccessibilityNodeInfo.ACTION_CLICK },
                node.isEnabled, node.isVisibleToUser, editor, node.screenBox()))
            if (!editor && node.isVisibleToUser) {
                // Current ChatGPT builds sometimes expose the assistant's reply
                // only via a semantic contentDescription, not the text property.
                // Read descriptions only when they carry our exact session
                // marker; toolbar labels and navigation are never dialogue.
                val visibleText = node.text?.toString().orEmpty()
                if (visibleText.isNotBlank()) conversation.add(visibleText)
                val description = node.contentDescription?.toString().orEmpty()
                if (description.length <= 55_000 &&
                    (description.contains("JESSICA_REPLY_") || description.contains("JESSICA_END_") ||
                        ChatUiPolicy.conversationAtLimit(description)) &&
                    description != visibleText) conversation.add(description)
            }
            if (depth >= 40 && node.childCount > 0) { truncated = true; return }
            for (i in 0 until node.childCount) {
                if (nodes.size >= 1500) { truncated = true; break }
                node.getChild(i)?.let { walk(it, index, depth + 1, editor) }
            }
        }
        walk(root, -1, 0)
        val editor = if (!truncated) ChatComposerPolicy.choose(composers, root.screenBox(), density) else null
        val diagnostic = when {
            truncated -> "дерево недоступно целиком"
            composers.isEmpty() -> "редактор не виден"
            editor != null -> "редактор определён"
            composers.size == 1 -> {
                val candidate = composers.single()
                val bounds = candidate.box
                "одно поле, текст совпадает: ${candidate.matchesPrepared}, " +
                    "фокус: ${candidate.focused}, размер: ${bounds.width}x${bounds.height}"
            }
            else -> "различных полей: ${composers.size}, точных совпадений: " +
                composers.count { it.matchesPrepared }
        }
        val sends = if (truncated) emptySet() else ChatSendPolicy.targets(controls,editor,density)
        val send = if (truncated) null else ChatSendPolicy.uniqueSendTarget(controls,editor,density)
        val content = conversation.joinToString("\n").takeLast(120000)
        return ChatWindow(root, nodes, if (truncated) null else editor?.let { nodes[it] }, send?.let { nodes[it] },
            ChatSendPolicy.targets(controls,editor,density,stop=true).isNotEmpty(), content,
            composers.size, sends.size, diagnostic)
    }
}

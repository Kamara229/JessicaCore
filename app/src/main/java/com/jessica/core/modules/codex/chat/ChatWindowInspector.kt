package com.jessica.core.modules.codex.chat

import android.view.accessibility.AccessibilityNodeInfo

/** Parses ONLY a verified ChatGPT root selected by ChatWindowReader. */
internal object ChatWindowInspector {
    internal fun inspect(root: AccessibilityNodeInfo, density: Float, preparedDraft: String? = null): ChatWindow {
        val nodes = mutableListOf<AccessibilityNodeInfo>()
        val controls = mutableListOf<ChatControl>()
        val composers = mutableListOf<ComposerCandidate>()
        val conversation = mutableListOf<String>()
        val expectedReplyId = ChatReplySemanticPolicy.expectedReplyId(preparedDraft)
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
            if (!editor && !node.isVisibleToUser && expectedReplyId != null) {
                // Only a complete message for THIS nonce. An offscreen node
                // is not trusted to contain ordinary chat history.
                ChatReplySemanticPolicy.completedNodeText(
                    node.text?.toString(), expectedReplyId
                )?.let { conversation.add(it) }
                ChatReplySemanticPolicy.completedNodeText(
                    node.contentDescription?.toString(), expectedReplyId
                )?.let { value ->
                    if (value != node.text?.toString()) conversation.add(value)
                }
            }
            // A long conversation can expose more than 1,500 nodes.
            // Exhaust the visible tree within a bounded budget rather than
            // invalidating the composer simply because history is lengthy.
            if (depth >= 64 && node.childCount > 0) { truncated = true; return }
            for (i in 0 until node.childCount) {
                if (nodes.size >= 4000) { truncated = true; break }
                node.getChild(i)?.let { walk(it, index, depth + 1, editor) }
            }
        }
        walk(root, -1, 0)
        val editor = if (!truncated) ChatComposerPolicy.choose(
            composers, root.screenBox(), density,
            recoveringPreparedDraft = preparedDraft != null) else null
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

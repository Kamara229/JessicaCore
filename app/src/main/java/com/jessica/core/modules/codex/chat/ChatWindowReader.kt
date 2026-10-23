package com.jessica.core.modules.codex.chat

import android.accessibilityservice.AccessibilityService
import android.graphics.Rect
import android.view.accessibility.AccessibilityNodeInfo
import android.view.accessibility.AccessibilityWindowInfo

fun AccessibilityNodeInfo.screenBox(): UiBox {
    val rect = Rect()
    getBoundsInScreen(rect)
    return UiBox(rect.left, rect.top, rect.right, rect.bottom)
}

data class ChatWindow(val root: AccessibilityNodeInfo, val nodes: List<AccessibilityNodeInfo>,
                      val composer: AccessibilityNodeInfo?,
                      val send: AccessibilityNodeInfo?, val generating: Boolean, val content: String,
                      val editorCandidates: Int = 0, val sendCandidates: Int = 0,
                      val composerDiagnostic: String = "") {
    fun release() { nodes.asReversed().forEach { @Suppress("DEPRECATION") it.recycle() } }
}

/** Small window-selection facade; parsing belongs to ChatWindowInspector. */
object ChatWindowReader {
    fun read(service: AccessibilityService, preparedDraft: String? = null): ChatWindow? {
        val windows = service.windows
        val apps = windows.filter { it.type == AccessibilityWindowInfo.TYPE_APPLICATION }
        val roots = apps.mapNotNull { window -> window.root?.let { window to it } }
        val activeRoot = service.rootInActiveWindow
        var selected: AccessibilityNodeInfo? = null
        try {
            val chat = roots.filter { (_, root) ->
                root.packageName?.toString() == "com.openai.chatgpt" && root.isVisibleToUser
            }.maxWithOrNull(compareBy<Pair<AccessibilityWindowInfo, AccessibilityNodeInfo>> {
                it.first.isActive
            }.thenBy { it.first.isFocused })
            val activePackage = activeRoot?.packageName?.toString()
            val chatVisible = chat != null ||
                (activePackage == "com.openai.chatgpt" && activeRoot.isVisibleToUser)
            val otherApplicationActive = roots.any { (w,root) ->
                w.isActive && root.packageName?.toString() != "com.openai.chatgpt"
            }
            val systemShadeFocused = windows.any { w ->
                w.type == AccessibilityWindowInfo.TYPE_SYSTEM && w.isActive && w.isFocused &&
                    w.root?.let { root ->
                        try {
                            root.packageName?.toString() == "com.android.systemui" && root.isVisibleToUser
                        } finally { @Suppress("DEPRECATION") root.recycle() }
                    } == true
            }
            val inputMethodActive = windows.any { w ->
                w.type == AccessibilityWindowInfo.TYPE_INPUT_METHOD && w.isActive
            }
            if (!ChatForegroundPolicy.permits(
                activePackage, activeRoot?.isVisibleToUser == true, chatVisible,
                otherApplicationActive, systemShadeFocused, inputMethodActive
            )) return null
            // Foreground ChatGPT beats stale isActive flags from a notification.
            selected = if (activePackage == "com.openai.chatgpt" && activeRoot?.isVisibleToUser == true)
                activeRoot else chat?.second ?: return null
            return inspect(selected, service.resources.displayMetrics.density, preparedDraft)
        } finally {
            roots.forEach { (_,root) ->
                if (root !== selected) @Suppress("DEPRECATION") root.recycle()
            }
            if (activeRoot !== selected && roots.none { (_,root) -> root === activeRoot })
                @Suppress("DEPRECATION") activeRoot?.recycle()
        }
    }

    internal fun inspect(root: AccessibilityNodeInfo, density: Float,
                         preparedDraft: String? = null): ChatWindow =
        ChatWindowInspector.inspect(root, density, preparedDraft)
}

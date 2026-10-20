package com.jessica.core.ui.chat

/** Toolbar action is determined solely by the conversation being displayed.
 * Deleted conversations elsewhere in the search/trash must not change it.
 */
enum class ChatToolbarAction {
    DELETE, RESTORE, NONE
}

fun chatToolbarAction(
    activeThreadId: String?,
    previewDeletedThreadId: String?
): ChatToolbarAction = when {
    previewDeletedThreadId != null -> ChatToolbarAction.RESTORE
    activeThreadId != null -> ChatToolbarAction.DELETE
    else -> ChatToolbarAction.NONE
}

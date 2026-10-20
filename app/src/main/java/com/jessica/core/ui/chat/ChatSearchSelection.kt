package com.jessica.core.ui.chat

import com.jessica.core.modules.chat.ChatThread

/**
 * Opening a search result must never implicitly undo deletion.
 * A deleted row opens a read-only preview; restore is a separate action.
 */
enum class ChatSearchSelection { OPEN_ACTIVE, PREVIEW_DELETED }

fun chatSearchSelection(thread: ChatThread): ChatSearchSelection =
    if (thread.deletedAt != null) ChatSearchSelection.PREVIEW_DELETED
    else ChatSearchSelection.OPEN_ACTIVE

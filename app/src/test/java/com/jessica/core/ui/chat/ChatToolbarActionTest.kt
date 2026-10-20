package com.jessica.core.ui.chat

import org.junit.Assert.assertEquals
import org.junit.Test

class ChatToolbarActionTest {
    private val activeA = "11111111-1111-4111-8111-111111111111"
    private val deletedB = "22222222-2222-4222-8222-222222222222"

    @Test fun normalChatAlwaysHasTrashEvenIfAnotherChatWasDeleted() {
        // After B is soft-deleted, A becomes active; A must keep its trash.
        assertEquals(ChatToolbarAction.DELETE, chatToolbarAction(activeA, null))
    }

    @Test fun deletedPreviewShowsOnlyRestoreArrow() {
        // A remains the active stored chat while deleted B is only previewed.
        assertEquals(ChatToolbarAction.RESTORE, chatToolbarAction(activeA, deletedB))
    }

    @Test fun browsingAnotherChatClearsRestoreAction() {
        assertEquals(ChatToolbarAction.DELETE, chatToolbarAction(activeA, null))
    }

    @Test fun noChatSelectedHasNoDestructiveAction() {
        assertEquals(ChatToolbarAction.NONE, chatToolbarAction(null, null))
    }

    @Test fun deletedPreviewCanBeRestoredWithoutAnyActiveChat() {
        assertEquals(ChatToolbarAction.RESTORE, chatToolbarAction(null, deletedB))
    }
}

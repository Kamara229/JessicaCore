package com.jessica.core.ui.chat

import com.jessica.core.modules.chat.ChatThread
import org.junit.Assert.assertEquals
import org.junit.Test

class ChatSearchSelectionTest {
    private fun thread(deletedAt: Long?): ChatThread =
        ChatThread("11111111-1111-4111-8111-111111111111",
            "Тестовый чат", 100L, deletedAt=deletedAt)

    @Test fun clickingDeletedRowOnlyPreviewsIt() {
        assertEquals(ChatSearchSelection.PREVIEW_DELETED,
            chatSearchSelection(thread(deletedAt=200L)))
    }

    @Test fun clickingNormalRowOpensEditableChat() {
        assertEquals(ChatSearchSelection.OPEN_ACTIVE,
            chatSearchSelection(thread(deletedAt=null)))
    }
}

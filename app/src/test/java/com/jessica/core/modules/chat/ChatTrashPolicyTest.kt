package com.jessica.core.modules.chat

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatTrashPolicyTest {
    private val removed = 1_000_000L

    @Test fun deletedChatIsRecoverableBeforeDeadline() {
        assertTrue(ChatTrashPolicy.canRestore(removed, removed))
        assertTrue(ChatTrashPolicy.canRestore(removed, removed + ChatTrashPolicy.RETENTION_MS - 1))
        assertFalse(ChatTrashPolicy.isExpired(removed, removed + ChatTrashPolicy.RETENTION_MS - 1))
    }

    @Test fun deletionIsFinalExactlyAtSixtyMinutes() {
        val expiry = removed + 60 * 60 * 1000L
        assertEquals(expiry, ChatTrashPolicy.expiresAt(removed))
        assertFalse(ChatTrashPolicy.canRestore(removed, expiry))
        assertTrue(ChatTrashPolicy.isExpired(removed, expiry))
    }

    @Test fun activeChatsHaveNoExpiry() {
        assertFalse(ChatTrashPolicy.canRestore(null))
        assertFalse(ChatTrashPolicy.isExpired(null))
        assertEquals(0L, ChatTrashPolicy.minutesLeft(null))
    }

    @Test fun countdownRoundsUpCorrectly() {
        assertEquals(60L, ChatTrashPolicy.minutesLeft(removed, removed))
        assertEquals(1L, ChatTrashPolicy.minutesLeft(removed, removed + ChatTrashPolicy.RETENTION_MS - 1))
        assertEquals(0L, ChatTrashPolicy.minutesLeft(removed, removed + ChatTrashPolicy.RETENTION_MS))
    }

    @Test fun translatingPreviousReplyIncludesMoreThanEightHundredCharacters() {
        val original = "Jessica Core: " + "Detailed explanation. ".repeat(65)
        val window = ChatContextWindow.selectForTask(
            listOf(ChatMessage(text=original,isUser=false)),
            "переведи свой ответ на японский язык"
        )
        assertEquals(1, window.size)
        assertEquals(original, window.single().text)
    }
    @Test fun ordinaryRequestKeepsShortConversationHistory() {
        val original = "a".repeat(2500)
        val window = ChatContextWindow.selectForTask(
            listOf(ChatMessage(text=original,isUser=false)),
            "Расскажи о своем проекте"
        )
        assertEquals(800,window.single().text.length)
    }
}

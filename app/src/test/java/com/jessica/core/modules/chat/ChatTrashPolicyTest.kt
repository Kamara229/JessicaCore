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
        assertEquals(original.trim(), window.single().text)
    }
    @Test fun ordinaryRequestKeepsShortConversationHistory() {
        val original = "a".repeat(2500)
        val window = ChatContextWindow.selectForTask(
            listOf(ChatMessage(text=original,isUser=false)),
            "Расскажи о своем проекте"
        )
        assertEquals(800,window.single().text.length)
    }

    @Test fun referentialCodeFollowupKeepsMoreHistoryThanPlainRequest() {
        val code = "class Example {" + "override fun run() = 1; ".repeat(65) + "}"
        val history = listOf(ChatMessage(text=code,isUser=false))
        val normal = ChatContextWindow.selectForTask(history,"Расскажи о компиляции")
        val followup = ChatContextWindow.selectForTask(history,"Исправь предыдущий код")
        assertEquals(800,normal.single().text.length)
        assertTrue(followup.single().text.length > 800)
        assertTrue(followup.single().text.startsWith("class Example"))
    }
    @Test fun followupContextUsesOnlyRecentMessagesInSameChat() {
        val messages = (1..30).map { ChatMessage(text="Message " + it,isUser=false) }
        val picked = ChatContextWindow.selectForTask(messages,"добавь еще один пример")
        assertTrue(picked.size <= 10)
        assertEquals("Message 30",picked.last().text)
        assertFalse(picked.any { it.text == "Message 1" })
    }
}

package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.*
import org.junit.Assert.*
import org.junit.Test

class ChatAutomationTest {
    private val id = "12345678-abcd-4abc-8abc-123456789012"
    @Test fun selectedWorkAlwaysBlocksEvenWhenChatLabelAlsoAppears() {
        assertEquals(OrdinaryChatMode.WORK, ChatUiPolicy.mode(listOf(ChatControl("Чат", true, true), ChatControl("Работа", true, true))))
        assertEquals(OrdinaryChatMode.UNKNOWN, ChatUiPolicy.mode(listOf(ChatControl("Чат", false, false))))
        assertEquals(OrdinaryChatMode.CHAT, ChatUiPolicy.mode(listOf(ChatControl("Чат", true, true))))
        assertFalse(ChatUiPolicy.canSend("com.openai.chatgpt", OrdinaryChatMode.UNKNOWN, false, "", "prompt", false))
        assertFalse(ChatUiPolicy.canSend("com.other.app", OrdinaryChatMode.CHAT, false, "", "prompt", false))
        assertFalse(ChatUiPolicy.canSend("com.openai.chatgpt", OrdinaryChatMode.CHAT, true, "", "prompt", false))
        assertFalse(ChatUiPolicy.canSend("com.openai.chatgpt", OrdinaryChatMode.CHAT, false, "private draft", "prompt", false))
        assertTrue(ChatUiPolicy.canSend("com.openai.chatgpt", OrdinaryChatMode.CHAT, false, "prompt", "prompt", false))
    }
    @Test fun aPromptOrOldResponseCannotBeAcceptedAsCurrentReply() {
        assertNull(ChatUiPolicy.reply("Первая строка: JESSICA_REPLY_$id\n{\"status\":\"change\"}\nПоследняя строка: JESSICA_END_$id", id))
        assertNull(ChatUiPolicy.reply("JESSICA_REPLY_$id\n{}", id))
        assertEquals("{}", ChatUiPolicy.reply("JESSICA_REPLY_$id\n```json\n{}\n```\nJESSICA_END_$id", id))
        assertNull(ChatUiPolicy.reply("JESSICA_REPLY_other\n{}\nJESSICA_END_other", id))
    }
    @Test fun ambiguousSendIsNeverRepeatedAndPartialStreamingCannotFinishTheTurn() {
        val state = ChatRunState(0)
        assertTrue(state.beginSend(1000))
        assertFalse(state.beginSend(2000))
        assertNull(state.acceptReply("{}", true, 9000))
        assertNull(state.acceptReply("{}", false, 10000))
        assertNull(state.acceptReply("{x}", false, 15000))
        assertNull(state.acceptReply("{x}", false, 20000))
        assertEquals("{x}", state.acceptReply("{x}", false, 21000))
        assertTrue(state.beginSend(22000))
        state.stop()
        assertFalse(state.beginSend(23000))
        assertNull(state.acceptReply("{}", false, 24000))
    }
    @Test fun sessionTimeoutAndMessageBudgetAreHardLimits() {
        val state = ChatRunState(0, maxTurns = 1)
        assertTrue(state.beginSend(1))
        state.acceptReply("{}", false, 10000)
        assertEquals("{}", state.acceptReply("{}", false, 17000))
        assertFalse(state.beginSend(20000))
        assertTrue(state.expired(3600000))
    }
}

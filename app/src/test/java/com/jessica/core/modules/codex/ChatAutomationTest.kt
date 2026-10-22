package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.*
import org.junit.Assert.*
import org.junit.Test

class ChatAutomationTest {
    private val id = "12345678-abcd-4abc-8abc-123456789012"
    @Test fun onlyExplicitSelectionAuthorizesSendingRegardlessOfChatMode() {
        assertFalse(ChatUiPolicy.canSend("com.openai.chatgpt", false, false, "", "prompt", false))
        assertFalse(ChatUiPolicy.canSend("com.other.app", true, false, "", "prompt", false))
        assertFalse(ChatUiPolicy.canSend("com.openai.chatgpt", true, true, "", "prompt", false))
        assertFalse(ChatUiPolicy.canSend("com.openai.chatgpt", true, false, "private draft", "prompt", false))
        assertFalse(ChatUiPolicy.canSend("com.openai.chatgpt", true, false, "prompt", "prompt", true))
        assertTrue(ChatUiPolicy.canSend("com.openai.chatgpt", true, false, "prompt", "prompt", false))
        assertTrue(ChatUiPolicy.canSend("com.openai.chatgpt", true, false, "", "prompt", false))
    }
    @Test fun visibleCurrentRequestOrReplyPreventsReinsertionAfterManualSend() {
        val prompt = ChatConnectivityProbe.prompt(id)
        assertTrue(ChatUiPolicy.alreadyPosted("Работа\n" + prompt.replace("\n", "  "), prompt, id))
        val reply = "JESSICA_REPLY_$id\n{\"nonce\":\"$id\",\"ok\":true}\nJESSICA_END_$id"
        assertTrue(ChatUiPolicy.alreadyPosted(reply, prompt, id))
        val other = "23456789-abcd-4abc-8abc-123456789012"
        assertFalse(ChatUiPolicy.alreadyPosted(reply.replace(id, other), prompt, id))
        assertFalse(ChatUiPolicy.alreadyPosted("Работа\nпредыдущий разговор", prompt, id))
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
    @Test fun repairCannotConsumeThePreviousAnswerStillVisibleInTheChat() {
        val next = "23456789-abcd-4abc-8abc-123456789012"
        val old = "JESSICA_REPLY_$id\n{\"nonce\":\"$id\",\"status\":\"bad\"}\nJESSICA_END_$id"
        val prompt = ChatUiPolicy.repairPrompt(id, next, "invalid-json")
        assertTrue(prompt.contains("nonce $id"))
        assertNull(ChatUiPolicy.reply(old + "\n" + prompt, next))
        val repaired = """{"nonce":"$id","status":"no_change","files":[]}"""
        assertEquals(repaired, ChatUiPolicy.reply(old + "\n" + prompt +
            "\nJESSICA_REPLY_$next\n$repaired\nJESSICA_END_$next", next))
    }
    @Test fun noKeyConnectivityProbeMustMatchThisExactSession() {
        val prompt = ChatConnectivityProbe.prompt(id)
        assertTrue(prompt.contains("JESSICA_REPLY_$id"))
        assertTrue(prompt.contains("JESSICA_END_$id"))
        // The request itself must never contain standalone delimiters. Only the
        // actual assistant reply may be interpreted as successful.
        assertNull(ChatUiPolicy.reply(prompt,id))
        assertTrue(ChatConnectivityProbe.accepts("""{"nonce":"$id","ok":true}""",id))
        assertFalse(ChatConnectivityProbe.accepts("""{"nonce":"different","ok":true}""",id))
        assertFalse(ChatConnectivityProbe.accepts("""{"nonce":"$id","ok":false}""",id))
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

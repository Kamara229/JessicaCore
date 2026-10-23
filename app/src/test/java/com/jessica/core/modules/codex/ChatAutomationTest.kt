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
        assertNull(state.acceptReply("{x}", false, 16500))
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
    @Test fun exactCompletedReplyIsAcceptedPromptlyAndAllowsTheNextSend() {
        val state = ChatRunState(0L, maxTurns = 4)
        assertTrue(state.beginSend(1000L))
        assertNull(state.acceptReply(null, true, 1900L))
        assertNull(state.acceptReply("завершённый ответ", false, 4000L))
        assertNull(state.acceptReply("завершённый ответ", false, 6200L))
        assertEquals("завершённый ответ",
            state.acceptReply("завершённый ответ", false, 6600L))
        assertFalse(state.waiting)
        assertTrue("Next turn must be eligible without a manual restart",state.beginSend(7000L))
    }

    @Test fun chatgptLengthLimitStopsSendingRatherThanTreatingBannerAsAComposer() {
        val russian = "Вы достигли максимальной длины этого обсуждения, но можете продолжить обсуждение, начав новый чат."
        val english = "You've reached the maximum length for this conversation, but you can continue by starting a new chat."
        assertTrue(ChatUiPolicy.conversationAtLimit(russian))
        assertTrue(ChatUiPolicy.conversationAtLimit(english))
        assertFalse(ChatUiPolicy.conversationAtLimit("Какова максимальная длина обсуждения в ChatGPT?"))
        assertFalse(ChatUiPolicy.conversationAtLimit("Это обычное сообщение в выбранном чате."))
    }

    @Test fun longWaitRetainsSessionWithoutRetryingUncertainSend() {
        val state = ChatRunState(1000, maxTurns = 6, durationMs = 8 * 60 * 60 * 1000L)
        assertTrue(state.beginSend(2000))
        assertTrue(state.replyOverdue(120_000))
        assertFalse("Slow reply must not stop CodeX",state.expired(25 * 60 * 1000L))
        assertFalse("No blind resend",state.beginSend(120_001))
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

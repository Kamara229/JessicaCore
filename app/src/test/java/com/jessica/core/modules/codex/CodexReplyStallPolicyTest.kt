package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.CodexReplyStallPolicy
import com.jessica.core.modules.codex.chat.ChatRunState
import com.jessica.core.modules.codex.relay.ChatReplyCollector
import org.junit.Assert.*
import org.junit.Test

class CodexReplyStallPolicyTest {
    private val id = "ad205a6b-dba2-4343-a421-cf56e23000dc"
    private fun outcome(
        elapsed: Long,
        waiting: Boolean = true,
        visible: Boolean = true,
        connected: Boolean = true,
        generating: Boolean = false,
        started: Boolean = false,
        complete: Boolean = false
    ) = CodexReplyStallPolicy.classify(
        waiting, visible, connected, generating, started, complete, elapsed
    )

    @Test fun anUnmarkedChatGptAnswerCannotSilentlyWaitFor24Hours() {
        // Regression: the observed ChatGPT conversation returned a normal
        // multi-paragraph response instead of the unique reply delimiters.
        val collector = ChatReplyCollector()
        val plainAnswer = "Продолжаю исследование Calibri 2.0. " +
            "Результат обучения Core и Green Tunnel сохранён."
        assertNull(collector.observe(plainAnswer, id))
        assertFalse(collector.hasStarted(id))
        val state = ChatRunState(0L, maxTurns = 600)
        assertTrue(state.beginSend(1000L))
        assertNull(state.acceptReply(collector.observe(plainAnswer, id), false, 121_000L))
        assertTrue(state.waiting)
        assertEquals(CodexReplyStallPolicy.Failure.NONE,
            outcome(CodexReplyStallPolicy.NO_MARKERS_MS - 1))
        assertEquals(CodexReplyStallPolicy.Failure.REPLY_WITHOUT_VERIFIABLE_MARKERS,
            outcome(CodexReplyStallPolicy.NO_MARKERS_MS))
        assertTrue(CodexReplyStallPolicy.message(
            CodexReplyStallPolicy.Failure.REPLY_WITHOUT_VERIFIABLE_MARKERS
        ).contains("JESSICA_REPLY"))
        assertEquals(1, state.sends)
        assertFalse("No blind duplicate send", state.beginSend(122_000L))
    }

    @Test fun validExactMarkersAreStillTheOnlyAcceptedRelayCompletion() {
        val collector = ChatReplyCollector()
        val start = "JESSICA_REPLY_$id\nТолько исследование GitHub."
        assertNull(collector.observe(start, id))
        assertTrue(collector.hasStarted(id))
        assertEquals(CodexReplyStallPolicy.Failure.NONE,
            outcome(160_000L, started = true))
        val complete = "$start\nJESSICA_END_$id"
        assertEquals("Только исследование GitHub.", collector.observe(complete, id))
        assertEquals(CodexReplyStallPolicy.Failure.NONE,
            outcome(800_000L, started = true, complete = true))
    }

    @Test fun incompleteVisibleAnswerFailsSafelyAfterBoundedTime() {
        assertEquals(CodexReplyStallPolicy.Failure.PARTIAL_MARKED_REPLY,
            outcome(180_000L, started = true))
        assertEquals(CodexReplyStallPolicy.Failure.NONE,
            outcome(179_999L, started = true))
    }

    @Test fun slowStreamingAndOfflineOrHiddenChatsAreNotMisclassified() {
        assertEquals(CodexReplyStallPolicy.Failure.NONE,
            outcome(240_000L, generating = true))
        assertEquals(CodexReplyStallPolicy.Failure.GENERATION_OVERDUE,
            outcome(CodexReplyStallPolicy.GENERATING_MS, generating = true))
        assertEquals(CodexReplyStallPolicy.Failure.NONE,
            outcome(2_000_000L, visible = false))
        assertEquals(CodexReplyStallPolicy.Failure.NONE,
            outcome(2_000_000L, connected = false))
        assertEquals(CodexReplyStallPolicy.Failure.NONE,
            outcome(2_000_000L, waiting = false))
        assertEquals(CodexReplyStallPolicy.Failure.NONE,
            outcome(2_000_000L, complete = true))
    }

    @Test fun quotedMarkersInsideTheOwnersInstructionsNeverCountAsReply() {
        val collector = ChatReplyCollector()
        val owner = "Начни отдельной строкой JESSICA_REPLY_$id. " +
            "Последняя строка JESSICA_END_$id."
        assertNull(collector.observe(owner, id))
        assertFalse(collector.hasStarted(id))
    }
}

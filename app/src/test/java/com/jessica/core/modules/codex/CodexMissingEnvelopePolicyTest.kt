package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.CodexMissingEnvelopePolicy
import com.jessica.core.modules.codex.chat.CodexMissingEnvelopePolicy.Decision
import com.jessica.core.modules.codex.chat.ChatRunState
import org.junit.Assert.*
import org.junit.Test

/** Regression: a real Render relay streamed the only answer and then waited
 * for JESSICA_REPLY / JESSICA_END markers for hours. No testing via Jessica.
 * This policy never enables a resend, parses a free-form answer, or runs a test.
 */
class CodexMissingEnvelopePolicyTest {
    private val id = 10_000L

    @Test fun ordinaryUnmarkedAnswerStopsWithClearStatusInsteadOfWaitingAllDay() {
        val state = ChatRunState(0L, maxTurns = 600, durationMs = 86_400_000L)
        val guard = CodexMissingEnvelopePolicy()
        assertTrue(state.beginSend(id))
        assertEquals(Decision.WAIT, guard.observe(11_000L, state.sendAt,
            generating = true, startedMarker = false, completedEnvelope = false))
        assertEquals(Decision.WAIT, guard.observe(12_000L, state.sendAt,
            generating = false, startedMarker = false, completedEnvelope = false))
        assertEquals(Decision.WAIT, guard.observe(100_000L, state.sendAt,
            generating = false, startedMarker = false, completedEnvelope = false))
        assertEquals(Decision.STOP_UNMARKED, guard.observe(131_000L, state.sendAt,
            generating = false, startedMarker = false, completedEnvelope = false))
        assertEquals(1, state.sends)
        assertTrue(state.waiting)  // Policy authorizes no state transition/send.
        assertFalse(state.beginSend(150_000L))
    }

    @Test fun genuineCompletedEnvelopeNeverTriggersFailureAndCanAdvance() {
        val guard = CodexMissingEnvelopePolicy()
        val state = ChatRunState(0L, maxTurns = 10)
        assertTrue(state.beginSend(id))
        guard.observe(20_000L, state.sendAt,
            generating = true, startedMarker = false, completedEnvelope = false)
        assertEquals(Decision.WAIT, guard.observe(160_000L, state.sendAt,
            generating = false, startedMarker = true, completedEnvelope = true))
        assertNull(state.acceptReply("Подтверждённый ответ", false, 161_000L))
        assertEquals("Подтверждённый ответ",
            state.acceptReply("Подтверждённый ответ", false, 164_000L))
        assertTrue(state.beginSend(165_000L))
        assertEquals(Decision.WAIT, guard.observe(400_000L, state.sendAt,
            generating = false, startedMarker = false, completedEnvelope = false))
    }

    @Test fun noObservedGenerationNeverGuessesThatChatGPTAnswered() {
        val guard = CodexMissingEnvelopePolicy()
        assertEquals(Decision.WAIT, guard.observe(10_000_000L, id,
            generating = false, startedMarker = false, completedEnvelope = false))
        assertEquals(Decision.WAIT, guard.observe(11_000_000L, id,
            generating = false, startedMarker = true, completedEnvelope = false))
        assertEquals(Decision.WAIT, guard.observe(12_000_000L, id,
            generating = false, startedMarker = false, completedEnvelope = false))
    }

    @Test fun partialAnswerStillMissingEndStopsSafelyWithoutReplayingThePartial() {
        val guard = CodexMissingEnvelopePolicy()
        assertEquals(Decision.WAIT, guard.observe(11_000L, id,
            generating = true, startedMarker = true, completedEnvelope = false))
        assertEquals(Decision.WAIT, guard.observe(12_000L, id,
            generating = false, startedMarker = true, completedEnvelope = false))
        assertEquals(Decision.STOP_PARTIAL, guard.observe(200_000L, id,
            generating = false, startedMarker = true, completedEnvelope = false))
    }

    @Test fun continuedGenerationCannotBeMisreadAsCompletedTimeout() {
        val guard = CodexMissingEnvelopePolicy()
        guard.observe(11_000L, id,
            generating = true, startedMarker = false, completedEnvelope = false)
        assertEquals(Decision.WAIT, guard.observe(300_000L, id,
            generating = true, startedMarker = false, completedEnvelope = false))
        assertEquals(Decision.WAIT, guard.observe(301_000L, id,
            generating = false, startedMarker = false, completedEnvelope = false))
        assertEquals(Decision.WAIT, guard.observe(330_000L, id,
            generating = false, startedMarker = false, completedEnvelope = false))
        assertEquals(Decision.STOP_UNMARKED, guard.observe(350_000L, id,
            generating = false, startedMarker = false, completedEnvelope = false))
    }

    @Test fun resettingSessionClearsEvidenceFromFormerTurn() {
        val guard = CodexMissingEnvelopePolicy()
        guard.observe(11_000L, id, true, false, false)
        guard.observe(12_000L, id, false, false, false)
        guard.reset()
        assertEquals(Decision.WAIT, guard.observe(500_000L, id, false, false, false))
        // A different original sending event also clears old observations.
        guard.observe(510_000L, id, true, false, false)
        assertEquals(Decision.WAIT,
            guard.observe(600_000L, id + 1000L, false, false, false))
    }
}

package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.ChatRunState
import com.jessica.core.modules.codex.chat.CodexSessionLimits
import org.junit.Assert.*
import org.junit.Test

class CodexExtendedSessionTest {
    @Test fun anExplicitlyStartedSessionCanContinueBeyondSixtyTurnsWithoutDuplicatingSends() {
        val state = ChatRunState(
            startedAt = 0L,
            maxTurns = CodexSessionLimits.MAX_MESSAGES,
            durationMs = CodexSessionLimits.MAX_DURATION_MS
        )
        assertEquals(600, state.maxTurns)
        assertEquals(24L * 60L * 60L * 1000L, state.durationMs)
        for (turn in 1..600) {
            val now = turn * 10_000L
            assertTrue("Turn $turn must be eligible", state.beginSend(now))
            assertFalse("Never duplicate an uncertain send", state.beginSend(now + 1))
            assertNull(state.acceptReply("Ответ $turn", generating = false, now = now + 2_000L))
            assertEquals("Ответ $turn", state.acceptReply("Ответ $turn", generating = false, now = now + 6_000L))
        }
        assertEquals(600, state.sends)
        assertFalse("User must explicitly start a new bounded session", state.beginSend(6_100_000L))
    }

    @Test fun twentyFourHourGuardAndOwnerStopRemainHardLimits() {
        val active = ChatRunState(0L, CodexSessionLimits.MAX_MESSAGES, CodexSessionLimits.MAX_DURATION_MS)
        assertFalse(active.expired(CodexSessionLimits.MAX_DURATION_MS - 1))
        assertTrue(active.expired(CodexSessionLimits.MAX_DURATION_MS))
        assertFalse(active.beginSend(CodexSessionLimits.MAX_DURATION_MS))
        val stopped = ChatRunState(0L, CodexSessionLimits.MAX_MESSAGES, CodexSessionLimits.MAX_DURATION_MS)
        assertTrue(stopped.beginSend(1000L))
        stopped.stop()
        assertFalse(stopped.beginSend(2000L))
        assertNull(stopped.acceptReply("Поздний ответ", false, 8000L))
    }
}

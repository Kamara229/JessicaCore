package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.CodexInterpretationFailurePolicy
import org.junit.Assert.*
import org.junit.Test

class CodexInterpretationFailurePolicyTest {
    @Test fun quotaExhaustionReportsImmediatelyWithoutExecutingTests() {
        val quota = IllegalStateException("HTTP 429: rate limit")
        assertTrue(CodexInterpretationFailurePolicy.handoffNow(quota, 1))
        val report = CodexInterpretationFailurePolicy.report(quota)
        assertTrue(report.contains("НЕ смогла проанализировать"))
        assertTrue(report.contains("НЕ выполнялся"))
        assertTrue(report.contains("backend, Render и Supabase"))
        assertFalse(report.contains("успешно выполнен"))
    }

    @Test fun transientFailureGetsOnlyOneInterpretationRetry() {
        val failure = IllegalStateException("HTTP 503")
        assertFalse(CodexInterpretationFailurePolicy.handoffNow(failure, 1))
        assertTrue(CodexInterpretationFailurePolicy.handoffNow(failure, 2))
        assertTrue(CodexInterpretationFailurePolicy.handoffNow(failure, 10))
        val report = CodexInterpretationFailurePolicy.report(failure)
        assertTrue(report.contains("ошибка или недоступность"))
        assertTrue(report.contains("без явного запроса"))
    }
}

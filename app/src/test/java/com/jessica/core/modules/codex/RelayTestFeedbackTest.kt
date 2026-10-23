package com.jessica.core.modules.codex

import com.jessica.core.modules.AIResult
import com.jessica.core.modules.codex.relay.RelayTestFeedback
import org.junit.Assert.*
import org.junit.Test

class RelayTestFeedbackTest {
    private val task = "Сколько дней прошло с 9 октября по 23 октября? Назови число и кратко объясни."
    @Test fun missingExplanationIsExplicitlyReportedInsteadOfPassed() {
        val feedback=RelayTestFeedback.note(task,AIResult(true,
            "От 2026-10-09 до 2026-10-23 прошло 14 полных дней."))
        assertTrue(feedback.contains("объяснение"))
    }
    @Test fun explicitCalendarExplanationIsNotFlaggedAsMissing() {
        val feedback=RelayTestFeedback.note(task,AIResult(true,
            "14 дней. Начальная дата не учитывается, конечная включена."))
        assertFalse(feedback.contains("не обнаружено"))
    }
    @Test fun backendSuccessIsNotMistakenForTestQuality() {
        val feedback=RelayTestFeedback.note(task,AIResult(false,
            "Не удалось получить ответ сервера"))
        assertTrue(feedback.contains("не подтверждён"))
    }
}

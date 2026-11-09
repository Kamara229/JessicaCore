package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.relay.CodexPassiveLocalHandoff
import org.junit.Assert.*
import org.junit.Test

class CodexPassiveLocalHandoffTest {
    private val payload = "CALIBRI S101: CPU-проверка 456 TRAIN продолжается. " +
        "Следующий шаг — проверить точное совпадение EOS и скорость. LIVE не менять."
    private val uuid = "a296daee-1fc6-4d31-abd4-a4fd0f59e12a"

    @Test fun passiveProgressWorksWithoutNetworkOrGroq() {
        val result = CodexPassiveLocalHandoff.response(payload)
        assertNotNull(result)
        assertFalse(result!!.endSession)
        assertTrue(result.text.contains("CALIBRI S101"))
        assertTrue(result.text.contains("Тест Jessica не запускала"))
    }

    @Test fun shortSStageFromRealAssistantDoesNotCallGroqOrRender() {
        // Previously the prefix CALIBRI was mandatory on Android despite
        // the existing backend accepting /^S101:/ for safe no-test progress.
        val short = "S101: 353/456, 0 расхождений, CPU 1,392× S82, 0,69957 bpp. " +
            "Следующий шаг — проверить KV-cache и EOS. LIVE не менять."
        val direct = CodexPassiveLocalHandoff.response(short)
        assertNotNull(direct)
        assertFalse(direct!!.endSession)
        assertTrue(direct.text.contains("CALIBRI S101"))
        assertTrue(direct.text.contains("Тест Jessica не запускала"))
        val wrapped = "JESSICA_REPLY_$uuid\\n\\n$short\\n\\nJESSICA_END_$uuid"
        assertNotNull(CodexPassiveLocalHandoff.response(wrapped))
        assertNull(CodexPassiveLocalHandoff.response(
            wrapped.replace("JESSICA_END_$uuid",
                "JESSICA_END_00000000-0000-0000-0000-000000000000")))
    }

    @Test fun fullMatchedEnvelopeIsSafeButIncompleteEnvelopeRejected() {
        val wrapped = "JESSICA_REPLY_$uuid\n$payload\nJESSICA_END_$uuid"
        assertNotNull(CodexPassiveLocalHandoff.response(wrapped))
        assertNull(CodexPassiveLocalHandoff.response(wrapped.dropLast(20)))
        assertNull(CodexPassiveLocalHandoff.response(wrapped.replace("JESSICA_END_$uuid", "JESSICA_END_00000000-0000-0000-0000-000000000000")))
    }

    @Test fun explicitTasksOrChangesMustUseRealInterpreter() {
        val bad = listOf(
            payload + " Запусти тест в Jessica.",
            payload + " Выполни тест.",
            payload + " Создай новый тестовый чат.",
            payload + " Заверши сессию Кодекс.",
            payload + " Секрет и API-ключ.",
            payload + " Установи APK.",
            payload + " https://example.org",
            payload + " `source code`",
            "CALIBRI S101: Проведи тест. LIVE не менять.",
            "CALIBRI S101: Следующий шаг — проверить CPU.",
            "Привет! LIVE не менять.",
            "Кодекс Jessica: сервис интерпретации недоступен. Следующий шаг — проверить GitHub. LIVE не менять."
        )
        bad.forEach { assertNull("Unsafe progress must go through interpreter: $it",
            CodexPassiveLocalHandoff.response(it)) }
    }
}

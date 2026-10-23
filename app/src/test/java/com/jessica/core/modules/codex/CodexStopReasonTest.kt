package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.CodexStopReason
import org.junit.Assert.assertEquals
import org.junit.Test

class CodexStopReasonTest {
    @Test fun reasonsContainOnlySafeCategories() {
        assertEquals("owner_stop",CodexStopReason.classify("Кодекс остановлен пользователем"))
        assertEquals("plan_finished",CodexStopReason.classify("Переписка завершена"))
        assertEquals("send_guard",CodexStopReason.classify("Нажатие не подтверждено"))
        assertEquals("unhandled_error",CodexStopReason.classify("Кодекс приостановлен: HTTP 502"))
        assertEquals("service_disconnected",CodexStopReason.classify("Android отключил службу"))
    }
}

package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.CodexReplyProgress
import com.jessica.core.modules.codex.chat.CodexReplyProgressPolicy
import com.jessica.core.modules.codex.relay.ChatReplyCollector
import org.junit.Assert.*
import org.junit.Test

class CodexReplyProgressTest {
    private val id = "12345678-abcd-4abc-8abc-123456789012"
    @Test fun detectsReplyStartAndDisplaysStatusBeforeFinishing() {
        val c=ChatReplyCollector()
        assertFalse(c.hasStarted(id))
        c.observe("user: Начни ответ отдельной строкой JESSICA_REPLY_$id. Конец.",id)
        assertFalse("Quoted marker instruction is not a reply",c.hasStarted(id))
        c.observe("JESSICA_REPLY_$id\nПроверяю результат",id)
        assertTrue("Real opening delimiter must be recognized",c.hasStarted(id))
        assertEquals(CodexReplyProgress.PARTIAL,
            CodexReplyProgressPolicy.classify(true,false,false,false))
        c.observe("JESSICA_REPLY_$id\nПроверяю результат\nJESSICA_END_$id",id)
        assertEquals(CodexReplyProgress.ANALYZING,
            CodexReplyProgressPolicy.classify(c.hasStarted(id),true,false,false))
        assertEquals(CodexReplyProgress.ACCEPTED,
            CodexReplyProgressPolicy.classify(true,true,false,true))
        assertTrue(CodexReplyProgress.ACCEPTED.status.contains("анализирую"))
    }

    @Test fun neverTreatsStreamingOrUnmarkedResponseAsCompleted() {
        assertEquals(CodexReplyProgress.WAITING,
            CodexReplyProgressPolicy.classify(false,false,false,false))
        assertEquals(CodexReplyProgress.STREAMING,
            CodexReplyProgressPolicy.classify(false,false,true,false))
        assertEquals(CodexReplyProgress.STREAMING,
            CodexReplyProgressPolicy.classify(true,true,true,false))
        assertEquals(CodexReplyProgress.PARTIAL,
            CodexReplyProgressPolicy.classify(true,false,false,false))
    }

    @Test fun collectorKeepsResponseStartWhileChatGptVirtualizesItAndResetsById() {
        val c=ChatReplyCollector()
        c.observe("JESSICA_REPLY_$id\nВижу ответ",id)
        assertTrue(c.hasStarted(id))
        assertTrue(c.hasStarted(id))
        val newId="23456789-abcd-4abc-8abc-123456789012"
        c.observe("Ожидаю",newId)
        assertFalse(c.hasStarted(id))
        assertFalse(c.hasStarted(newId))
    }
}

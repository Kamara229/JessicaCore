package com.jessica.core.modules.codex.chat

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.jessica.core.modules.AIResult
import com.jessica.core.modules.codex.relay.ChatReplyCollector
import com.jessica.core.modules.codex.relay.RelayConversation
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** Real chat storage + deterministic fake ChatGPT/AI. No real user chat sent. */
@RunWith(AndroidJUnit4::class)
class CodexCompleteLoopTest {
    private val firstId = "12345678-abcd-4abc-8abc-123456789012"
    private val secondId = "23456789-abcd-4abc-8abc-123456789012"

    @Test fun firstMessageAnswerTestReportAndSecondChatTurnCompleteInOrder() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        var tests = 0
        val relay = RelayConversation(context,
            interpret = { request ->
                val message = request.getString("message")
                if (message.contains("Сколько будет 2 + 2?")) {
                    JSONObject().put("action", "test").put("task", "Сколько будет 2 + 2?")
                } else {
                    assertTrue(request.getJSONArray("history").toString().contains("Фактический ответ Jessica"))
                    JSONObject().put("action", "reply").put("reply",
                        "Результат передан. Продолжай план backend без повторного теста.")
                }
            },
            executeTest = { task, previous ->
                tests++
                assertEquals("Сколько будет 2 + 2?", task)
                assertTrue(previous.isEmpty())
                AIResult(true, "Результат: 4")
            }
        )
        val collector = ChatReplyCollector()
        val state = ChatRunState(0L, maxTurns = 5)
        relay.recordOutgoing("Джессика готова к работе в чате.")

        assertTrue(state.beginSend(1000L))
        val firstBody = "Выполни через обычный чат Jessica: Сколько будет 2 + 2?"
        val firstScreen = "JESSICA_REPLY_$firstId\n$firstBody\nJESSICA_END_$firstId"
        assertNull(state.acceptReply(collector.observe(firstScreen, firstId), false, 10000L))
        val verifiedFirst = state.acceptReply(
            collector.observe("Редактирование поля ChatGPT", firstId), false, 17000L)
        assertEquals(firstBody, verifiedFirst)

        val testReport = relay.respond(verifiedFirst!!) { }
        assertFalse(testReport.endSession)
        assertEquals(1, tests)
        assertTrue(testReport.text.contains("Результат: 4"))
        assertTrue(testReport.text.contains("Точная задача:"))
        relay.recordOutgoing(testReport.text)

        assertTrue(state.beginSend(20000L))
        val secondBody = "Принял фактический результат. Продолжим улучшения backend."
        val secondScreen = "JESSICA_REPLY_$secondId\n$secondBody\nJESSICA_END_$secondId"
        assertNull(state.acceptReply(collector.observe(secondScreen, secondId), false, 29000L))
        val verifiedSecond = state.acceptReply(
            collector.observe("ChatGPT idle", secondId), false, 36000L)
        assertEquals(secondBody, verifiedSecond)
        val next = relay.respond(verifiedSecond!!) { }
        assertFalse(next.endSession)
        assertEquals(1, tests)
        assertTrue(next.text.contains("Продолжай план backend"))
        assertEquals(2, state.sends)
    }
}

package com.jessica.core.modules.codex.chat

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.jessica.core.modules.AIResult
import com.jessica.core.modules.chat.ChatThreadStore
import com.jessica.core.modules.codex.relay.RelayConversation
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class RelayConversationTest {
    @Test fun realTestIsStoredAndReturnedVerbatimThenContextContinues() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        var tests = 0
        val conversation = RelayConversation(context, interpret = { payload ->
            if (payload.getString("message").contains("Сколько")) JSONObject().put("action","test").put("task","Сколько будет 2 + 2?")
            else {
                assertTrue(payload.getJSONArray("history").toString().contains("Неподтверждённый результат"))
                JSONObject().put("action","reply").put("reply","Результат передан. Продолжим проверку по плану.")
            }
        }, executeTest = { task, history ->
            tests++
            assertEquals("Сколько будет 2 + 2?",task)
            assertTrue(history.isEmpty())
            AIResult(false,"Неподтверждённый результат: ошибка сервера.")
        })
        conversation.plan = "Проверить арифметику"
        conversation.recordOutgoing("Джессика готова к работе в чате.")
        val report = conversation.respond("Проверь в своём чате: Сколько будет 2 + 2?") {}
        assertTrue(report.text.contains("Неподтверждённый результат: ошибка сервера."))
        assertTrue(report.text.contains("Запрос обработан сервером: false"))
        assertFalse(report.text.contains("Тест пройден"))
        val store = ChatThreadStore.shared(context)
        assertTrue(store.list().any { t -> store.messages(t.id).any { !it.isUser && it.text == "Неподтверждённый результат: ошибка сервера." } })
        conversation.recordOutgoing(report.text)
        val reply = conversation.respond("Исправил ошибку backend. Продолжим план.") {}
        assertEquals(1,tests)
        assertFalse(reply.endSession)
    }
    @Test fun finishDoesNotExecuteTests() = runBlocking {
        val conversation = RelayConversation(InstrumentationRegistry.getInstrumentation().targetContext,
            interpret = { JSONObject().put("action","finish").put("reply","План завершён, заканчиваю сессию.") },
            executeTest = { _,_ -> error("Unexpected test execution") })
        assertTrue(conversation.respond("План выполнен, на сегодня всё.") {}.endSession)
    }
    @Test fun explicitFollowUpKeepsItsOwnTestHistory() = runBlocking {
        var calls = 0
        val conversation = RelayConversation(InstrumentationRegistry.getInstrumentation().targetContext,
            interpret = { payload -> JSONObject().put("action","test").put("task",payload.getString("message"))
                .put("continueThread", calls > 0) },
            executeTest = { task, history ->
                calls++
                if (calls == 1) assertTrue(history.isEmpty())
                else {
                    assertEquals(2, history.size)
                    assertEquals("Запомни число 42", history[0].text)
                    assertEquals("Запомнила 42", history[1].text)
                    assertEquals("Какое число я назвал?",task)
                }
                AIResult(true, if (calls == 1) "Запомнила 42" else "42")
            })
        val first = conversation.respond("Запомни число 42") {}
        conversation.recordOutgoing(first.text)
        val second = conversation.respond("Какое число я назвал?") {}
        assertEquals(2,calls)
        assertTrue(second.text.contains("\n42\n"))
    }
}

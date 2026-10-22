package com.jessica.core.modules.codex.relay

import android.content.Context
import com.jessica.core.modules.JessicaAIEngine
import com.jessica.core.modules.AIResult
import com.jessica.core.modules.chat.ChatMessage
import com.jessica.core.modules.chat.ChatThreadStore
import com.jessica.core.modules.codex.CodexClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.time.Instant

data class RelayReply(val text: String, val endSession: Boolean = false)

/** Tests use the same JessicaAIEngine and local chat storage as regular Jessica chat. */
class RelayConversation(
    context: Context,
    private val interpret: suspend (JSONObject) -> JSONObject = { CodexClient().interpret(it) },
    private val executeTest: suspend (String, List<ChatMessage>) -> AIResult = { task, history -> JessicaAIEngine().solve(task, history) }
) {
    private val store = ChatThreadStore.shared(context)
    private val history = mutableListOf<ChatMessage>()
    private var journal: String? = null
    private var testThread: String? = null
    var plan: String = ""

    private suspend fun log(text: String, fromChatGPT: Boolean) = withContext(Dispatchers.IO) {
        val id = journal ?: store.create().id.also {
            journal = it
            store.applyAutomaticTitle(it, "Кодекс — переписка с ChatGPT")
        }
        store.append(id, ChatMessage(text, isUser = fromChatGPT))
        history.add(ChatMessage(text, isUser = fromChatGPT))
    }

    suspend fun recordOutgoing(text: String) { log(text, false) }
    suspend fun recordIncoming(text: String) { log(text, true) }

    suspend fun respond(message: String, notice: (String) -> Unit): RelayReply {
        val recent = JSONArray()
        history.takeLast(16).forEach { recent.put(JSONObject().put("role", if (it.isUser) "user" else "assistant")
            .put("content", it.text.takeLast(6000))) }
        recordIncoming(message)
        notice("Читаю ответ ChatGPT с учётом плана и результатов предыдущих тестов…")
        val decision = interpret(JSONObject().put("message", message).put("plan", plan.take(12000)).put("history", recent))
        return when (decision.getString("action")) {
            "test" -> {
                val task = decision.getString("task").trim()
                check(task.isNotBlank() && task.length <= 20000 && message.contains(task)) { "Формулировка теста не подтверждена сообщением ChatGPT" }
                notice("Выполняю тест в чате Джессики. Фактический ответ будет передан ChatGPT.")
                val id = withContext(Dispatchers.IO) {
                    if (decision.optBoolean("continueThread") && testThread != null) testThread!!
                    else store.create().id.also { testThread = it; store.applyAutomaticTitle(it, "Кодекс — тест ${Instant.now()}") }
                }
                val before = withContext(Dispatchers.IO) { store.messages(id) }
                val started = Instant.now().toString()
                withContext(Dispatchers.IO) { store.append(id, ChatMessage(task, true)) }
                // An error or no_verified_result is preserved verbatim. HTTP success is NOT a passing test.
                val result = executeTest(task, before)
                withContext(Dispatchers.IO) { store.append(id, ChatMessage(result.text, false)) }
                val report = "Попытка выполнения теста через мой обычный чат завершена.\nЧат Jessica: $id\nНачало: $started\nОкончание: ${Instant.now()}\n" +
                    "Запрос обработан сервером: ${result.success}. Это не оценка прохождения теста.\n" +
                    "Точная задача:\n$task\n\nФактический ответ Jessica, без пересказа:\n${result.text}\n\n" +
                    "Проверь этот результат по ожидаемому поведению и продолжай план."
                check(report.length <= 90000) { "Ответ слишком большой для одного сообщения. Полный результат сохранён в тестовом чате Jessica $id." }
                RelayReply(report)
            }
            "reply" -> RelayReply(decision.getString("reply"))
            "finish", "pause" -> RelayReply(decision.getString("reply"), endSession = true)
            else -> error("Неизвестное действие; переписка остановлена")
        }
    }
}

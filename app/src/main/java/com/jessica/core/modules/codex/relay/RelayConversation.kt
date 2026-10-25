package com.jessica.core.modules.codex.relay

import android.content.Context
import com.jessica.core.modules.JessicaAIEngine
import com.jessica.core.modules.AIResult
import com.jessica.core.modules.chat.ChatMessage
import com.jessica.core.modules.chat.ChatThreadStore
import com.jessica.core.modules.codex.CodexClient
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.time.Instant

data class RelayReply(val text: String, val endSession: Boolean = false)

/** Only interpretation has not produced any task or send-side effect. Safe to retry. */
class RelayInterpretationUnavailable(cause: Throwable): RuntimeException("Сервис анализа временно недоступен",cause)

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

    // Preserve the trailing-lambda API used by existing Android callers.
    suspend fun respond(message: String, notice: (String) -> Unit): RelayReply =
        respond(message, retryingInterpretation = false, notice = notice)

    suspend fun respond(message: String, retryingInterpretation: Boolean,
                        notice: (String) -> Unit): RelayReply {
        // Retry only the remote classification. An earlier failure already
        // stored this incoming answer; no duplicate journal/message writes.
        val context = if (retryingInterpretation &&
            history.lastOrNull()?.let { it.isUser && it.text == message } == true)
            history.dropLast(1) else history
        val recent = JSONArray()
        context.takeLast(16).forEach { recent.put(JSONObject().put("role", if (it.isUser) "user" else "assistant")
            .put("content", it.text.takeLast(6000))) }
        if (!retryingInterpretation) recordIncoming(message)
        notice("Вижу ответ ChatGPT и анализирую. Учитываю план и предыдущие результаты…")
        val decision = try {
            interpret(JSONObject().put("message", message).put("plan", plan.take(12000))
                .put("history", recent))
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            // A test has NOT been executed yet, so interpretation alone is
            // safe to reattempt without duplicating chat or learning effects.
            throw RelayInterpretationUnavailable(e)
        }
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
                val result = try {
                    withTimeout(90_000L) { executeTest(task, before) }
                } catch (_: TimeoutCancellationException) {
                    // Timeout is a failed test, never a reason to silently
                    // cancel the relay loop or execute the test again.
                    AIResult(success = false,
                        text = "Jessica не завершила тест за 90 секунд. Результат не подтверждён. " +
                            "Повторный запуск не выполнялся; передаю сбой ChatGPT.")
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    // Task dispatch may have reached the server. NEVER submit
                    // a second identical test on this uncertain outcome.
                    AIResult(success = false,
                        text = "Исполнение теста завершилось сбоем: " +
                            e.javaClass.simpleName + ". Автоматического повтора не было.")
                }
                withContext(Dispatchers.IO) { store.append(id, ChatMessage(result.text, false)) }
                val report = "Попытка выполнения теста через мой обычный чат завершена.\nЧат Jessica: $id\nНачало: $started\nОкончание: ${Instant.now()}\n" +
                    "Запрос обработан сервером: ${result.success}. Это не оценка прохождения теста.\n" +
                    "Точная задача:\n$task\n\nФактический ответ Jessica, без пересказа:\n${result.text}\n\n" +
                    "Автоматическая оценка полноты: ${RelayTestFeedback.note(task, result)}\n\n" +
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

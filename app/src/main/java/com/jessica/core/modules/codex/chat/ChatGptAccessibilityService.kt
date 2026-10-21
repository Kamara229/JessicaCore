package com.jessica.core.modules.codex.chat

import android.accessibilityservice.AccessibilityService
import android.app.KeyguardManager
import android.graphics.PixelFormat
import android.os.Bundle
import android.os.SystemClock
import android.view.Gravity
import android.view.WindowManager
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.widget.Button
import android.widget.LinearLayout
import com.jessica.core.modules.codex.CodexClient
import com.jessica.core.modules.codex.PhoneCodexHandoff
import kotlinx.coroutines.*
import org.json.JSONObject

/** Explicitly started, package-scoped UI assistant. No credentials or private HTTP endpoints from ChatGPT. */
class ChatGptAccessibilityService : AccessibilityService() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var loop: Job? = null
    private var run: ChatRunState? = null
    private var confirmed = false
    private var composed = false
    private var overlay: LinearLayout? = null
    private var confirmButton: Button? = null
    private var lastNotice = ""
    private val client = CodexClient()

    override fun onServiceConnected() { PhoneChatRuntime.controller = this }
    override fun onAccessibilityEvent(event: AccessibilityEvent?) { /* A bounded poll also handles quiet screens. */ }
    override fun onInterrupt() { stopByUser() }
    override fun onDestroy() {
        stopByUser()
        PhoneChatRuntime.controller = null
        scope.cancel()
        super.onDestroy()
    }

    fun begin() {
        loop?.cancel()
        confirmed = false
        composed = false
        val launch = PhoneChatRuntime.launch ?: return
        run = ChatRunState(SystemClock.elapsedRealtime(), maxTurns = if (launch.connectivityOnly) 1 else 3)
        showOverlay()
        if (!PhoneCodexHandoff.openChatGPT(this, launch.chatUrl.ifBlank { null })) {
            finish("ChatGPT не удалось открыть"); return
        }
        loop = scope.launch {
            delay(2000)
            while (isActive && PhoneChatRuntime.launch === launch) {
                try { tick(launch) }
                catch (error: CancellationException) { throw error }
                catch (_: Exception) { finish("Ошибка чтения интерфейса. Сессия остановлена; повторной отправки нет.") }
                delay(1500)
            }
        }
    }

    private suspend fun tick(launch: PhoneChatLaunch) {
        val state = run ?: return
        if (state.expired(SystemClock.elapsedRealtime())) { finish("Время ожидания истекло. Результат не подтверждён."); return }
        if (getSystemService(KeyguardManager::class.java).isKeyguardLocked) {
            stopByUser("Телефон заблокирован. Сессия остановлена."); return
        }
        val window = ChatWindowReader.read(this)
        if (window == null) {
            confirmed = false
            notice("Открой выбранный обычный чат. Для продолжения нажми «Чат выбран».")
            confirmButton?.visibility = android.view.View.VISIBLE
            return
        }
        try {
            if (window.mode == OrdinaryChatMode.WORK) {
                confirmed = false
                confirmButton?.visibility = android.view.View.VISIBLE
                notice("Выбран режим «Работа»/Codex. Переключи на обычный «Чат».")
                return
            }
            if (!confirmed) { notice("Выбери обычный чат и нажми «Чат выбран» на панели Джессики."); return }
            if (state.waiting) {
                val response = ChatUiPolicy.reply(window.content.takeLast(60000), launch.id)
                state.acceptReply(response, window.generating, SystemClock.elapsedRealtime())?.let { handleReply(launch, it) }
                return
            }
            val composer = window.composer ?: run { notice("Не найдено однозначное поле сообщения. Ничего не отправлено."); return }
            val text = composer.text?.toString().orEmpty()
            if (text.isNotBlank() && text != launch.prompt) {
                stopByUser("В поле есть другой текст. Джессика не будет его заменять."); return
            }
            if (state.sends > 0 && !window.content.contains("JESSICA_REPLY_${launch.id}")) {
                stopByUser("Не видна текущая переписка Джессики. Сессия остановлена."); return
            }
            // UNKNOWN is accepted only after the user explicitly confirms the visible ordinary-chat screen.
            val mode = if (confirmed && window.mode == OrdinaryChatMode.UNKNOWN) OrdinaryChatMode.CHAT else window.mode
            if (!ChatUiPolicy.canSend("com.openai.chatgpt", mode, false, text, launch.prompt, window.generating)) return
            if (!composed || text != launch.prompt) {
                val args = Bundle().apply { putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, launch.prompt) }
                if (!composer.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)) {
                    finish("ChatGPT не разрешил ввод текста через интерфейс. Ничего не отправлено."); return
                }
                composed = true
                notice("Запрос подготовлен; проверяю кнопку отправки.")
                return // Re-read the window after the composer changes.
            }
            val send = window.send ?: run { notice("Кнопка отправки не распознана. Нажатия по координатам отключены."); return }
            if (!state.beginSend(SystemClock.elapsedRealtime())) { finish("Достигнут лимит сообщений этой сессии."); return }
            // Record BEFORE clicking. An ambiguous result must never cause a duplicate send.
            notice("Отправляю запрос ${state.sends}/${if (launch.connectivityOnly) 1 else 3}. При перезапуске повторной отправки не будет.")
            if (!send.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                finish("Отправка не подтверждена интерфейсом. Проверь чат; автоматического повтора нет."); return
            }
            composed = false
            notice("Жду полный ответ ChatGPT (${state.sends}/${if (launch.connectivityOnly) 1 else 3}).")
        } finally { window.release() }
    }

    private suspend fun handleReply(launch: PhoneChatLaunch, text: String) {
        if (launch.connectivityOnly) {
            val ok = ChatConnectivityProbe.accepts(text, launch.id)
            finish(if (ok)
                "✅ Связь подтверждена: Джессика отправила тестовое сообщение и получила ответ из обычного ChatGPT. Ключи и GitHub не использовались."
                else "Ответ получен, но контрольная метка не совпала. Код не менялся.")
            return
        }
        PhoneChatRuntime.saveReply(this, launch.id, text)
        notice("Ответ получен. Проверяю формат и исходную версию кода.")
        val reply = try { JSONObject(text) } catch (_: Exception) { repairFormat(launch, "invalid-json"); return }
        try {
            val result = client.phone("submit", launch.ownerKey, JSONObject().put("id", launch.id).put("reply", reply))
            if (PhoneChatRuntime.launch !== launch) return
            if (result.optString("state") == "no-change") { finish("ChatGPT не предложил доказуемого исправления. Код не изменён."); return }
            PhoneChatRuntime.update(this, "Исправление в отдельной ветке. Выполняются независимые тесты.", true, result.optString("url"))
            while (PhoneChatRuntime.launch === launch) {
                delay(15000)
                if (getSystemService(KeyguardManager::class.java).isKeyguardLocked) {
                    stopByUser("Телефон заблокирован. Ожидание применения остановлено."); return
                }
                if (run?.expired(SystemClock.elapsedRealtime()) == true) { stopByUser("Истекло время сессии. Проверки можно посмотреть по ссылке."); return }
                val status = client.phone("status", launch.ownerKey, JSONObject().put("id", launch.id))
                if (PhoneChatRuntime.launch !== launch) return
                when (status.optString("state")) {
                    "testing" -> notice("Независимые тесты ещё выполняются.")
                    "merged" -> { finish("Исправление прошло тесты и применено в main. Развёртывание проверяется отдельно."); return }
                    "verified" -> { finish("Исправление прошло тесты. PR готов; автоматическое применение было выключено."); return }
                    "tests-failed" -> { finish("Тесты не подтвердили исправление. Основной код и успешный Experience не изменены."); return }
                    else -> { finish("Применение остановлено: " + status.optString("state") + ". Проверь PR."); return }
                }
            }
        } catch (error: CancellationException) { throw error }
        catch (error: Exception) {
            if (PhoneChatRuntime.launch !== launch) return
            val reason = error.message.orEmpty()
            val repairable = setOf("reply-does-not-match-session", "expected-source-change-and-one-new-regression", "invalid-or-duplicate-path",
                "protected-path", "invalid-source-content", "stale-blob", "unchanged-source", "new-test-sha-must-be-null", "regression-required", "patch-too-large")
            if (reason in repairable) repairFormat(launch, reason)
            else stopByUser("Связь с проверкой прервана: $reason. Повторное применение не запускается.")
        }
    }

    private fun repairFormat(launch: PhoneChatLaunch, reason: String) {
        if ((run?.sends ?: 3) >= 3) { finish("Формат ответа не подтверждён после трёх сообщений. Код не изменён."); return }
        launch.prompt = "Исправь формат предыдущего ответа: $reason. Сохрани исходный baseSha, nonce и разрешённые пути. Верни полный JSON. " +
            "Первая отдельная строка: JESSICA_REPLY_${launch.id}\nПоследняя отдельная строка: JESSICA_END_${launch.id}\n" +
            "Если корректное исправление невозможно, верни status no_change и files []. Оставайся в обычном чате."
        composed = false
        notice("Запрашиваю исправление формата ответа.")
    }

    fun stopByUser(message: String = "Остановлено пользователем") {
        val current = PhoneChatRuntime.launch
        run?.stop()
        loop?.cancel()
        finish(message)
        if (current != null && !current.connectivityOnly) scope.launch {
            try {
                val result = client.phone("stop", current.ownerKey, JSONObject().put("id", current.id))
                if (PhoneChatRuntime.launch == null && result.optString("state") in setOf("applying", "merged", "application-status-unknown"))
                    PhoneChatRuntime.update(this@ChatGptAccessibilityService, "Диалог остановлен. Применение уже началось или завершено; проверь PR.", false)
            } catch (_: Exception) {
                if (PhoneChatRuntime.launch == null)
                    PhoneChatRuntime.update(this@ChatGptAccessibilityService, "$message. Серверную остановку подтвердить не удалось; проверь PR.", false)
            }
        }
    }
    private fun notice(text: String) {
        if (lastNotice != text) { lastNotice = text; PhoneChatRuntime.update(this, text) }
    }
    private fun finish(text: String) {
        run?.stop()
        PhoneChatRuntime.update(this, text, false)
        PhoneChatRuntime.clear()
        overlay?.let { getSystemService(WindowManager::class.java).removeView(it) }
        overlay = null
    }
    private fun showOverlay() {
        if (overlay != null) return
        val layout = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        confirmButton = Button(this).apply {
            text = "Чат выбран — начать"
            setOnClickListener {
                val window = ChatWindowReader.read(this@ChatGptAccessibilityService)
                if (window != null) {
                    try {
                        if (window.mode != OrdinaryChatMode.WORK && window.composer != null) {
                            confirmed = true; visibility = android.view.View.GONE
                            notice("Обычный чат подтверждён пользователем. Запускаю сессию.")
                        } else notice("Сначала открой поле сообщения в обычном чате.")
                    } finally { window.release() }
                }
            }
        }
        layout.addView(confirmButton)
        layout.addView(Button(this).apply { text = "■ Стоп Джессика"; setOnClickListener { stopByUser() } })
        getSystemService(WindowManager::class.java).addView(layout, WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
            PixelFormat.TRANSLUCENT).apply { gravity = Gravity.TOP or Gravity.END; y = 180 })
        overlay = layout
    }
}

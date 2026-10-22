package com.jessica.core.modules.codex.chat

import android.accessibilityservice.AccessibilityService
import android.app.KeyguardManager
import android.content.Intent
import android.graphics.PixelFormat
import android.os.Bundle
import android.os.SystemClock
import android.view.Gravity
import android.view.WindowManager
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import com.jessica.core.modules.codex.PhoneCodexHandoff
import com.jessica.core.modules.codex.relay.RelayProtocol
import com.jessica.core.modules.codex.relay.ChatReplyCollector
import com.jessica.core.modules.codex.relay.ChatReplyExtraction
import kotlinx.coroutines.*
import java.util.UUID

/** Participant in one owner-selected ChatGPT conversation. No infrastructure-writing client. */
class ChatGptAccessibilityService : AccessibilityService() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var loop: Job? = null
    private var run: ChatRunState? = null
    private var confirmed = false
    private var composed = false
    private var overlay: LinearLayout? = null
    private var confirmButton: Button? = null
    private var statusLabel: TextView? = null
    private var calibrateButton: Button? = null
    private var calibratedSend: CalibratedSendTarget? = null
    private var calibrating = false
    private val sendPicker by lazy { SendTargetPicker(this) }
    private var lastNotice = ""
    private val replyCollector = ChatReplyCollector()
    private var verifiedReplyId = ""
    private var verifiedReplyAnchor = ""
    private var invisibleSince = 0L
    private var lastReplyWaitNotice = 0L

    override fun onCreate() { super.onCreate(); ChatServiceDiagnostics.record(this, "created") }
    override fun onServiceConnected() {
        super.onServiceConnected()
        PhoneChatRuntime.connected(this)
        ChatServiceDiagnostics.record(this, "connected")
    }
    override fun onAccessibilityEvent(event: AccessibilityEvent?) { }
    override fun onInterrupt() {
        if (PhoneChatRuntime.controller === this && PhoneChatRuntime.launch != null) stopByUser("Android прервал управление чатом.")
    }
    override fun onUnbind(intent: Intent?): Boolean {
        disconnect()
        ChatServiceDiagnostics.record(this, "unbound")
        return super.onUnbind(intent)
    }
    private fun disconnect() {
        if (PhoneChatRuntime.controller === this) {
            if (PhoneChatRuntime.launch != null) stopByUser("Android отключил службу. Кодекс остановлен.")
            PhoneChatRuntime.disconnected(this)
        }
    }
    override fun onDestroy() {
        disconnect()
        ChatServiceDiagnostics.record(this, "destroyed")
        scope.cancel()
        super.onDestroy()
    }

    fun begin() {
        loop?.cancel()
        confirmed = false
        composed = false
        calibratedSend = null
        calibrating = false
        sendPicker.close()
        lastNotice = ""
        replyCollector.reset()
        verifiedReplyId = ""
        verifiedReplyAnchor = ""
        invisibleSince = 0L
        lastReplyWaitNotice = 0L
        val launch = PhoneChatRuntime.launch ?: return
        run = ChatRunState(SystemClock.elapsedRealtime(), maxTurns = 60, durationMs = 8 * 60 * 60 * 1000L)
        try { showOverlay() } catch (e: Exception) {
            ChatServiceDiagnostics.record(this, "overlay-failed:${e.javaClass.simpleName}")
            finish("Не удалось показать панель Джессики. Открой диагностику службы."); return
        }
        if (!PhoneCodexHandoff.openChatGPT(this, null)) { finish("Открой официальное приложение ChatGPT и запусти Кодекс снова."); return }
        notice("Открой чат с планом и нажми «Чат выбран».")
        loop = scope.launch {
            delay(2000)
            while (isActive && PhoneChatRuntime.launch === launch) {
                try { tick(launch) }
                catch (e: CancellationException) { throw e }
                catch (e: Exception) { finish("Кодекс приостановлен: ${e.message ?: "не удалось продолжить"}. Повторная отправка и повторный тест не запускаются.") }
                delay(1500)
            }
        }
    }

    private suspend fun tick(launch: PhoneChatLaunch) {
        val state = run ?: return
        if (state.expired(SystemClock.elapsedRealtime())) { finish("Истекло время сессии или ожидания ответа. История сохранена."); return }
        if (getSystemService(KeyguardManager::class.java).isKeyguardLocked) { stopByUser("Телефон заблокирован. Кодекс остановлен."); return }
        if (calibrating) return
        val window = ChatWindowReader.read(this)
        if (window == null) {
            val now = SystemClock.elapsedRealtime()
            if (invisibleSince == 0L) invisibleSince = now
            if (now - invisibleSince < 12_000L) {
                // IME/keyboard and accessibility overlays briefly change the
                // active window. Do not silently forget the selected chat.
                notice("Ожидаю возвращения окна ChatGPT; выбранный чат сохранён.")
                return
            }
            confirmed = false
            calibratedSend = null
            calibrateButton?.visibility = android.view.View.GONE
            confirmButton?.visibility = android.view.View.VISIBLE
            notice("Чат ChatGPT не виден более 12 секунд. Вернись в тот же чат и нажми «Чат выбран».")
            return
        }
        invisibleSince = 0L
        var response: String? = null
        try {
            if (!confirmed) { notice("Открой подготовленный чат и нажми «Чат выбран»."); return }
            if (state.waiting) {
                val now = SystemClock.elapsedRealtime()
                val complete = replyCollector.observe(window.content, launch.replyId)
                response = state.acceptReply(complete, window.generating, now)
                if (response != null) {
                    verifiedReplyId = launch.replyId
                    verifiedReplyAnchor = response.replace(Regex("\\s+"), " ").trim().takeLast(140)
                }
                if (response == null) {
                    // Android ACTION_CLICK returning true does not mean ChatGPT
                    // actually sent anything. Never retry an uncertain click.
                    val draft = window.composer?.text?.toString().orEmpty()
                    if (now - state.sendAt > 15_000L && draft == launch.prompt &&
                        !ChatUiPolicy.alreadyPosted(window.content, launch.prompt, launch.replyId)) {
                        finish("ChatGPT не отправил подготовленный текст: он остался в поле ввода. " +
                            "Кодекс остановлен без повторной отправки. Проверь кнопку в выбранном чате.")
                        return
                    }
                    if (now - state.sendAt > 20_000L && now - lastReplyWaitNotice >= 20_000L) {
                        lastReplyWaitNotice = now
                        notice(if (window.generating) "ChatGPT формирует ответ. Жду завершения, ничего повторно не отправляю."
                            else if (complete == null) "Жду ответ ChatGPT с парой меток JESSICA_REPLY / JESSICA_END. " +
                                "Если они не появились, Кодекс не принимает непроверенный текст."
                            else "Метки ответа обнаружены. Проверяю, что ответ завершён и стабилен.")
                    }
                }
            } else {
                val composer = window.composer ?: run { notice("Не найдено однозначное поле сообщения. Отправки нет."); return }
                val text = composer.text?.toString().orEmpty()
                val alreadyPosted = RelayProtocol.reply(window.content, launch.replyId) != null ||
                    ChatUiPolicy.alreadyPosted(window.content, launch.prompt, launch.replyId)
                if (alreadyPosted) {
                    check(state.beginSend(SystemClock.elapsedRealtime())) { "Достигнут предел 60 сообщений" }
                    launch.conversation.recordOutgoing(launch.message)
                    if (launch.endAfterSend) finish("Переписка завершена. История и тесты сохранены в чатах Джессики.")
                    else notice("Сообщение уже в переписке. Читаю ответ без повторной отправки.")
                    return
                }
                if (composed && text != launch.prompt) { finish("Поле ввода изменилось. Повторного ввода и отправки не будет; проверь чат."); return }
                if (text.isNotBlank() && text != launch.prompt) { finish("В поле есть твой черновик. Убери его перед новым запуском Кодекса."); return }
                // Filling a long report may cover the preceding reply. Verify continuity before composing.
                if (!composed && state.sends > 0) {
                    // After a long answer ChatGPT may virtualize its START
                    // marker out of view, keeping only its END or last words.
                    // Never treat the owner's quoted marker instruction as proof.
                    val endVisible = ChatReplyExtraction.hasStandaloneEnd(
                        window.content, launch.previousReplyId
                    )
                    val replyTailVisible = verifiedReplyId == launch.previousReplyId &&
                        verifiedReplyAnchor.length >= 65 &&
                        window.content.replace(Regex("\\s+"), " ")
                            .contains(verifiedReplyAnchor)
                    if (!endVisible && !replyTailVisible) {
                        finish("Не вижу подтверждённый предыдущий ответ в выбранном чате. " +
                            "Кодекс остановлен: продолжение в другой переписке запрещено.")
                        return
                    }
                }
                if (!ChatUiPolicy.canSend("com.openai.chatgpt", confirmed, false, text, launch.prompt, window.generating)) return
                if (!composed) {
                    val args = Bundle().apply { putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, launch.prompt) }
                    check(composer.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)) { "ChatGPT не разрешил ввод текста" }
                    composed = true
                    notice("Сообщение подготовлено. Проверяю отправку.")
                    return
                }
                val send = window.send
                val point = if (send == null) calibratedSend?.point(window.root.screenBox(), composer.screenBox(), resources.displayMetrics.density) else null
                if (send == null && point == null) {
                    calibratedSend = null
                    calibrateButton?.visibility = android.view.View.VISIBLE
                    notice("Не удалось распознать стрелку отправки. Нажми «Указать стрелку» и покажи её один раз для этой сессии.")
                    return
                }
                calibrateButton?.visibility = android.view.View.GONE
                check(state.beginSend(SystemClock.elapsedRealtime())) { "Достигнут предел 60 сообщений" }
                // Mark before clicking: an uncertain click must not be automatically retried.
                val accepted = if (send != null) send.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                    else tapSelectedSend(point!!, launch)
                if (!accepted) { finish("Нажатие не подтверждено. Проверь чат; автоматического повтора нет. Если ты указывал стрелку, выключи и включи службу Jessica после обновления."); return }
                composed = false
                launch.conversation.recordOutgoing(launch.message)
                if (launch.endAfterSend) finish("Переписка завершена. История и тесты сохранены в чатах Джессики.")
                else notice("Жду законченный ответ ChatGPT. Сообщений в сессии: ${state.sends}/60.")
            }
        } finally { window.release() }
        val answer = response ?: return
        if (state.sends >= state.maxTurns) {
            launch.conversation.recordIncoming(answer)
            finish("Достигнут предел 60 сообщений. Последний ответ сохранён; новый тест не запускался.")
            return
        }
        notice("Ответ прочитан. Определяю следующий шаг по контексту.")
        val next = launch.conversation.respond(answer, ::notice)
        if (PhoneChatRuntime.launch !== launch) return
        launch.previousReplyId = launch.replyId
        launch.replyId = UUID.randomUUID().toString()
        launch.message = next.text
        launch.prompt = RelayProtocol.wrap(next.text, launch.replyId)
        launch.endAfterSend = next.endSession
        composed = false
        notice(if (next.endSession) "Передаю итоговое сообщение и завершаю сессию." else "Ответ Джессики подготовлен для выбранного чата.")
    }

    fun stopByUser(message: String = "Кодекс остановлен пользователем. История сохранена.") {
        loop?.cancel()
        finish(message)
    }
    private fun notice(text: String) {
        if (lastNotice != text) {
            lastNotice = text
            statusLabel?.text = text
            PhoneChatRuntime.update(this, text)
        }
    }
    private fun finish(text: String) {
        sendPicker.close()
        calibrating = false
        calibratedSend = null
        run?.stop()
        PhoneChatRuntime.update(this, text, false)
        PhoneChatRuntime.clear()
        overlay?.let { view -> runCatching { getSystemService(WindowManager::class.java).removeView(view) } }
        overlay = null
        statusLabel = null
        calibrateButton = null
    }

    private fun tapSelectedSend(point: Pair<Float,Float>, launch: PhoneChatLaunch): Boolean {
        return dispatchSendTap(this, point) {
            if (PhoneChatRuntime.launch === launch) stopByUser("Android прервал нажатие. Проверь чат; повторной отправки не будет.")
        }
    }

    private fun selectSendTarget(launch: PhoneChatLaunch) {
        if (PhoneChatRuntime.launch !== launch || !confirmed || !composed || run?.waiting != false) return
        calibrating = true
        try {
            sendPicker.show(onPick = { x,y ->
                calibrating = false
                val window = ChatWindowReader.read(this)
                try {
                    val editor = window?.composer
                    calibratedSend = if (PhoneChatRuntime.launch === launch && window != null && editor != null && editor.text?.toString() == launch.prompt && !window.generating &&
                        !getSystemService(KeyguardManager::class.java).isKeyguardLocked)
                        CalibratedSendTarget.select(window.root.screenBox(),editor.screenBox(),x,y,resources.displayMetrics.density) else null
                    notice(if (calibratedSend != null) "Стрелка выбрана. Отправляю подготовленное сообщение."
                        else "Точка не подходит. Нажми «Указать стрелку» и коснись синей стрелки справа от сообщения.")
                } finally { window?.release() }
            }, onCancel = { calibrating = false; notice("Выбор стрелки отменён. Сообщение не отправлено.") })
        } catch (e: Exception) { calibrating = false; finish("Не удалось показать выбор стрелки: ${e.message}") }
    }
    private fun showOverlay() {
        val launch = PhoneChatRuntime.launch ?: return
        val layout = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(0xffeeeeee.toInt()) }
        statusLabel = TextView(this).apply {
            text = "Джессика: выбери чат"
            setTextColor(0xff222222.toInt())
            maxLines = 3
            maxWidth = (resources.displayMetrics.density * 245).toInt()
            setPadding(12, 6, 12, 6)
        }
        layout.addView(statusLabel)
        confirmButton = Button(this).apply {
            text = "Чат выбран"
            setOnClickListener {
                val window = ChatWindowReader.read(this@ChatGptAccessibilityService)
                if (window == null) { notice("Сначала открой чат в приложении ChatGPT."); return@setOnClickListener }
                try {
                    if (window.composer != null) {
                        if (launch.conversation.plan.isBlank()) launch.conversation.plan = window.content.takeLast(12000)
                        confirmed = true
                        visibility = android.view.View.GONE
                        notice("Чат выбран. Джессика начинает переписку.")
                    } else notice("Открой переписку с полем сообщения.")
                } finally { window.release() }
            }
        }
        layout.addView(confirmButton)
        calibrateButton = Button(this).apply {
            text = "Указать стрелку"
            visibility = android.view.View.GONE
            setOnClickListener { selectSendTarget(launch) }
        }
        layout.addView(calibrateButton)
        layout.addView(Button(this).apply { text = "■ Стоп Джессика"; setOnClickListener { stopByUser() } })
        getSystemService(WindowManager::class.java).addView(layout, WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT, WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
            PixelFormat.TRANSLUCENT).apply { gravity = Gravity.TOP or Gravity.END; y = 150 })
        overlay = layout
    }
}

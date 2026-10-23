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
import kotlinx.coroutines.*
import java.util.UUID

/** Participant in one owner-selected ChatGPT conversation. No infrastructure-writing client. */
internal class CodexChatCoordinator(internal val service: ChatGptAccessibilityService) {
    internal val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    internal var loop: Job? = null
    internal var run: ChatRunState? = null
    internal var confirmed = false
    internal var composed = false
    internal var draftEditorMissingSince = 0L
    internal var continuityMissingSince = 0L
    internal val scrollFollower = CodexScrollFollower(service.resources.displayMetrics.density)
    internal var overlay: LinearLayout? = null
    internal var confirmButton: Button? = null
    internal var statusLabel: TextView? = null
    internal var calibrateButton: Button? = null
    internal var calibratedSend: CalibratedSendTarget? = null
    internal var calibrating = false
    internal val sendPicker by lazy { SendTargetPicker(service) }
    internal var lastNotice = ""
    internal val replyCollector = ChatReplyCollector()
    internal var verifiedReplyId = ""
    internal var verifiedReplyAnchor = ""
    internal var invisibleSince = 0L
    internal var lastReplyWaitNotice = 0L
    internal var lastReplyProgress: CodexReplyProgress? = null
    internal var lastReportedPhase = ""
    internal var pendingInterpretation: String? = null
    internal var interpretationFailures = 0
    internal var retryInterpretAt = 0L

    fun begin() {
        loop?.cancel()
        confirmed = false
        composed = false
        draftEditorMissingSince = 0L
        continuityMissingSince = 0L
        scrollFollower.reset()
        calibratedSend = null
        calibrating = false
        sendPicker.close()
        lastNotice = ""
        replyCollector.reset()
        verifiedReplyId = ""
        verifiedReplyAnchor = ""
        invisibleSince = 0L
        lastReplyWaitNotice = 0L
        lastReplyProgress = null
        lastReportedPhase = ""
        pendingInterpretation = null
        interpretationFailures = 0
        retryInterpretAt = 0L
        val launch = PhoneChatRuntime.launch ?: return
        run = ChatRunState(SystemClock.elapsedRealtime(), maxTurns = 60, durationMs = 8 * 60 * 60 * 1000L)
        reportPhase("started")
        try { showOverlay() } catch (e: Exception) {
            ChatServiceDiagnostics.record(service, "overlay-failed:${e.javaClass.simpleName}")
            finish("Не удалось показать панель Джессики. Открой диагностику службы."); return
        }
        if (!PhoneCodexHandoff.openChatGPT(service, null)) { finish("Открой официальное приложение ChatGPT и запусти Кодекс снова."); return }
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

    internal suspend fun tick(launch: PhoneChatLaunch) {
        val state = run ?: return
        if (state.expired(SystemClock.elapsedRealtime())) { finish("Истекло время сессии или ожидания ответа. История сохранена."); return }
        if (service.getSystemService(KeyguardManager::class.java).isKeyguardLocked) { stopByUser("Телефон заблокирован. Кодекс остановлен."); return }
        if (calibrating) return
        // Only remote interpretation is retried, never an uncertain send or
        // a partially executed Jessica test. No need to read a UI snapshot
        // while waiting for the backend's next interpretation attempt.
        pendingInterpretation?.let { answer ->
            if (SystemClock.elapsedRealtime() >= retryInterpretAt)
                processAcceptedReply(launch, answer, retryingInterpretation = true)
            return
        }
        val window = ChatWindowReader.read(service, if (composed || state.waiting) launch.prompt else null)
        if (window == null) {
            val now = SystemClock.elapsedRealtime()
            if (invisibleSince == 0L) invisibleSince = now
            if (now - invisibleSince < 12_000L) {
                // IME/keyboard and accessibility overlays briefly change the
                // active window. Do not silently forget the selected chat.
                notice("Ожидаю возвращения окна ChatGPT; выбранный чат сохранён.")
                return
            }
            if (state.waiting || state.sends > 0 || composed) {
                // Keyboard/IME changes and owner scrolling must not erase a
                // selected conversation after sending. Do nothing while the
                // verified ChatGPT window is absent; never touch another app.
                notice("Ожидаю окно выбранного ChatGPT. Сессия сохранена, повторной отправки нет.")
                return
            }
            confirmed = false
            calibratedSend = null
            calibrateButton?.visibility = android.view.View.GONE
            confirmButton?.visibility = android.view.View.VISIBLE
            notice("Чат ChatGPT не виден. Вернись в чат и нажми «Чат выбран».")
            return
        }
        invisibleSince = 0L
        var response: String? = null
        try {
            if (ChatUiPolicy.conversationAtLimit(window.content)) {
                finish("В выбранном чате ChatGPT достигнут предел длины обсуждения. " +
                    "Продолжение возможно только в новом чате, который ты выберешь самостоятельно. " +
                    "Джессика ничего не отправляла.")
                return
            }
            if (!confirmed) { notice("Открой подготовленный чат и нажми «Чат выбран»."); return }
            if (state.waiting) {
                response = observeIncoming(window, launch, state)
            } else {
                advanceOutgoing(window, launch, state)
            }
        } finally { window.release() }
        val answer = response ?: return
        processAcceptedReply(launch, answer, retryingInterpretation = false)
    }

    fun stopByUser(message: String = "Кодекс остановлен пользователем. История сохранена.") {
        loop?.cancel()
        finish(message)
    }
    /**
     * Phase-only diagnostics. Never upload a ChatGPT message, selected chat
     * title, source code, test text, user identity or device identifier.
     * Errors must not interrupt the phone relay or trigger a retry of a send.
     */


    internal fun notice(text: String) {
        if (lastNotice != text) {
            lastNotice = text
            statusLabel?.text = text
            PhoneChatRuntime.update(service, text)
        }
    }
    internal fun finish(text: String) {
        sendPicker.close()
        calibrating = false
        calibratedSend = null
        reportPhase("stopped")
        run?.stop()
        PhoneChatRuntime.update(service, text, false)
        PhoneChatRuntime.clear()
        overlay?.let { view -> runCatching { service.getSystemService(WindowManager::class.java).removeView(view) } }
        overlay = null
        statusLabel = null
        calibrateButton = null
    }





    internal fun shutdown() { loop?.cancel(); scope.cancel() }
}

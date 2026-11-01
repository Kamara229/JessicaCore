package com.jessica.core.modules.codex.chat

import android.os.SystemClock
import android.view.WindowManager
import com.jessica.core.modules.codex.PhoneCodexHandoff
import kotlinx.coroutines.*

internal fun CodexChatCoordinator.startCodexSession() {
        loop?.cancel()
        guardianJob?.cancel()
        while (windowWake.tryReceive().isSuccess) { }
        confirmed = false
        composed = false
        outgoingLogPending = false
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
        lengthBannerSeen = false
        lastReplyWaitNotice = 0L
        lastReplyProgress = null
        lastReportedPhase = ""
        shareDialogBackAttempted = false
        pendingInterpretation = null
        interpretationFailures = 0
        retryInterpretAt = 0L
        lastProgressAt = SystemClock.elapsedRealtime()
        idlePulseEnabled = false
        lastIdlePulseAttemptAt = lastProgressAt
        idlePulseButton = null
        val launch = PhoneChatRuntime.launch ?: return
        run = ChatRunState(SystemClock.elapsedRealtime(), maxTurns = 60, durationMs = 8 * 60 * 60 * 1000L)
        reportPhase("started")
        try { showOverlay() } catch (e: Exception) {
            ChatServiceDiagnostics.record(service, "overlay-failed:${e.javaClass.simpleName}")
            finish("Не удалось показать панель Джессики. Открой диагностику службы."); return
        }
        if (!PhoneCodexHandoff.openChatGPT(service, null)) { finish("Открой официальное приложение ChatGPT и запусти Кодекс снова."); return }
        notice("Открой чат с планом и нажми «Чат выбран».")
        guardianJob = startCodexGuardian(launch)
        loop = scope.launch {
            delay(2000)
            while (isActive && PhoneChatRuntime.launch === launch) {
                try { tick(launch) }
                catch (e: CancellationException) { throw e }
                catch (e: Exception) { finish("Кодекс приостановлен: ${e.message ?: "не удалось продолжить"}. Повторная отправка и повторный тест не запускаются.") }
                withTimeoutOrNull(1500L) { windowWake.receive() }
            }
        }
}
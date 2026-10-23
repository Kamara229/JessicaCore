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
    internal val windowWake = kotlinx.coroutines.channels.Channel<Unit>(
        kotlinx.coroutines.channels.Channel.CONFLATED
    )
    internal fun onChatWindowEvent() {
        if (PhoneChatRuntime.launch?.stopRequested == false) windowWake.trySend(Unit)
    }
    internal var lastReplyWaitNotice = 0L
    internal var lastReplyProgress: CodexReplyProgress? = null
    internal var lastReportedPhase = ""
    internal var pendingInterpretation: String? = null
    internal var interpretationFailures = 0
    internal var retryInterpretAt = 0L
    internal var lastProgressAt = 0L
    internal var guardianJob: Job? = null

    /** Lightweight facade. State and module orchestration stay separate. */
    fun begin() = startCodexSession()
    internal suspend fun tick(launch: PhoneChatLaunch) = tickCodexSession(launch)

    fun stopByUser(message: String = "Кодекс остановлен пользователем. История сохранена.") {
        PhoneChatRuntime.launch?.requestStop()
        loop?.cancel()
        guardianJob?.cancel()
        finish(message)
    }
    /**
     * Phase-only diagnostics. Never upload a ChatGPT message, selected chat
     * title, source code, test text, user identity or device identifier.
     * Errors must not interrupt the phone relay or trigger a retry of a send.
     */


    internal fun notice(text: String) {
        if (PhoneChatRuntime.launch?.stopRequested != false) return
        if (lastNotice != text) {
            lastNotice = text
            statusLabel?.text = text
            PhoneChatRuntime.update(service, text)
        }
    }
    internal fun finish(text: String) {
        PhoneChatRuntime.launch?.requestStop()
        loop?.cancel()
        guardianJob?.cancel()
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





    internal fun shutdown() { loop?.cancel(); guardianJob?.cancel(); scope.cancel() }
}

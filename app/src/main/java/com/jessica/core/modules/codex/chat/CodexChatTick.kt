package com.jessica.core.modules.codex.chat

import android.app.KeyguardManager
import android.os.SystemClock

internal suspend fun CodexChatCoordinator.tickCodexSession(launch: PhoneChatLaunch) {
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
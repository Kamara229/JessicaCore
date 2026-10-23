package com.jessica.core.modules.codex.chat

import android.app.KeyguardManager
import android.os.SystemClock

internal suspend fun CodexChatCoordinator.tickCodexSession(launch: PhoneChatLaunch) {
        val state = run ?: return
        if (state.expired(SystemClock.elapsedRealtime())) { finish("Истекло время сессии или ожидания ответа. История сохранена."); return }
        if (service.getSystemService(KeyguardManager::class.java).isKeyguardLocked) {
            notice("Телефон заблокирован. Кодекс сохраняет сессию и продолжит после разблокировки.")
            return
        }
        if (calibrating) return
        // Only remote interpretation is retried, never an uncertain send or
        // a partially executed Jessica test. No need to read a UI snapshot
        // while waiting for the backend's next interpretation attempt.
        pendingInterpretation?.let { answer ->
            if (SystemClock.elapsedRealtime() >= retryInterpretAt)
                processAcceptedReply(launch, answer, retryingInterpretation = true)
            return
        }
        val window = findSelectedChatWindow(launch, state) ?: return
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
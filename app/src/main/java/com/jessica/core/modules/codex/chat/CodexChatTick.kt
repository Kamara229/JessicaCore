package com.jessica.core.modules.codex.chat

import android.app.KeyguardManager
import android.os.SystemClock
import android.os.PowerManager

internal suspend fun CodexChatCoordinator.tickCodexSession(launch: PhoneChatLaunch) {
        if (launch.stopRequested) return
        if (launch.stopRequested || PhoneChatRuntime.launch !== launch) return
        val state = run ?: return
        if (state.expired(SystemClock.elapsedRealtime())) { finish("Истекло время сессии или ожидания ответа. История сохранена."); return }
        // Never synthesize touches or send from an off/locked screen.
        // The selected-chat overlay already requests FLAG_KEEP_SCREEN_ON.
        // Android/OEM behavior must be validated on the owner's actual phone.
        if (!service.getSystemService(PowerManager::class.java).isInteractive) {
            notice("Экран выключен. Кодекс ожидает разблокировки, действия в ChatGPT запрещены.")
            return
        }
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
        if (launch.stopRequested) { window.release(); return }
        var response: String? = null
        try {
            // ChatGPT's length banner is not a command to stop: this chat
            // may still accept messages. Observe it only, then continue the
            // usual composer, send-button and selected-thread safety checks.
            observeConversationBanner(window.content)
            if (!confirmed) { notice("Открой подготовленный чат и нажми «Чат выбран»."); return }
            // Never type or send through an obscuring share-link dialog.
            if (recoverShareLinkPrompt(window, launch)) return
            when (CodexTurnWindowPolicy.next(state.waiting, window.composer != null)) {
                CodexTurnWindowPolicy.Next.READ_REPLY ->
                    response = observeIncoming(window, launch, state)
                CodexTurnWindowPolicy.Next.WAIT_COMPOSER -> {
                    notice("Поле ввода выбранного чата скрыто. " +
                        "Сохраняю сессию без прокрутки и без отправки.")
                    return
                }
                CodexTurnWindowPolicy.Next.PREPARE_SEND -> {
                    if (launch.stopRequested) return
                    advanceOutgoing(window, launch, state)
                }
            }
        } finally { window.release() }
        val answer = response ?: return
        processAcceptedReply(launch, answer, retryingInterpretation = false)
}
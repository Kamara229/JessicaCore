package com.jessica.core.modules.codex.chat

import android.os.SystemClock
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Independent liveness observer. Network hangs or ChatGPT scroll glitches
 * cannot quietly conceal the stalled state. Never clicks, resends a prompt,
 * repeats a test or writes Experience.
 */
internal fun CodexChatCoordinator.startCodexGuardian(session: PhoneChatLaunch): Job =
    scope.launch {
        var lastAlert = CodexGuardianPolicy.Alert.NONE
        var lastNoticeAt = 0L
        while (isActive && PhoneChatRuntime.launch === session) {
            delay(6000L)
            val state = run ?: continue
            if (state.stopped) break
            val now = SystemClock.elapsedRealtime()
            val alert = CodexGuardianPolicy.evaluate(
                state.waiting, lastReportedPhase, now - lastProgressAt,
                invisibleSince != 0L, confirmed
            )
            if (CodexNetworkObserver.offline(service)) {
                if (now - lastNoticeAt >= 20_000L) {
                    notice("Сеть недоступна. Кодекс сохраняет сессию и проверит ответ после восстановления соединения.")
                    lastNoticeAt = now
                }
                onChatWindowEvent()
                continue
            }
            if (alert == CodexGuardianPolicy.Alert.NONE) {
                lastAlert = alert
                continue
            }
            if (alert != lastAlert || now - lastNoticeAt > 45_000L) {
                val status = when (alert) {
                    CodexGuardianPolicy.Alert.WAITING_REPLY ->
                        if (lastReportedPhase == "reply_out_of_view")
                            "Ответ частично скрыт экраном: конечная метка не видна. " +
                                "Для безопасности новых отправок нет."
                        else "Кодекс работает: ищу завершённый ответ ChatGPT. Повторной отправки нет."
                    CodexGuardianPolicy.Alert.INTERPRETING ->
                        "Кодекс работает: выполняется анализ или тест. Тест не дублирую."
                    CodexGuardianPolicy.Alert.PREPARING ->
                        "Кодекс работает: проверяю чат и ищу поле ввода следующего сообщения."
                    CodexGuardianPolicy.Alert.WINDOW_HIDDEN ->
                        "Кодекс сохраняет сессию и ожидает возвращения окна ChatGPT."
                    CodexGuardianPolicy.Alert.NONE, CodexGuardianPolicy.Alert.OFFLINE -> ""
                }
                if (status.isNotBlank()) notice(status)
                lastAlert = alert
                lastNoticeAt = now
            }
            // Wake the regular poll without assuming previous actions succeeded.
            onChatWindowEvent()
        }
    }

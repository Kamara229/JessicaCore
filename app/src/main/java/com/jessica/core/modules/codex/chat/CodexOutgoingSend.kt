package com.jessica.core.modules.codex.chat

import android.os.SystemClock
import android.view.accessibility.AccessibilityNodeInfo

/** One accessibility click only. Uncertain sends cannot be repeated. */
internal suspend fun CodexChatCoordinator.dispatchOutgoing(
    window: ChatWindow, launch: PhoneChatLaunch, state: ChatRunState,
    composer: AccessibilityNodeInfo
) {
                val send = window.send
                val point = if (send == null) calibratedSend?.point(window.root.screenBox(), composer.screenBox(), service.resources.displayMetrics.density) else null
                if (send == null && point == null) {
                    reportPhase("send_control_missing")
                    calibratedSend = null
                    calibrateButton?.visibility = android.view.View.VISIBLE
                    notice("Не удалось распознать стрелку отправки. Нажми «Указать стрелку» и покажи её один раз для этой сессии.")
                    return
                }
                calibrateButton?.visibility = android.view.View.GONE
                check(state.beginSend(SystemClock.elapsedRealtime())) { "Достигнут предел 60 сообщений" }
                // Mark before clicking: an uncertain click must not be automatically retried.
                reportPhase("send_requested")
                val accepted = if (send != null) send.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                    else tapSelectedSend(point!!, launch)
                if (!accepted) { finish("Нажатие не подтверждено. Проверь чат; автоматического повтора нет. Если ты указывал стрелку, выключи и включи службу Jessica после обновления."); return }
                composed = false
                launch.conversation.recordOutgoing(launch.message)
                reportPhase("sent")
                if (launch.endAfterSend) finish("Переписка завершена. История и тесты сохранены в чатах Джессики.")
                else notice("Жду законченный ответ ChatGPT. Сообщений в сессии: ${state.sends}/60.")
            
}

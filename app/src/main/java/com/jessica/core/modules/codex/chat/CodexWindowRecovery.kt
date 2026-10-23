package com.jessica.core.modules.codex.chat

import android.os.SystemClock

/**
 * Recover from Android notifications and temporarily hidden ChatGPT windows.
 * Keep the user-selected thread and pending turn. Never navigate, tap a
 * notification, select a different chat or send while the window is absent.
 */
internal fun CodexChatCoordinator.findSelectedChatWindow(
    launch: PhoneChatLaunch, state: ChatRunState
): ChatWindow? {
    val window=ChatWindowReader.read(service, if (composed || state.waiting) launch.prompt else null)
    val now=SystemClock.elapsedRealtime()
    if (window == null) {
        if (invisibleSince == 0L) {
            invisibleSince = now
            reportPhase("chat_away")
        }
        if (confirmed) {
            notice("Окно ChatGPT временно скрыто. Сессия сохранена; " +
                "продолжу автоматически при возвращении в выбранный чат.")
            return null
        }
        if (now - invisibleSince >= 12_000L)
            notice("Открой нужный чат ChatGPT и нажми «Чат выбран».")
        return null
    }
    if (invisibleSince != 0L) {
        invisibleSince = 0L
        if (confirmed) {
            reportPhase("chat_resumed")
            notice("Вижу окно ChatGPT снова. Продолжаю с сохранённого шага.")
        }
    }
    return window
}

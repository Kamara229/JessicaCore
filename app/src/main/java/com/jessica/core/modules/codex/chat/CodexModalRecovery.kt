package com.jessica.core.modules.codex.chat

import android.accessibilityservice.AccessibilityService

/**
 * Only an unmistakable share-link confirmation can be dismissed.
 * Never tap its affirmative button, change chats or send while it is up.
 * At most one BACK is issued per session; an ambiguous UI remains untouched.
 */
internal fun CodexChatCoordinator.recoverShareLinkPrompt(
    window: ChatWindow, launch: PhoneChatLaunch
): Boolean {
    if (!ChatModalRecoveryPolicy.isShareLinkPrompt(window.content)) return false
    if (!confirmed || launch.stopRequested || PhoneChatRuntime.launch !== launch) return true
    if (!shareDialogBackAttempted) {
        shareDialogBackAttempted = true
        val acknowledged = service.performGlobalAction(AccessibilityService.GLOBAL_ACTION_BACK)
        notice(if (acknowledged)
            "Закрываю окно «Поделиться ссылкой на чат?» без создания ссылки. Перепроверяю выбранный чат."
        else "Вижу окно «Поделиться ссылкой на чат?». Автоматически закрыть не удалось. " +
            "Ожидаю закрытия окна, не отправляя сообщений.")
    } else {
        notice("Окно «Поделиться ссылкой на чат?» ещё открыто. " +
            "Повторно кнопку «Назад» не нажимаю и сообщения не отправляю.")
    }
    return true
}

package com.jessica.core.modules.codex.chat

import android.os.SystemClock
import com.jessica.core.modules.codex.relay.ChatReplyExtraction

/**
 * Called on every live ChatGPT accessibility snapshot while a reply is due.
 * The UI state is observable at each milestone, but only a stable and fully
 * delimited answer may transition to interpreting and the next send.
 */
internal fun CodexChatCoordinator.observeIncoming(
    window: ChatWindow, launch: PhoneChatLaunch, state: ChatRunState
): String? {
    val now = SystemClock.elapsedRealtime()
    val completed = replyCollector.observe(window.content, launch.replyId)
    val accepted = state.acceptReply(completed, window.generating, now)
    val progress = CodexReplyProgressPolicy.classify(
        started = replyCollector.hasStarted(launch.replyId),
        completed = completed != null,
        generating = window.generating,
        accepted = accepted != null
    )
    if (progress != lastReplyProgress) {
        lastReplyProgress = progress
        reportPhase(progress.phase)
        notice(progress.status)
    }

    if (CodexReplyVisibilityPolicy.incompleteOutsideViewport(
            replyCollector.hasStarted(launch.replyId), completed != null,
            window.generating, now - state.sendAt
        )) {
        reportPhase("reply_out_of_view")
        notice("Вижу начало ответа, но конечная метка не доступна на экране. " +
            "Автопрокрутка отключена; не отправляю новое сообщение без проверки.")
    }

    if (accepted != null) {
        verifiedReplyId = launch.replyId
        verifiedReplyAnchor = accepted.replace(Regex("\\s+"), " ").trim().takeLast(140)
        return accepted
    }

    // Safety lock: never navigate or scroll the interface while waiting.
    // Missing reply markers are not evidence that a new chat should be opened.
    if (completed == null && ChatReplyExtraction.hasStandaloneEnd(window.content, launch.replyId)
        && !replyCollector.hasStarted(launch.replyId)) {
        notice("Ответ частично скрыт. Не прокручиваю и не меняю выбранный чат.")
    }

    // ACTION_CLICK success cannot prove that the message was actually posted.
    val draft = window.composer?.text?.toString().orEmpty()
    if (now - state.sendAt > 15_000L && draft == launch.prompt &&
        !ChatUiPolicy.alreadyPosted(window.content, launch.prompt, launch.replyId)) {
        // ACTION_CLICK returned but posting may be delayed or unacknowledged.
        // This is NOT permission to abort the whole Codex session or click again.
        notice("Черновик остался в поле ChatGPT. Сессия сохранена; " +
            "повторной отправки нет, проверяю появление ответа.")
        return null
    }

    return null
}

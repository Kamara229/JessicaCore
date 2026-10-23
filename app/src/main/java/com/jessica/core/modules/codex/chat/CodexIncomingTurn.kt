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

    if (accepted != null) {
        verifiedReplyId = launch.replyId
        verifiedReplyAnchor = accepted.replace(Regex("\\s+"), " ").trim().takeLast(140)
        return accepted
    }

    // Stay on the latest portion of the same, owner-selected ChatGPT thread
    // even after the owner manually scrolls elsewhere. No re-sends or taps.
    if (completed == null || window.generating) {
        val seekingStart = completed == null && scrollFollower.recoverStart(
            window, now, launch.replyId,
            replyCollector.hasStarted(launch.replyId),
            ChatReplyExtraction.hasStandaloneEnd(window.content, launch.replyId)
        )
        if (seekingStart) {
            reportPhase("reply_seeking_start")
            notice("Вижу конец ответа ChatGPT, восстанавливаю начало сообщения для анализа.")
        } else {
            scrollFollower.follow(window, now, launch.replyId)
        }
    }

    // ACTION_CLICK success cannot prove that the message was actually posted.
    val draft = window.composer?.text?.toString().orEmpty()
    if (now - state.sendAt > 15_000L && draft == launch.prompt &&
        !ChatUiPolicy.alreadyPosted(window.content, launch.prompt, launch.replyId)) {
        finish("ChatGPT оставил подготовленный текст в поле. Кодекс остановлен " +
            "без повторной отправки, чтобы не продублировать сообщение.")
        return null
    }

    return null
}

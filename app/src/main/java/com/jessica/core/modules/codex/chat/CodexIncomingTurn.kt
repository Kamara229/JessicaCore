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
    // Real Render telemetry showed reply_streaming -> awaiting_markers for
    // hours after an ordinary ChatGPT answer without our unique envelope.
    // The timeout is activated ONLY after the UI actually showed generation;
    // it never guesses answer text and NEVER sends a recovery prompt or test.
    if (accepted == null) {
        when (missingEnvelopePolicy.observe(
            now = now,
            currentSendAt = state.sendAt,
            generating = window.generating,
            startedMarker = replyCollector.hasStarted(launch.replyId),
            completedEnvelope = completed != null
        )) {
            CodexMissingEnvelopePolicy.Decision.STOP_UNMARKED -> {
                reportPhase("reply_unmarked_timeout")
                finish("ChatGPT сформировал ответ без проверяемых меток JESSICA_REPLY / " +
                    "JESSICA_END. Безопасно завершаю переписку без повторной " +
                    "отправки и без тестов. Исследование CALIBRI продолжается отдельно.")
                return null
            }
            CodexMissingEnvelopePolicy.Decision.STOP_PARTIAL -> {
                reportPhase("reply_partial_timeout")
                finish("Ответ ChatGPT виден лишь частично: конечная метка не " +
                    "подтверждена. Без прокрутки, повторной отправки и тестов " +
                    "завершаю сессию с понятным статусом.")
                return null
            }
            CodexMissingEnvelopePolicy.Decision.WAIT -> Unit
        }
    }
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

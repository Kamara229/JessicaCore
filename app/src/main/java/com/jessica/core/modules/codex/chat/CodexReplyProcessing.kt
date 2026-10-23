package com.jessica.core.modules.codex.chat

import android.os.SystemClock
import com.jessica.core.modules.codex.relay.RelayInterpretationUnavailable
import com.jessica.core.modules.codex.relay.RelayProtocol
import java.util.UUID

internal suspend fun CodexChatCoordinator.processAcceptedReply(
    launch: PhoneChatLaunch, answer: String, retryingInterpretation: Boolean
) {
    val state = run ?: return
    if (state.sends >= state.maxTurns) {
        if (!retryingInterpretation) launch.conversation.recordIncoming(answer)
        finish("Достигнут предел сообщений.")
        return
    }
    reportPhase("interpret_started")
    notice("Вижу ответ ChatGPT и анализирую.")
    val next = try {
        launch.conversation.respond(answer, ::notice, retryingInterpretation)
    } catch (_: RelayInterpretationUnavailable) {
        if (PhoneChatRuntime.launch !== launch) return
        pendingInterpretation = answer
        interpretationFailures++
        retryInterpretAt = SystemClock.elapsedRealtime() +
            5000L * interpretationFailures.coerceAtMost(12)
        reportPhase("interpret_retry")
        notice("Анализ временно недоступен. Безопасно повторяю анализ.")
        return
    }
    if (PhoneChatRuntime.launch !== launch) return
    pendingInterpretation = null
    interpretationFailures = 0
    retryInterpretAt = 0L
    launch.previousReplyId = launch.replyId
    launch.replyId = UUID.randomUUID().toString()
    launch.message = next.text
    launch.prompt = RelayProtocol.wrap(next.text, launch.replyId)
    launch.endAfterSend = next.endSession
    composed = false
    continuityMissingSince = 0L
    lastReplyProgress = null
    reportPhase("next_ready")
    notice(if (next.endSession) "Передаю итоговое сообщение."
        else "Ответ проанализирован. Готовлю следующее сообщение.")
}

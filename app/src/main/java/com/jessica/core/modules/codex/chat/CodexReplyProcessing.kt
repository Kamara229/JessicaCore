package com.jessica.core.modules.codex.chat

import android.os.SystemClock
import com.jessica.core.modules.codex.relay.RelayInterpretationUnavailable
import com.jessica.core.modules.codex.relay.RelayReply
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
    if (outgoingLogPending && !retryingInterpretation) {
        outgoingLogPending = false
        try {
            launch.conversation.recordOutgoing(launch.message)
        } catch (e: kotlinx.coroutines.CancellationException) {
            throw e
        } catch (_: Exception) {
            ChatServiceDiagnostics.record(service,"uncertain-journal-error")
        }
    }
    reportPhase("interpret_started")
    notice("Вижу ответ ChatGPT и анализирую.")
    var interpretationFailed = false
    val next = try {
        launch.conversation.respond(answer, retryingInterpretation, ::notice)
    } catch (error: RelayInterpretationUnavailable) {
        if (PhoneChatRuntime.launch !== launch) return
        interpretationFailures++
        if (!CodexInterpretationFailurePolicy.handoffNow(error, interpretationFailures)) {
            pendingInterpretation = answer
            retryInterpretAt = SystemClock.elapsedRealtime() + 4000L
            reportPhase("interpret_retry")
            notice("Сервис анализа недоступен. Одна безопасная повторная попытка.")
            return
        }
        // No interpretation => no test executed. Deliver the failure as the
        // next normal ChatGPT message (with markers and selected-chat checks).
        interpretationFailed = true
        reportPhase("interpret_failed")
        RelayReply(CodexInterpretationFailurePolicy.report(error))
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
    notice(if (interpretationFailed) "Анализ недоступен. Передаю ChatGPT отчёт о сбое, без повтора теста."
        else if (next.endSession) "Передаю итоговое сообщение."
        else "Ответ проанализирован. Готовлю следующее сообщение.")
}

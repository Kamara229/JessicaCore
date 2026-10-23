package com.jessica.core.modules.codex.chat

import android.os.SystemClock

internal fun CodexChatCoordinator.observeIncoming(window: ChatWindow, launch: PhoneChatLaunch, state: ChatRunState): String? {
    var response: String? = null

                val now = SystemClock.elapsedRealtime()
                val complete = replyCollector.observe(window.content, launch.replyId)
                response = state.acceptReply(complete, window.generating, now)
                if (response != null) {
                    reportPhase("reply_verified")
                    verifiedReplyId = launch.replyId
                    verifiedReplyAnchor = response.replace(Regex("\\s+"), " ").trim().takeLast(140)
                }
                if (response == null) {
                    // Recheck the current snapshot FIRST, then gently follow
                    // the visible conversation downward. No taps or chat switch.
                    if (complete == null || window.generating)
                        scrollFollower.follow(window, now, launch.replyId)
                    // Android ACTION_CLICK returning true does not mean ChatGPT
                    // actually sent anything. Never retry an uncertain click.
                    val draft = window.composer?.text?.toString().orEmpty()
                    if (now - state.sendAt > 15_000L && draft == launch.prompt &&
                        !ChatUiPolicy.alreadyPosted(window.content, launch.prompt, launch.replyId)) {
                        finish("ChatGPT не отправил подготовленный текст: он остался в поле ввода. " +
                            "Кодекс остановлен без повторной отправки. Проверь кнопку в выбранном чате.")
                        return null
                    }
                    if (now - state.sendAt > 20_000L && now - lastReplyWaitNotice >= 20_000L) {
                        lastReplyWaitNotice = now
                        reportPhase(if (window.generating) "reply_streaming"
                            else if (complete == null) "awaiting_markers"
                            else "reply_stabilizing")
                        notice(if (window.generating) "ChatGPT формирует ответ. Жду завершения, ничего повторно не отправляю."
                            else if (complete == null) "Жду ответ ChatGPT с парой меток JESSICA_REPLY / JESSICA_END. " +
                                "Если они не появились, Кодекс не принимает непроверенный текст."
                            else "Метки ответа обнаружены. Проверяю, что ответ завершён и стабилен.")
                    }
                }
            
    return response
}

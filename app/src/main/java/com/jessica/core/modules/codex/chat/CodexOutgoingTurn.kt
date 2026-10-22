package com.jessica.core.modules.codex.chat

import android.os.Bundle
import android.os.SystemClock
import android.view.accessibility.AccessibilityNodeInfo
import com.jessica.core.modules.codex.relay.RelayProtocol

internal suspend fun CodexChatCoordinator.advanceOutgoing(window: ChatWindow, launch: PhoneChatLaunch, state: ChatRunState) {

                val composer = window.composer ?: run {
                    notice("Не удалось определить поле ввода ChatGPT (найдено полей: ${window.editorCandidates}). " +
                        "Сообщение не отправлено. Проверь, что открыта переписка, а не поиск.")
                    return
                }
                val text = composer.text?.toString().orEmpty()
                val alreadyPosted = RelayProtocol.reply(window.content, launch.replyId) != null ||
                    ChatUiPolicy.alreadyPosted(window.content, launch.prompt, launch.replyId)
                if (alreadyPosted) {
                    check(state.beginSend(SystemClock.elapsedRealtime())) { "Достигнут предел 60 сообщений" }
                    launch.conversation.recordOutgoing(launch.message)
                    reportPhase("sent")
                    if (launch.endAfterSend) finish("Переписка завершена. История и тесты сохранены в чатах Джессики.")
                    else notice("Сообщение уже в переписке. Читаю ответ без повторной отправки.")
                    return
                }
                if (composed && text != launch.prompt) { finish("Поле ввода изменилось. Повторного ввода и отправки не будет; проверь чат."); return }
                if (text.isNotBlank() && text != launch.prompt) { finish("В поле есть твой черновик. Убери его перед новым запуском Кодекса."); return }
                // Filling a long report may cover the preceding reply. Verify continuity before composing.
                if (!composed && state.sends > 0) {
                    // After a long answer ChatGPT may virtualize its START
                    // marker out of view, keeping only its END or last words.
                    // Never treat the owner's quoted marker instruction as proof.
                    if (!ChatConversationContinuity.permits(
                            window.content, launch.previousReplyId, verifiedReplyId, verifiedReplyAnchor
                        )) {
                        finish("Не вижу подтверждённый предыдущий ответ в выбранном чате. " +
                            "Кодекс остановлен: продолжение в другой переписке запрещено.")
                        return
                    }
                }
                if (!ChatUiPolicy.canSend("com.openai.chatgpt", confirmed, false, text, launch.prompt, window.generating)) return
                if (!composed) {
                    val args = Bundle().apply { putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, launch.prompt) }
                    check(composer.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)) { "ChatGPT не разрешил ввод текста" }
                    composed = true
                    reportPhase("draft_ready")
                    notice("Сообщение подготовлено. Проверяю отправку.")
                    return
                }
                val send = window.send
                val point = if (send == null) calibratedSend?.point(window.root.screenBox(), composer.screenBox(), service.resources.displayMetrics.density) else null
                if (send == null && point == null) {
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

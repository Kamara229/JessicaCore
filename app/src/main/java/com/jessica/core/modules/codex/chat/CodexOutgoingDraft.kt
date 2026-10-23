package com.jessica.core.modules.codex.chat

import android.os.Bundle
import android.os.SystemClock
import android.view.accessibility.AccessibilityNodeInfo
import com.jessica.core.modules.codex.relay.RelayProtocol

/**
 * Verify conversation continuity and prepare the exact draft. Returns true
 * when this tick is handled and must not proceed to send.
 */
internal suspend fun CodexChatCoordinator.prepareOutgoingDraft(
    window: ChatWindow, launch: PhoneChatLaunch, state: ChatRunState,
    composer: AccessibilityNodeInfo
): Boolean {
    val text = composer.text?.toString().orEmpty()
                val alreadyPosted = RelayProtocol.reply(window.content, launch.replyId) != null ||
                    ChatUiPolicy.alreadyPosted(window.content, launch.prompt, launch.replyId)
                if (alreadyPosted) {
                    check(state.beginSend(SystemClock.elapsedRealtime())) { "Достигнут предел 60 сообщений" }
                    launch.conversation.recordOutgoing(launch.message)
                    reportPhase("sent")
                    if (launch.endAfterSend) finish("Переписка завершена. История и тесты сохранены в чатах Джессики.")
                    else notice("Сообщение уже в переписке. Читаю ответ без повторной отправки.")
                    return true
                }
                if (composed && text != launch.prompt) {
                    finish("Кодекс остановлен: текст доступного поля не совпадает с черновиком " +
                        "(получено ${text.length} символов из ${launch.prompt.length}). " +
                        "Отправка без проверки полного текста запрещена.")
                    return true
                }
                if (text.isNotBlank() && text != launch.prompt) { finish("В поле есть твой черновик. Убери его перед новым запуском Кодекса."); return true }
                // Filling a long report may cover the preceding reply. Verify continuity before composing.
                if (!composed && state.sends > 0) {
                    // After a long answer ChatGPT may virtualize its START
                    // marker out of view, keeping only its END or last words.
                    // Never treat the owner's quoted marker instruction as proof.
                    if (!ChatConversationContinuity.permits(
                            window.content, launch.previousReplyId, verifiedReplyId, verifiedReplyAnchor
                        )) {
                        val now = SystemClock.elapsedRealtime()
                        if (continuityMissingSince == 0L) continuityMissingSince = now
                        // Never end Codex merely because the owner scrolled
                        // the already verified answer out of the visible tree.
                        // Keep following the current ChatGPT viewport, but DO NOT
                        // compose or send until continuity is visible again.
                        scrollFollower.follow(window, now, launch.previousReplyId)
                        notice(if (now - continuityMissingSince < 30_000L)
                            "Прокручиваю переписку вниз для проверки предыдущего ответа. Отправки нет."
                            else "Кодекс продолжает искать подтверждённый ответ в выбранном чате. " +
                                "Пока он не виден, новые сообщения не отправляются.")
                        return true
                    }
                    continuityMissingSince = 0L
                }
                if (!ChatUiPolicy.canSend("com.openai.chatgpt", confirmed, false, text, launch.prompt, window.generating)) return true
                if (!composed) {
                    val args = Bundle().apply { putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, launch.prompt) }
                    check(composer.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)) { "ChatGPT не разрешил ввод текста" }
                    composed = true
                    reportPhase("draft_ready")
                    notice("Сообщение подготовлено. Проверяю отправку.")
                    return true
                }
    return false
}

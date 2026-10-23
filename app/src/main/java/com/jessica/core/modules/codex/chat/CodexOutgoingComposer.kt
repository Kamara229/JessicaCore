package com.jessica.core.modules.codex.chat

import android.os.SystemClock
import android.view.accessibility.AccessibilityNodeInfo

/** Recover only the selected ChatGPT composer; never send during recovery. */
internal fun CodexChatCoordinator.resolveOutgoingComposer(
    window: ChatWindow, launch: PhoneChatLaunch, state: ChatRunState
): AccessibilityNodeInfo? {
                val composer = window.composer ?: run {
                    if (!composed && state.sends > 0) {
                        // Scrolling the transcript may temporarily hide the
                        // pinned composer after a completed answer. Recover it
                        // by scrolling the same selected chat to its newest
                        // messages, without ever touching a draft or sending.
                        // UI is not scrolled automatically; a drawer must never be touched.
                        notice("Поле ввода скрыто. Ожидаю возвращения в выбранный чат без автопрокрутки.")
                        return null
                    }
                    if (composed) {
                        reportPhase("editor_missing")
                        val now = SystemClock.elapsedRealtime()
                        if (draftEditorMissingSince == 0L) draftEditorMissingSince = now
                        if (now - draftEditorMissingSince > 20_000L) {
                            finish("Кодекс остановлен: подготовленный текст не удалось проверить за 20 секунд. " +
                                "Причина: ${window.composerDiagnostic}. Автоматической повторной отправки нет.")
                            return null
                        }
                    }
                    notice("Не удалось определить поле ввода ChatGPT (найдено полей: ${window.editorCandidates}). " +
                        "Причина: ${window.composerDiagnostic}. Сообщение не отправлено.")
                    return null
                }
    draftEditorMissingSince = 0L
    return composer
}

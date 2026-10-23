package com.jessica.core.modules.codex.chat

/**
 * A ChatGPT length notice is advisory information about the thread, not an
 * instruction from its owner and not proof that the composer is unusable.
 * This observer never switches chats, presses buttons or sends messages.
 */
internal fun CodexChatCoordinator.observeConversationBanner(content: String) {
    if (lengthBannerSeen || !ChatUiPolicy.conversationAtLimit(content)) return
    lengthBannerSeen = true
    reportPhase("conversation_length_notice")
    notice("ChatGPT показал уведомление о длине обсуждения. Кодекс остаётся " +
        "в выбранном чате и продолжает работу, пока доступна отправка. " +
        "Автоматически новый чат не открываю.")
}

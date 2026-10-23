package com.jessica.core.modules.codex.chat

/** Small outgoing-turn facade over composer, draft and send submodules. */
internal suspend fun CodexChatCoordinator.advanceOutgoing(
    window: ChatWindow, launch: PhoneChatLaunch, state: ChatRunState
) {
    val composer = resolveOutgoingComposer(window, launch, state) ?: return
    if (prepareOutgoingDraft(window, launch, state, composer)) return
    dispatchOutgoing(window, launch, state, composer)
}

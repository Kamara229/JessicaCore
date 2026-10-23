package com.jessica.core.modules.codex.chat

/**
 * Accessibility trees can become stale between Android notifications and
 * ChatGPT redraws. Reading a snapshot has no side effects, so it may be
 * retried on the next poll without losing a session or duplicating a send.
 */
internal fun CodexChatCoordinator.readGuardedChatWindow(prepared: String?): ChatWindow? =
    try {
        ChatWindowReader.read(service,prepared)
    } catch (e: Exception) {
        reportPhase("window_snapshot_retry")
        notice("Интерфейс ChatGPT изменился во время чтения. Повторно проверяю окно.")
        null
    }

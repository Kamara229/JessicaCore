package com.jessica.core.modules.codex.chat

import android.accessibilityservice.AccessibilityService
import android.content.Intent
import android.view.accessibility.AccessibilityEvent

/** Android lifecycle facade; send, reply, overlay, calibration and telemetry are small modules. */
class ChatGptAccessibilityService : AccessibilityService() {
    private val coordinator by lazy { CodexChatCoordinator(this) }
    override fun onCreate() { super.onCreate(); ChatServiceDiagnostics.record(this, "created") }
    override fun onServiceConnected() {
        super.onServiceConnected()
        PhoneChatRuntime.connected(this)
        ChatServiceDiagnostics.record(this, "connected")
    }
    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}
    override fun onInterrupt() {
        // Android may interrupt AccessibilityService feedback on a user swipe.
        // It does not mean the owner ended the Codex session. Actual disconnect
        // and explicit Stop are handled separately.
        ChatServiceDiagnostics.record(this, "feedback-interrupted")
    }
    override fun onUnbind(intent: Intent?): Boolean {
        disconnect()
        ChatServiceDiagnostics.record(this, "unbound")
        return super.onUnbind(intent)
    }
    private fun disconnect() {
        if (PhoneChatRuntime.controller === this) {
            if (PhoneChatRuntime.launch != null)
                stopByUser("Android отключил службу. Кодекс остановлен.")
            PhoneChatRuntime.disconnected(this)
        }
    }
    override fun onDestroy() {
        disconnect()
        coordinator.shutdown()
        ChatServiceDiagnostics.record(this, "destroyed")
        super.onDestroy()
    }
    fun begin() = coordinator.begin()
    fun stopByUser(message: String = "Кодекс остановлен пользователем. История сохранена.") =
        coordinator.stopByUser(message)
}

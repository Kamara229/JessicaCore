package com.jessica.core.modules.codex.chat

import android.accessibilityservice.AccessibilityServiceInfo
import android.content.ComponentName
import android.content.Context
import android.provider.Settings
import android.view.accessibility.AccessibilityManager

/** Distinguish Android permission from an actually bound service process. */
enum class ChatServiceState {
    CONNECTED, ENABLED_NOT_CONNECTED, DISABLED, NOT_REGISTERED
}

data class ChatServiceSnapshot(
    val state: ChatServiceState,
    val registered: Boolean,
    val enabled: Boolean,
    val connected: Boolean
)

/** Pure classification used by tests and the Android diagnostics screen. */
fun classifyChatService(registered: Boolean, enabled: Boolean, connected: Boolean): ChatServiceState =
    when {
        connected -> ChatServiceState.CONNECTED
        !registered -> ChatServiceState.NOT_REGISTERED
        enabled -> ChatServiceState.ENABLED_NOT_CONNECTED
        else -> ChatServiceState.DISABLED
}

object ChatServiceDiagnostics {
    fun inspect(context: Context): ChatServiceSnapshot {
        val component = ComponentName(context, ChatGptAccessibilityService::class.java)
        val manager = context.getSystemService(Context.ACCESSIBILITY_SERVICE) as AccessibilityManager
        val registered = try {
            manager.installedAccessibilityServiceList.any { service ->
                ComponentName.unflattenFromString(service.id) == component
            }
        } catch (_: Exception) { false }
        val enabledByManager = try {
            manager.getEnabledAccessibilityServiceList(AccessibilityServiceInfo.FEEDBACK_ALL_MASK)
                .any { service -> ComponentName.unflattenFromString(service.id) == component }
        } catch (_: Exception) { false }
        val enabledBySettings = try {
            Settings.Secure.getString(
                context.contentResolver, Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES
            ).orEmpty().split(':').any { value ->
                ComponentName.unflattenFromString(value) == component
            }
        } catch (_: Exception) { false }
        val enabled = enabledByManager || enabledBySettings
        val connected = PhoneChatRuntime.controller != null
        return ChatServiceSnapshot(
            state = classifyChatService(registered, enabled, connected),
            registered = registered, enabled = enabled, connected = connected
        )
    }
}

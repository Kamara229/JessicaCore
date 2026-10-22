package com.jessica.core.modules.codex.chat

import android.accessibilityservice.AccessibilityServiceInfo
import android.app.ActivityManager
import android.os.Build
import com.jessica.core.BuildConfig
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
) {
    val message: String get() = when (state) {
        ChatServiceState.CONNECTED -> "Служба подключена"
        ChatServiceState.ENABLED_NOT_CONNECTED -> "Разрешение включено, но Android ещё не подключил службу Jessica."
        ChatServiceState.DISABLED -> "Служба выключена в специальных возможностях Android."
        ChatServiceState.NOT_REGISTERED -> "Android не обнаружил службу Jessica. Обнови приложение поверх установленного."
    }
}

/** Pure classification used by tests and the Android diagnostics screen. */
fun classifyChatService(registered: Boolean, enabled: Boolean, connected: Boolean): ChatServiceState =
    when {
        connected -> ChatServiceState.CONNECTED
        enabled -> ChatServiceState.ENABLED_NOT_CONNECTED
        !registered -> ChatServiceState.NOT_REGISTERED
        else -> ChatServiceState.DISABLED
}

object ChatServiceDiagnostics {
    private const val PREFS = "jessica_chat_service"
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
    fun record(context: Context, event: String) {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val entries = prefs.getString("events", "").orEmpty().lines().filter { it.isNotBlank() }
        prefs.edit().putString("events", (entries + "${System.currentTimeMillis()} v${BuildConfig.VERSION_NAME} $event")
            .takeLast(12).joinToString("\n")).apply()
    }

    fun report(context: Context): String {
        val state = inspect(context)
        val component = ComponentName(context, ChatGptAccessibilityService::class.java)
        val listed = runCatching {
            context.getSystemService(AccessibilityManager::class.java)
                .getEnabledAccessibilityServiceList(AccessibilityServiceInfo.FEEDBACK_ALL_MASK)
                .any { ComponentName.unflattenFromString(it.id) == component }
        }.getOrNull()
        val exits = if (Build.VERSION.SDK_INT >= 30) runCatching {
            context.getSystemService(ActivityManager::class.java)
                .getHistoricalProcessExitReasons(context.packageName, 0, 3)
                .joinToString("\n") { "exit: time=${it.timestamp} reason=${it.reason} status=${it.status}" }
        }.getOrDefault("exit: unavailable") else ""
        return "Jessica ${BuildConfig.VERSION_NAME}; Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})\n" +
            "registered=${state.registered}; enabled=${state.enabled}; systemListed=$listed; connected=${state.connected}\n" +
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString("events", "Нет событий запуска службы") +
            "\n" + exits
    }
}

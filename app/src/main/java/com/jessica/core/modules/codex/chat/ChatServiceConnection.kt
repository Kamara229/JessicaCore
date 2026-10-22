package com.jessica.core.modules.codex.chat

import android.accessibilityservice.AccessibilityServiceInfo
import android.app.ActivityManager
import android.content.ComponentName
import android.content.Context
import android.os.Build
import android.provider.Settings
import android.view.accessibility.AccessibilityManager
import com.jessica.core.BuildConfig

data class ChatServiceConnection(val enabled: Boolean?, val connected: Boolean) {
    val message: String get() = when {
        connected -> "Служба подключена"
        enabled == true -> "Служба включена в Android, но ещё не подключена к Джессике."
        enabled == false -> "Служба выключена в специальных возможностях Android."
        else -> "Не удалось прочитать состояние службы Android."
    }
}

/** Diagnostic data contains lifecycle codes only, never screen text, prompts or credentials. */
object ChatServiceDiagnostics {
    private const val PREFS = "jessica_chat_service"
    fun read(context: Context): ChatServiceConnection {
        val component = ComponentName(context, ChatGptAccessibilityService::class.java)
        val enabled = runCatching {
            Settings.Secure.getString(context.contentResolver, Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES)
                .orEmpty().split(':').any { ComponentName.unflattenFromString(it) == component }
        }.getOrNull()
        return ChatServiceConnection(enabled, PhoneChatRuntime.controller != null)
    }

    fun record(context: Context, event: String) {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val entries = prefs.getString("events", "").orEmpty().lines().filter { it.isNotBlank() }
        prefs.edit().putString("events", (entries + "${System.currentTimeMillis()} v${BuildConfig.VERSION_NAME} $event")
            .takeLast(12).joinToString("\n")).apply()
    }

    fun report(context: Context): String {
        val state = read(context)
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
            "enabled=${state.enabled}; systemListed=$listed; connected=${state.connected}\n" +
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString("events", "Нет событий запуска службы") +
            "\n" + exits
    }
}

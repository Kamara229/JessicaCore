package com.jessica.core.modules.codex.chat

import android.app.UiAutomation
import android.content.ComponentName
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** Runs only on a disposable CI emulator; never enables permissions in production. */
@RunWith(AndroidJUnit4::class)
class ChatServiceBindingTest {
    @Test fun systemBindsDisconnectsAndReconnectsWithoutStartingAChat() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val automation = instrumentation.getUiAutomation(UiAutomation.FLAG_DONT_SUPPRESS_ACCESSIBILITY_SERVICES)
        fun shell(command: String) {
            android.os.ParcelFileDescriptor.AutoCloseInputStream(automation.executeShellCommand(command))
                .use { it.readBytes() }
        }
        fun awaitConnection(expected: Boolean) {
            val deadline = SystemClock.elapsedRealtime() + 15000
            while ((PhoneChatRuntime.controller != null) != expected && SystemClock.elapsedRealtime() < deadline) {
                SystemClock.sleep(100)
            }
            assertEquals(ChatServiceDiagnostics.report(context), expected, PhoneChatRuntime.controller != null)
        }
        val component = ComponentName(context, ChatGptAccessibilityService::class.java).flattenToString()
        try {
            shell("settings put secure enabled_accessibility_services $component")
            shell("settings put secure accessibility_enabled 1")
            awaitConnection(true)
            assertEquals(true, ChatServiceDiagnostics.inspect(context).enabled)
            assertNull("Binding alone must never send a message", PhoneChatRuntime.launch)
            shell("settings delete secure enabled_accessibility_services")
            awaitConnection(false)
            assertEquals(false, ChatServiceDiagnostics.inspect(context).enabled)
            shell("settings put secure enabled_accessibility_services $component")
            shell("settings put secure accessibility_enabled 1")
            awaitConnection(true)
            assertNull(PhoneChatRuntime.launch)
        } finally {
            shell("settings delete secure enabled_accessibility_services")
            shell("settings put secure accessibility_enabled 0")
        }
    }
}

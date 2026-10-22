package com.jessica.core.modules.codex.chat

import android.app.UiAutomation
import android.content.Intent
import android.content.ComponentName
import android.os.SystemClock
import android.view.accessibility.AccessibilityNodeInfo
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ChatSendButtonTest {
    @Test fun childLabelResolvesToARealClickableParentAndSendsOnce() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val automation = instrumentation.getUiAutomation(UiAutomation.FLAG_DONT_SUPPRESS_ACCESSIBILITY_SERVICES)
        instrumentation.runOnMainSync {
            context.startActivity(Intent(context,ChatSendFixtureActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        }
        val deadline = SystemClock.elapsedRealtime() + 10000
        var window: ChatWindow? = null
        while (SystemClock.elapsedRealtime() < deadline) {
            val root = automation.rootInActiveWindow
            if (root != null) {
                val candidate = ChatWindowReader.inspect(root,context.resources.displayMetrics.density)
                if (candidate.composer?.text?.toString() == "Проверка вложенной стрелки" && candidate.send != null) {
                    window = candidate; break
                }
                candidate.release()
            }
            SystemClock.sleep(100)
        }
        val found = requireNotNull(window) { "Nested send control was not found in the Android view hierarchy" }
        try {
            assertTrue(found.send!!.performAction(AccessibilityNodeInfo.ACTION_CLICK))
            instrumentation.waitForIdleSync()
            val clickDeadline = SystemClock.elapsedRealtime() + 3000
            while (ChatSendFixtureActivity.clicks.get() == 0 && SystemClock.elapsedRealtime() < clickDeadline) SystemClock.sleep(50)
            assertEquals(1,ChatSendFixtureActivity.clicks.get())
            assertEquals("Проверка вложенной стрелки",ChatSendFixtureActivity.sentText)
        } finally { found.release(); automation.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK) }
    }
    @Test fun selectedPointUsesRealAccessibilityGestureWithoutDuplicateTap() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val automation = instrumentation.getUiAutomation(UiAutomation.FLAG_DONT_SUPPRESS_ACCESSIBILITY_SERVICES)
        fun shell(command: String) {
            android.os.ParcelFileDescriptor.AutoCloseInputStream(automation.executeShellCommand(command)).use { it.readBytes() }
        }
        val component = ComponentName(context,ChatGptAccessibilityService::class.java).flattenToString()
        try {
            shell("settings put secure enabled_accessibility_services $component")
            shell("settings put secure accessibility_enabled 1")
            val deadline = SystemClock.elapsedRealtime() + 15000
            while (PhoneChatRuntime.controller == null && SystemClock.elapsedRealtime() < deadline) SystemClock.sleep(100)
            val service = requireNotNull(PhoneChatRuntime.controller)
            instrumentation.runOnMainSync {
                context.startActivity(Intent(context,ChatSendFixtureActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            }
            instrumentation.waitForIdleSync()
            var target: Pair<Float,Float>? = null
            val uiDeadline = SystemClock.elapsedRealtime() + 10000
            while (target == null && SystemClock.elapsedRealtime() < uiDeadline) {
                automation.rootInActiveWindow?.let { root ->
                    val window = ChatWindowReader.inspect(root,context.resources.displayMetrics.density)
                    try {
                        val box = window.send?.screenBox()
                        if (box != null && window.composer?.text?.toString() == "Проверка вложенной стрелки") target = box.centerX to box.centerY
                    } finally { window.release() }
                }
                if (target == null) SystemClock.sleep(100)
            }
            val point = requireNotNull(target)
            val picked = java.util.concurrent.atomic.AtomicReference<Pair<Float,Float>?>(null)
            val picker = SendTargetPicker(service)
            try {
                instrumentation.runOnMainSync { picker.show(onPick = { x,y -> picked.set(x to y) }, onCancel = {}) }
                instrumentation.waitForIdleSync()
                val now = SystemClock.uptimeMillis()
                val down = android.view.MotionEvent.obtain(now,now,android.view.MotionEvent.ACTION_DOWN,point.first,point.second,0)
                val up = android.view.MotionEvent.obtain(now,now+80,android.view.MotionEvent.ACTION_UP,point.first,point.second,0)
                down.source = android.view.InputDevice.SOURCE_TOUCHSCREEN
                up.source = android.view.InputDevice.SOURCE_TOUCHSCREEN
                try { automation.injectInputEvent(down,true); automation.injectInputEvent(up,true) }
                finally { down.recycle(); up.recycle() }
                instrumentation.waitForIdleSync()
                assertNotNull("Owner selection must reach the overlay",picked.get())
                assertEquals("Selecting the arrow must not send the message",0,ChatSendFixtureActivity.clicks.get())
            } finally { instrumentation.runOnMainSync { picker.close() } }
            // Overlay removal and window relayout are asynchronous on Android.
            // Resolve the current clickable target again, as the real Codex
            // does on its next periodic accessibility snapshot.
            SystemClock.sleep(450)
            var freshPoint: Pair<Float, Float>? = null
            val readyUntil = SystemClock.elapsedRealtime() + 5000
            while (freshPoint == null && SystemClock.elapsedRealtime() < readyUntil) {
                automation.rootInActiveWindow?.let { root ->
                    val view = ChatWindowReader.inspect(root, context.resources.displayMetrics.density)
                    try {
                        if (view.composer?.text?.toString() == "Проверка вложенной стрелки") {
                            freshPoint = view.send?.screenBox()?.let { it.centerX to it.centerY }
                        }
                    } finally { view.release() }
                }
                if (freshPoint == null) SystemClock.sleep(100)
            }
            val currentTarget = requireNotNull(freshPoint) { "Send button missing after selector overlay dismissal" }
            var accepted = false
            val cancelled = java.util.concurrent.atomic.AtomicBoolean(false)
            instrumentation.runOnMainSync { accepted = dispatchSendTap(service,currentTarget) { cancelled.set(true) } }
            assertTrue("Android must grant the declared gesture capability",accepted)
            val tapDeadline = SystemClock.elapsedRealtime() + 3000
            while (ChatSendFixtureActivity.clicks.get() == 0 && SystemClock.elapsedRealtime() < tapDeadline) SystemClock.sleep(50)
            assertFalse(cancelled.get())
            assertEquals(1,ChatSendFixtureActivity.clicks.get())
            assertEquals("Проверка вложенной стрелки",ChatSendFixtureActivity.sentText)
            assertNull("Test must not start a ChatGPT conversation",PhoneChatRuntime.launch)
        } finally {
            automation.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK)
            shell("settings delete secure enabled_accessibility_services")
            shell("settings put secure accessibility_enabled 0")
        }
    }

    @Test fun preparingLongCodexDraftKeepsItsExactEditorAndSendsAtMostOnce() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val automation = instrumentation.getUiAutomation(UiAutomation.FLAG_DONT_SUPPRESS_ACCESSIBILITY_SERVICES)
        instrumentation.runOnMainSync {
            context.startActivity(Intent(context, ChatSendFixtureActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        }
        instrumentation.waitForIdleSync()
        val prompt = "Джессика готова к работе в чате. " +
            "Продолжаем план улучшения backend Jessica Core. ".repeat(65) +
            "JESSICA_REPLY_12345678-abcd-4abc-8abc-123456789012"
        try {
            var setText = false
            val ready = SystemClock.elapsedRealtime() + 10000
            while (!setText && SystemClock.elapsedRealtime() < ready) {
                automation.rootInActiveWindow?.let { root ->
                    val window = ChatWindowReader.inspect(root, context.resources.displayMetrics.density)
                    try {
                        window.composer?.let { editor ->
                            val args = android.os.Bundle().apply {
                                putCharSequence(
                                    AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, prompt)
                            }
                            setText = editor.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
                        }
                    } finally { window.release() }
                }
                if (!setText) SystemClock.sleep(100)
            }
            assertTrue("Long draft must be composed once", setText)
            var clicked = false
            val deadline = SystemClock.elapsedRealtime() + 10000
            while (!clicked && SystemClock.elapsedRealtime() < deadline) {
                automation.rootInActiveWindow?.let { root ->
                    val window = ChatWindowReader.inspect(root,
                        context.resources.displayMetrics.density, prompt)
                    try {
                        if (window.composer?.text?.toString() == prompt) {
                            val send = window.send
                            if (send != null) clicked = send.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                        }
                    } finally { window.release() }
                }
                if (!clicked) SystemClock.sleep(100)
            }
            assertTrue("Long exact draft must retain a clickable send control", clicked)
            val accepted = SystemClock.elapsedRealtime() + 3000
            while (ChatSendFixtureActivity.clicks.get() == 0 && SystemClock.elapsedRealtime() < accepted)
                SystemClock.sleep(50)
            assertEquals("Never send the prepared draft twice", 1, ChatSendFixtureActivity.clicks.get())
            assertEquals(prompt, ChatSendFixtureActivity.sentText)
        } finally {
            automation.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK)
        }
    }

}

package com.jessica.core.modules.codex.chat

import android.view.accessibility.AccessibilityNodeInfo
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class CodexSafeStopTest {
    private val id = "12345678-abcd-4abc-8abc-123456789012"

    @Test fun scrollingRemainsDisabledEvenIfChatGptAdvertisesScrollActions() {
        val root = AccessibilityNodeInfo.obtain()
        root.packageName = "com.openai.chatgpt"
        root.addAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD)
        root.addAction(AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD)
        var gestureCalls = 0
        val follower = CodexScrollFollower(1f) { _, _ ->
            gestureCalls++
            true
        }
        val window = ChatWindow(root, listOf(root), null, null, false, "")
        assertFalse(follower.follow(window, 5_000L, id))
        assertFalse(follower.recoverStart(window, 10_000L, id, false, true))
        assertEquals("No gesture fallback may run, even if markers are hidden",0,gestureCalls)
        @Suppress("DEPRECATION")
        root.recycle()
    }

    @Test fun stopRequestedIsAnImmediateLatchedCondition() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val session = PhoneChatLaunch(ctx)
        assertFalse(session.stopRequested)
        session.requestStop()
        assertTrue(session.stopRequested)
        session.requestStop()
        assertTrue(session.stopRequested)
    }
}

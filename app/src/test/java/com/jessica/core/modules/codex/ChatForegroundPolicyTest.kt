package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.ChatForegroundPolicy
import org.junit.Assert.*
import org.junit.Test

class ChatForegroundPolicyTest {
    @Test fun chatGptReturnWinsOverStaleOtherAppWindowFlag() {
        assertTrue(ChatForegroundPolicy.permits("com.openai.chatgpt",true,true,true,false,false))
    }
    @Test fun notificationShadeSuspendsButCannotEndTheSession() {
        assertFalse(ChatForegroundPolicy.permits("com.android.systemui",true,true,false,true,false))
        assertTrue(ChatForegroundPolicy.permits("com.openai.chatgpt",true,true,false,false,false))
    }
    @Test fun neverReadsWhenAnotherAppOwnsTheForeground() {
        assertFalse(ChatForegroundPolicy.permits("com.example.mail",true,true,true,false,false))
        assertFalse(ChatForegroundPolicy.permits("com.android.systemui",true,true,false,false,false))
        assertFalse(ChatForegroundPolicy.permits("com.openai.chatgpt",false,false,false,false,false))
    }
    @Test fun keyboardFocusMayTemporarilyOwnTheActiveRoot() {
        assertTrue(ChatForegroundPolicy.permits("com.google.android.inputmethod.latin",true,true,false,false,true))
        assertFalse(ChatForegroundPolicy.permits("com.google.android.inputmethod.latin",true,true,true,false,true))
    }
    @Test fun nullActiveRootMayRecoverFromVisibleSelectedWindow() {
        assertTrue(ChatForegroundPolicy.permits(null,false,true,false,false,false))
    }
    @Test fun staleSystemUiActiveRootRecoversWhenChatGptIsFocused() {
        assertTrue(ChatForegroundPolicy.permits("com.android.systemui",true,true,false,
            false,false,chatFocused=true))
        assertFalse(ChatForegroundPolicy.permits("com.android.systemui",true,true,false,
            true,false,chatFocused=true))
        assertFalse(ChatForegroundPolicy.permits("com.android.systemui",true,true,true,
            false,false,chatFocused=true))
    }
}


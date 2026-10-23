package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.CodexScrollSafetyPolicy
import org.junit.Assert.assertFalse
import org.junit.Test

class CodexScrollSafetyPolicyTest {
    @Test fun noAutomaticScrollingMaySwitchSelectedChat() {
        assertFalse(CodexScrollSafetyPolicy.mayScrollAutomatically())
    }
}

package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.CodexGuardianPolicy
import org.junit.Assert.*
import org.junit.Test

class CodexGuardianPolicyTest {
    @Test fun slowResponseDoesNotCountAsFinished() {
        assertEquals(CodexGuardianPolicy.Alert.NONE,
            CodexGuardianPolicy.evaluate(true,"awaiting_markers",9000,false,true))
        assertEquals(CodexGuardianPolicy.Alert.WAITING_REPLY,
            CodexGuardianPolicy.evaluate(true,"reply_partial",36000,false,true))
    }
    @Test fun analysisAndWindowRecoveryAreDistinctFromSending() {
        assertEquals(CodexGuardianPolicy.Alert.INTERPRETING,
            CodexGuardianPolicy.evaluate(false,"interpret_retry",39000,false,true))
        assertEquals(CodexGuardianPolicy.Alert.WINDOW_HIDDEN,
            CodexGuardianPolicy.evaluate(true,"awaiting_markers",39000,true,true))
        assertEquals(CodexGuardianPolicy.Alert.PREPARING,
            CodexGuardianPolicy.evaluate(false,"editor_missing",39000,false,true))
        assertEquals(CodexGuardianPolicy.Alert.NONE,
            CodexGuardianPolicy.evaluate(false,"next_ready",39000,false,false))
    }
}

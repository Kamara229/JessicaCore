package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.CodexSendConfirmationPolicy
import org.junit.Assert.*
import org.junit.Test

class CodexSendConfirmationPolicyTest {
    @Test fun unacknowledgedClickCannotBeInterpretedAsFailureToSend() {
        assertEquals(CodexSendConfirmationPolicy.Decision.WAIT_FOR_EVIDENCE,
            CodexSendConfirmationPolicy.decide(true,false,false))
        assertEquals(CodexSendConfirmationPolicy.Decision.PROCESS_REPLY,
            CodexSendConfirmationPolicy.decide(true,false,true))
        assertEquals(CodexSendConfirmationPolicy.Decision.NONE,
            CodexSendConfirmationPolicy.decide(false,false,false))
    }
}

package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.CodexTurnWindowPolicy
import org.junit.Assert.assertEquals
import org.junit.Test

class CodexTurnWindowPolicyTest {
    @Test fun missingComposerMustNotBlockFirstAnswer() {
        assertEquals(CodexTurnWindowPolicy.Next.READ_REPLY,
            CodexTurnWindowPolicy.next(waiting = true, composerVisible = false))
        assertEquals(CodexTurnWindowPolicy.Next.READ_REPLY,
            CodexTurnWindowPolicy.next(waiting = true, composerVisible = true))
    }
    @Test fun sendingRequiresVisibleComposer() {
        assertEquals(CodexTurnWindowPolicy.Next.WAIT_COMPOSER,
            CodexTurnWindowPolicy.next(waiting = false, composerVisible = false))
        assertEquals(CodexTurnWindowPolicy.Next.PREPARE_SEND,
            CodexTurnWindowPolicy.next(waiting = false, composerVisible = true))
    }
}

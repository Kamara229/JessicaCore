package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.CodexDraftVerificationPolicy
import org.junit.Assert.*
import org.junit.Test

class CodexDraftVerificationPolicyTest {
    @Test fun harmlessAndroidNewlineNormalizationDoesNotKillCodex() {
        val expected="Продолжаем чат.\nОтвет JESSICA_REPLY_id."
        assertTrue(CodexDraftVerificationPolicy.matchesExpected(
            expected.replace("\n","\r\n"),expected))
        assertTrue(CodexDraftVerificationPolicy.matchesExpected(
            expected+"\n",expected))
        assertTrue(CodexDraftVerificationPolicy.matchesExpected(
            expected.replace("Продолжаем чат.","Продолжаем\u00a0чат."), expected))
    }
    @Test fun partialOrDifferentDraftNeverAuthorizesSend() {
        val expected="Продолжаем backend. JESSICA_REPLY_token"
        assertFalse(CodexDraftVerificationPolicy.matchesExpected(expected.take(20),expected))
        assertFalse(CodexDraftVerificationPolicy.matchesExpected("",expected))
        assertFalse(CodexDraftVerificationPolicy.matchesExpected("Мой черновик",expected))
    }
}

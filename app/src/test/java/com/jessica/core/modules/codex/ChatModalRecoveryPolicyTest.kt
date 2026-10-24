package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.ChatModalRecoveryPolicy
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ChatModalRecoveryPolicyTest {
    @Test fun recognizesVisibleChatShareLinkDialog() {
        val visible = "Поделиться ссылкой на чат?\nПри этом создаётся копия беседы,\nв которой другие могут общаться в чате."
        assertTrue(ChatModalRecoveryPolicy.isShareLinkPrompt(visible))
    }
    @Test fun neverTreatsGenericChatShareTextAsDialog() {
        assertFalse(ChatModalRecoveryPolicy.isShareLinkPrompt(
            "Прочитай инструкцию: Поделиться ссылкой на чат?"))
        assertFalse(ChatModalRecoveryPolicy.isShareLinkPrompt(
            "Поделиться ссылкой на чат?\nЯ спросил о тексте диалога."))
        assertFalse(ChatModalRecoveryPolicy.isShareLinkPrompt(
            "При этом создаётся копия, в которой другие могут общаться."))
    }
    @Test fun englishSharePromptAlsoRecognizedOnlyWithContext() {
        assertTrue(ChatModalRecoveryPolicy.isShareLinkPrompt(
            "Share a link to this chat?\nAnyone with the link can view a copy of your chat."))
        assertFalse(ChatModalRecoveryPolicy.isShareLinkPrompt(
            "Share a link to this chat?\nDiscussed earlier in the conversation."))
    }
}

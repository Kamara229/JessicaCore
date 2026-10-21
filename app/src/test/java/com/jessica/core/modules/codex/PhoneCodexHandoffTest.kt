package com.jessica.core.modules.codex

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PhoneCodexHandoffTest {
    private val privateUrl = "https://chatgpt.com/c/12345678-abcd-4abc-8abc-123456789012"

    @Test fun onlyThreePreApprovedGoals() {
        for(goal in listOf("language","context","foundations")){
            assertTrue(PhoneCodexHandoff.supportedGoal(goal))
            val prompt=PhoneCodexHandoff.buildPrompt(goal)
            assertTrue(prompt.contains("Jessica Core"))
            assertTrue(prompt.contains("Kamara229/JessicaCore"))
        }
        assertFalse(PhoneCodexHandoff.supportedGoal("delete-database"))
    }

    @Test fun recognizesOnlyPrivateChatUriAndNotPublicSharedConversations() {
        assertEquals(privateUrl, PhoneCodexHandoff.normalizePrivateChatLink(privateUrl))
        assertEquals(privateUrl, PhoneCodexHandoff.normalizePrivateChatLink(privateUrl.replace("chatgpt.com", "www.chatgpt.com")))
        for(url in listOf(
            "",
            "http://chatgpt.com/c/12345678-abcd-4abc-8abc-123456789012",
            "https://evilchatgpt.com/c/12345678-abcd-4abc-8abc-123456789012",
            "https://chatgpt.com/share/12345678-abcd-4abc-8abc-123456789012",
            "https://chatgpt.com/c/12345678-abcd-4abc-8abc-123456789012?token=test",
            "https://chatgpt.com:444/c/12345678-abcd-4abc-8abc-123456789012",
            "https://example.com/c/12345678-abcd-4abc-8abc-123456789012",
            "https://chatgpt.com/c/%2Fetc%2Fpasswd",
            "https://user@chatgpt.com/c/12345678-abcd-4abc-8abc-123456789012"
        )){
            assertEquals("Accepted unsafe link: " + url, null, PhoneCodexHandoff.normalizePrivateChatLink(url))
        }
    }

    @Test fun promptDoesNotClaimToReadOrControlChatGptApp() {
        val prompt=PhoneCodexHandoff.buildPrompt("context")
        assertTrue(prompt.contains("проверь") || prompt.contains("Проверь"))
        assertTrue(prompt.contains("не") || prompt.contains("Не"))
    }
}

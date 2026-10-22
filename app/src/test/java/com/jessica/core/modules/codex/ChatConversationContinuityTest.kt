package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.ChatConversationContinuity
import org.junit.Assert.*
import org.junit.Test

class ChatConversationContinuityTest {
    private val id = "12345678-abcd-4abc-8abc-123456789012"
    private val other = "23456789-abcd-4abc-8abc-123456789012"
    private val tail = "Это подтверждённый последний ответ ChatGPT в выбранной беседе. " +
        "Он достаточно длинный для отличия от любого обычного короткого сообщения."

    @Test fun standaloneEndAllowsFollowupWhenOpeningMarkerScrolledAway() {
        assertTrue(ChatConversationContinuity.permits("Ответ\n**JESSICA_END_$id**",id,id,""))
    }
    @Test fun answerTailAllowsFollowupWhenBothMarkersScrolledAway() {
        assertTrue(ChatConversationContinuity.permits("Содержимое\n$tail",id,id,tail))
    }
    @Test fun quotedInstructionMarkersNeverAuthorizeAnotherConversation() {
        val quoted = "Начни строкой JESSICA_REPLY_$id; закончи JESSICA_END_$id."
        assertFalse(ChatConversationContinuity.permits(quoted,id,id,""))
        assertFalse(ChatConversationContinuity.permits(quoted,id,other,tail))
        assertFalse(ChatConversationContinuity.permits("JESSICA_END_$id",id,other,tail))
    }
    @Test fun genericSmallReplyIsNotSufficientEvidenceForFollowup() {
        assertFalse(ChatConversationContinuity.permits("Готово",id,id,"Готово"))
        assertFalse(ChatConversationContinuity.permits("Готово",id,id,tail))
    }
}

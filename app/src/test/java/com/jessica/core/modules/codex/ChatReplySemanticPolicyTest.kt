package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.ChatReplySemanticPolicy
import org.junit.Assert.*
import org.junit.Test

class ChatReplySemanticPolicyTest {
    private val id = "12345678-abcd-4abc-8abc-123456789012"
    private val other = "23456789-abcd-4abc-8abc-123456789012"
    @Test fun onlyCompleteCurrentReplyCanBeReadFromOffscreenSemanticNode() {
        val prompt = "Начни ответ отдельной строкой JESSICA_REPLY_$id. " +
            "Закончи отдельной строкой JESSICA_END_$id."
        assertEquals(id,ChatReplySemanticPolicy.expectedReplyId(prompt))
        val node = "JESSICA_REPLY_$id\nКороткий следующий шаг.\nJESSICA_END_$id"
        assertEquals(node,ChatReplySemanticPolicy.completedNodeText(node,id))
        assertNull(ChatReplySemanticPolicy.completedNodeText(node,other))
        assertNull(ChatReplySemanticPolicy.completedNodeText(node,null))
    }
    @Test fun neitherOwnersInstructionNorStreamingFragmentMayBecomeAnAnswer() {
        val quoted = "Начни ответ отдельной строкой JESSICA_REPLY_$id. " +
            "Закончи отдельной строкой JESSICA_END_$id."
        assertNull(ChatReplySemanticPolicy.completedNodeText(quoted,id))
        assertNull(ChatReplySemanticPolicy.completedNodeText(
            "JESSICA_REPLY_$id\nДлинный ответ пока продолжается",id))
    }
}

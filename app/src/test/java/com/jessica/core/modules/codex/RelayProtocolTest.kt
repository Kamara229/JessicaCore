package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.relay.RelayProtocol
import com.jessica.core.modules.codex.chat.ChatRunState
import org.junit.Assert.*
import org.junit.Test

class RelayProtocolTest {
    private val id = "12345678-abcd-4abc-8abc-123456789012"
    @Test fun readinessIsTheFirstSentenceAndHandoffRestrictsWorkToBackend() {
        assertTrue(RelayProtocol.introduction.startsWith("Джессика готова к работе в чате."))
        assertTrue(RelayProtocol.introduction.contains("не редактирую репозиторий"))
        assertTrue(RelayProtocol.introduction.contains("тесты через свой обычный чат"))
        assertNull(RelayProtocol.reply(RelayProtocol.wrap(RelayProtocol.introduction,id),id))
    }
    @Test fun completeNaturalLanguageReplyNeedsFreshMarkersButNotJson() {
        val body = "Backend обновлён.\nВыполни в своём чате: Сколько будет 2 + 2?"
        assertEquals(body,RelayProtocol.reply("JESSICA_REPLY_$id\n$body\nJESSICA_END_$id",id))
        assertNull(RelayProtocol.reply("JESSICA_REPLY_$id\n$body",id))
        assertNull(RelayProtocol.reply("JESSICA_REPLY_old\n$body\nJESSICA_END_old",id))
    }
    @Test fun aDaySessionHasMultipleTurnsButNeverRetriesAnUncertainSend() {
        val state = ChatRunState(0,maxTurns=60,durationMs=8*60*60*1000L)
        assertTrue(state.beginSend(1))
        assertFalse(state.beginSend(2000))
        assertNull(state.acceptReply("Готово",false,10000))
        assertEquals("Готово",state.acceptReply("Готово",false,17000))
        assertFalse(state.expired(60*60*1000L))
        assertTrue(state.expired(8*60*60*1000L))
        state.stop()
        assertFalse(state.beginSend(18000))
    }
    @Test fun acceptsReplyWithMarkdownMarkersFromChatGPTRenderedMessage() {
        val body = "Проверка исправлена. Следующий тест: Сколько будет 2 + 2?"
        val screen = "**JESSICA_REPLY_${id}**\n${body}\n**JESSICA_END_${id}**"
        assertEquals(body, RelayProtocol.reply(screen,id))
    }
    @Test fun acceptsAccessibilitySplittingMarkerBeforeUuid() {
        val body = "Следующий шаг — проверить тест Джессики."
        val screen = "JESSICA_REPLY_\n${id}\n${body}\nJESSICA_END_\n${id}"
        assertEquals(body,RelayProtocol.reply(screen,id))
    }
    @Test fun ignoresUserPromptMentioningMarkersInAnInstruction() {
        val request = "Начни ответ отдельной строкой JESSICA_REPLY_${id}; закончи отдельной строкой JESSICA_END_${id}."
        assertNull(RelayProtocol.reply(request,id))
        assertNull(RelayProtocol.reply("JESSICA_REPLY_${id}\nНезавершённый ответ",id))
    }
    @Test fun choosesLatestCompleteReplyAndPreservesSourceBlocks() {
        val first = "JESSICA_REPLY_${id}\nСтарый ответ\nJESSICA_END_${id}"
        val quoted = "JESSICA_REPLY_other\nДругой ответ\nJESSICA_END_other"
        val second = "JESSICA_REPLY_${id}\n" +
            "Код:\n" + "```python\nprint(42)\n```" + "\nJESSICA_END_${id}"
        assertEquals("Код:\n```python\nprint(42)\n```",RelayProtocol.reply(first+"\n"+quoted+"\n"+second,id))
    }

}

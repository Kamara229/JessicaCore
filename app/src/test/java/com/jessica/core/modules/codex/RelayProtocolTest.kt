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
}

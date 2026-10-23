package com.jessica.core.modules.codex.chat

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.jessica.core.modules.codex.relay.RelayConversation
import com.jessica.core.modules.codex.relay.RelayInterpretationUnavailable
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class RelayInterpretationRetryTest {
    @Test fun reattemptingInterpretationPreservesSingleIncomingRecord() = runBlocking {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        var attempts=0
        val relay=RelayConversation(ctx, interpret={ req ->
            attempts++
            assertEquals(1,req.getJSONArray("history").length())
            if (attempts==1) throw java.io.IOException("temporary")
            JSONObject().put("action","reply").put("reply","Продолжаю.")
        })
        relay.recordOutgoing("Начало")
        try {
            relay.respond("Первый ответ",{})
            fail("Expected interpretation failure")
        } catch (_: RelayInterpretationUnavailable) {}
        val result=relay.respond("Первый ответ",true,{})
        assertEquals("Продолжаю.",result.text)
        assertFalse(result.endSession)
        assertEquals(2,attempts)
    }
}

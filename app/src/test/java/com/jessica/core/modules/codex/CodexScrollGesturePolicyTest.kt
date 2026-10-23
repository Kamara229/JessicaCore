package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.*
import org.junit.Assert.*
import org.junit.Test

class CodexScrollGesturePolicyTest {
    private val screen=UiBox(0,0,400,900)
    private val editor=UiBox(30,780,390,890)
    @Test fun fallbackSwipesInsideConversationNotOnComposer() {
        val back=CodexScrollGesturePolicy.calculate(screen,editor,true,1f)!!
        val forward=CodexScrollGesturePolicy.calculate(screen,editor,false,1f)!!
        assertTrue(back.startY<back.endY)
        assertTrue(forward.startY>forward.endY)
        assertTrue(back.endY<editor.top)
        assertTrue(forward.startY<editor.top)
        assertEquals(back.x,forward.x)
    }
    @Test fun neverSwipesWhenConversationViewportIsTooSmall() {
        assertNull(CodexScrollGesturePolicy.calculate(UiBox(0,0,400,280),null,true,1f))
    }
}

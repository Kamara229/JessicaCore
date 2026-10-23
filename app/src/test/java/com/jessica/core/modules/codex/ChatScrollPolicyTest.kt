package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.CodexScrollCandidate
import com.jessica.core.modules.codex.chat.CodexScrollPolicy
import com.jessica.core.modules.codex.chat.UiBox
import org.junit.Assert.*
import org.junit.Test

class ChatScrollPolicyTest {
    private val frame = UiBox(0, 0, 420, 900)
    private val editor = UiBox(25, 770, 395, 890)
    private fun c(index:Int, box:UiBox, down:Boolean=true,
                  forward:Boolean=true, enabled:Boolean=true,visible:Boolean=true) =
        CodexScrollCandidate(index,box,visible,enabled,down,forward)

    @Test fun scrollsOnlyTheLargeCentralConversationArea() {
        val items = listOf(
            c(1,UiBox(0,0,420,900)), // outside scroll container
            c(2,UiBox(0,80,420,760)), // chat viewport
            c(3,UiBox(5,800,415,900)), // composer area
            c(4,UiBox(360,90,412,700)) // narrow tool/sidebar
        )
        assertEquals(2, CodexScrollPolicy.choose(items,frame,editor,1f)?.index)
    }

    @Test fun doesNotScrollWhenOnlyEditorOrSidebarIsScrollable() {
        assertNull(CodexScrollPolicy.choose(listOf(
            c(1,UiBox(30,780,390,890)),
            c(2,UiBox(320,100,410,600))
        ),frame,editor,1f))
    }

    @Test fun ignoresInvisibleOrDisabledContainersAndSupportsForwardOnly() {
        val hidden = c(1,UiBox(0,80,420,760),visible=false)
        val disabled = c(2,UiBox(0,80,420,760),enabled=false)
        val body = c(3,UiBox(0,80,420,760),down=false)
        assertEquals(3,CodexScrollPolicy.choose(listOf(hidden,disabled,body),
            frame,editor,1f)?.index)
    }

    @Test fun scrollingDoesNotDependOnEditorRemainingVisible() {
        val scroller = c(5,UiBox(0,50,420,850))
        assertEquals(5,CodexScrollPolicy.choose(listOf(scroller),
            frame,null,1f)?.index)
        assertNull(CodexScrollPolicy.choose(emptyList(),frame,null,1f))
    }

    @Test fun rejectsInvalidFrameOrDensity() {
        val scroller = c(1,UiBox(0,50,420,850))
        assertNull(CodexScrollPolicy.choose(listOf(scroller),UiBox(0,0,0,0),editor,1f))
        assertNull(CodexScrollPolicy.choose(listOf(scroller),frame,editor,0f))
    }
}

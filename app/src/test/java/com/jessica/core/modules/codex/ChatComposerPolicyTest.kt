package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.ChatComposerPolicy
import com.jessica.core.modules.codex.chat.ComposerCandidate
import com.jessica.core.modules.codex.chat.UiBox
import org.junit.Assert.*
import org.junit.Test

class ChatComposerPolicyTest {
    private val window = UiBox(0, 0, 420, 900)
    private fun candidate(index: Int, left: Int, top: Int, right: Int, bottom: Int,
                          focused: Boolean = false, editable: Boolean = true,
                          setText: Boolean = true) =
        ComposerCandidate(index, UiBox(left, top, right, bottom), focused, editable, setText)

    @Test fun actualChatComposerIsChosenDespiteAdditionalSearchField() {
        val nodes = listOf(
            candidate(1, 30, 55, 400, 100, focused = true), // search at top
            candidate(5, 25, 655, 395, 760)
        )
        assertEquals(5, ChatComposerPolicy.choose(nodes,window,1f))
    }

    @Test fun virtualParentAndNativeEditableChildAreOneField() {
        val nodes = listOf(
            candidate(3,25,645,395,770,editable=true,setText=false),
            candidate(4,30,653,391,763,focused=true,editable=true,setText=true)
        )
        assertEquals(4,ChatComposerPolicy.choose(nodes,window,1f))
    }

    @Test fun multilineEditorRemainsAvailableWhenKeyboardMovesComposerUpward() {
        val nodes = listOf(
            candidate(1,30,65,390,112),
            candidate(6,25,350,395,525,focused=true)
        )
        assertEquals(6,ChatComposerPolicy.choose(nodes,window,1f))
    }

    @Test fun twoDifferentPlausibleEditorsAtSameBottomRemainAmbiguous() {
        val nodes = listOf(
            candidate(5,5,670,202,765),
            candidate(9,215,670,410,765)
        )
        assertNull(ChatComposerPolicy.choose(nodes,window,1f))
    }

    @Test fun nonTextButtonsAndInvisibleShortControlsCannotReceiveMessage() {
        assertNull(ChatComposerPolicy.choose(listOf(
            candidate(2,310,700,367,750,editable=false,setText=false),
            candidate(4,10,500,12,530)
        ),window,1f))
    }

    @Test fun composerWithSetTextButWithoutEditableFlagIsUsable() {
        assertEquals(9,ChatComposerPolicy.choose(listOf(
            candidate(9,25,670,395,745,editable=false,setText=true)
        ),window,1f))
    }
}

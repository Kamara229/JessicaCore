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
                          setText: Boolean = true, matchesPrepared: Boolean = false) =
        ComposerCandidate(index, UiBox(left, top, right, bottom), focused, editable, setText, matchesPrepared)

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
    @Test fun longPreparedDraftStillFindsExactSingleEditorAfterChatGptBoundsShift() {
        // Reproduces field-count=1/composer=null after ChatGPT grows a long draft.
        val weird = candidate(7, -600, -1200, -100, -600, matchesPrepared = true)
        assertEquals(7, ChatComposerPolicy.choose(listOf(weird), window, 1f))
        assertNull(ChatComposerPolicy.choose(listOf(weird.copy(matchesPrepared = false)), window, 1f))
    }

    @Test fun existingPreparedDraftWinsOverSearchAndOtherEmptyEditors() {
        val search = candidate(2, 30, 50, 390, 90, focused = true)
        val draft = candidate(10, 20, 600, 405, 900, matchesPrepared = true)
        assertEquals(10, ChatComposerPolicy.choose(listOf(search, draft), window, 1f))
    }

    @Test fun duplicatePreparedDraftInDistinctEditorsMustNotBeSent() {
        val first = candidate(8, 30, 600, 200, 860, matchesPrepared = true)
        val second = candidate(9, 215, 600, 408, 860, matchesPrepared = true)
        assertNull(ChatComposerPolicy.choose(listOf(first, second), window, 1f))
    }

    @Test fun preparedFlagNeverMakesNonEditableNodeAnEditor() {
        val label = candidate(9, 0, 0, 1, 1, editable = false, setText = false, matchesPrepared = true)
        assertNull(ChatComposerPolicy.choose(listOf(label), window, 1f))
    }

    @Test fun twoNestedEditableNodesWithSamePreparedDraftResolveToOneComposer() {
        val nodes = listOf(
            candidate(3,25,645,395,770,editable=true,setText=false,matchesPrepared=true),
            candidate(4,30,653,391,763,focused=true,editable=true,setText=true,matchesPrepared=true)
        )
        assertEquals(4, ChatComposerPolicy.choose(nodes,window,1f))
    }

    @Test fun samePreparedDraftInTwoDistinctFieldsMustRemainUnsafe() {
        val nodes = listOf(
            candidate(3,25,645,189,770,matchesPrepared=true),
            candidate(4,215,645,391,770,matchesPrepared=true)
        )
        assertNull(ChatComposerPolicy.choose(nodes,window,1f))
    }

    @Test fun exactPreparedDraftSurvivesZeroSizeChatGptRootBounds() {
        val invalidRoot = UiBox(0, 0, 0, 0)
        val exact = candidate(6, 30, 560, 390, 760, matchesPrepared = true)
        assertEquals(6, ChatComposerPolicy.choose(listOf(exact), invalidRoot, 1f))
    }

    @Test fun existingDraftRecoversSoleEditorWhenAndroidReportsTinyBounds() {
        val tiny = candidate(7, 100, 640, 120, 665, focused = false, matchesPrepared = false)
        assertNull(ChatComposerPolicy.choose(listOf(tiny), window, 1f))
        assertEquals(7, ChatComposerPolicy.choose(
            listOf(tiny), window, 1f, recoveringPreparedDraft = true))
    }

    @Test fun draftRecoveryCannotChooseOneOfSeveralPossibleFields() {
        val tiny = candidate(7, 100, 640, 120, 665)
        val search = candidate(9, 10, 100, 40, 115)
        assertNull(ChatComposerPolicy.choose(
            listOf(tiny, search), window, 1f, recoveringPreparedDraft = true))
    }

    @Test fun zeroSizeRootCannotResolveWrongOrMissingDraft() {
        val invalidRoot = UiBox(0, 0, 0, 0)
        val wrong = candidate(6, 30, 560, 390, 760)
        assertNull(ChatComposerPolicy.choose(listOf(wrong), invalidRoot, 1f))
        assertNull(ChatComposerPolicy.choose(emptyList(), invalidRoot, 1f))
    }

}

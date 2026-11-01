package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.CodexIdlePulsePolicy
import com.jessica.core.modules.codex.chat.CodexScrollSafetyPolicy
import com.jessica.core.modules.codex.chat.UiBox
import org.junit.Assert.*
import org.junit.Test

class CodexIdlePulsePolicyTest {
    private fun due(
        now: Long = 181_000L,
        last: Long = 0L,
        optedIn: Boolean = true,
        selected: Boolean = true,
        waiting: Boolean = true,
        generating: Boolean = true,
        interactive: Boolean = true,
        locked: Boolean = false,
        anchored: Boolean = true
    ) = CodexIdlePulsePolicy.due(
        optedIn=optedIn,selected=selected,waiting=waiting,generating=generating,
        screenInteractive=interactive,keyguardLocked=locked,
        eligibleTranscript=anchored,sendAt=1_000L,lastAttemptAt=last,now=now
    )

    @Test fun firstGestureRequires180SecondsAndCurrentGeneratingReply() {
        assertFalse(due(now=180_999L))
        assertTrue(due(now=181_000L))
        assertFalse(due(now=181_001L,last=180_999L))
        assertTrue(due(now=361_000L,last=181_000L))
        assertFalse(due(generating=false))
        assertFalse(due(waiting=false))
    }

    @Test fun neverTouchUnconfirmedLockedHiddenOrOptedOutConversation() {
        assertFalse(due(optedIn=false))
        assertFalse(due(selected=false))
        assertFalse(due(interactive=false))
        assertFalse(due(locked=true))
        assertFalse(due(anchored=false))
        assertFalse(CodexScrollSafetyPolicy.mayScrollAutomatically())
        assertFalse(CodexScrollSafetyPolicy.mayScrollAutomatically(true,false))
        assertFalse(CodexScrollSafetyPolicy.mayScrollAutomatically(false,true))
        assertTrue(CodexScrollSafetyPolicy.mayScrollAutomatically(true,true))
    }

    @Test fun strokeCoordinatesKeepBothTinySwipesOutOfComposerAndToolbars() {
        val transcript=UiBox(0,100,400,700)
        val composer=UiBox(0,770,400,850)
        val pair=CodexIdlePulsePolicy.coordinates(transcript,composer,1f)
        assertNotNull(pair)
        val points=pair!!
        assertEquals(points.upStartY,points.downEndY,0.001f)
        assertEquals(points.upEndY,points.downStartY,0.001f)
        assertTrue(points.upStartY < composer.top)
        assertEquals(32f,points.upStartY-points.upEndY,0.001f)
        assertNull(CodexIdlePulsePolicy.coordinates(UiBox(0,0,400,780),composer,1f))
        assertNull(CodexIdlePulsePolicy.coordinates(UiBox(0,100,400,280),composer,1f))
        assertNull(CodexIdlePulsePolicy.coordinates(transcript,composer,0f))
    }
}

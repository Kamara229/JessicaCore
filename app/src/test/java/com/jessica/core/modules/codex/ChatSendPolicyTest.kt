package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.*
import org.junit.Assert.*
import org.junit.Test

class ChatSendPolicyTest {
    private fun node(parent: Int, label: String = "", clickable: Boolean = false, box: UiBox = UiBox(310,700,365,755),
                     editor: Boolean = false, enabled: Boolean = true, id: String = "") =
        ChatControl(parent,listOf(label),id,clickable,enabled,true,editor,box)
    private val root = node(-1,box=UiBox(0,0,400,850))
    private val editor = node(0,"Отправить сообщение",true,UiBox(20,500,380,690),editor=true)
    @Test fun labelOnChildAndClickableParentResolveToOneTarget() {
        val nodes = listOf(root,editor,node(0,clickable=true),node(2,"Отправить сообщение"))
        assertEquals(setOf(2),ChatSendPolicy.targets(nodes,1,1f))
    }
    @Test fun nestedClickableSemanticsAreNotTwoDifferentButtons() {
        val nodes = listOf(root,editor,node(0,"Send",true),node(2,"Send message",true))
        assertEquals(setOf(2),ChatSendPolicy.targets(nodes,1,1f))
    }
    @Test fun resourceIdsAndRoleSuffixesAreRecognizedButIndependentButtonsStayAmbiguous() {
        val nodes = listOf(root,editor,node(0,clickable=true,id="com.openai.chatgpt:id/send_button"),node(0,"Отправить сообщение, кнопка",true))
        assertEquals(2,ChatSendPolicy.targets(nodes,1,1f).size)
        assertNull(ChatSendPolicy.targets(nodes,1,1f).singleOrNull())
    }
    @Test fun draftsDisabledButtonsAndDistantConversationTextCannotBecomeSendTargets() {
        val nodes = listOf(root,editor,node(0,"Send",true,enabled=false),
            node(0,"Send",true,UiBox(310,50,365,105)),node(1,"Send",true,editor=true))
        assertTrue(ChatSendPolicy.targets(nodes,1,1f).isEmpty())
    }
    @Test fun chosenPointFollowsComposerHeightButIsInvalidatedByWidthChange() {
        val frame = UiBox(0,0,400,850)
        val box = UiBox(20,500,380,690)
        val target = requireNotNull(CalibratedSendTarget.select(frame,box,340f,730f,1f))
        assertEquals(340f to 430f,target.point(frame,UiBox(20,300,380,390),1f))
        assertNull(target.point(UiBox(0,0,800,850),box,1f))
        assertNull(target.point(frame,UiBox(40,500,350,690),1f))
        assertNull(CalibratedSendTarget.select(frame,box,30f,100f,1f))
        assertNull(target.point(UiBox(0,0,400,700),box,1f))
    }

    @Test fun duplicatedAccessibilitySiblingsAtSameSendButtonResolveAsOneVisualControl() {
        val nodes = listOf(root,editor,
            node(0,"Send",true,UiBox(315,708,369,757)),
            node(0,"Send message",true,UiBox(315,708,369,757)))
        assertEquals(2,ChatSendPolicy.targets(nodes,1,1f).size)
        assertNotNull(ChatSendPolicy.uniqueSendTarget(nodes,1,1f))
    }
    @Test fun distinctSendButtonsAreNeverGuessed() {
        val nodes = listOf(root,editor,
            node(0,"Send",true,UiBox(310,700,360,748)),
            node(0,"Send message",true,UiBox(260,700,305,748)))
        assertNull(ChatSendPolicy.uniqueSendTarget(nodes,1,1f))
    }
}

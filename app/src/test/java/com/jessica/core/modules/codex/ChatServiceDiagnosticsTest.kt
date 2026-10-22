package com.jessica.core.modules.codex

import com.jessica.core.modules.codex.chat.ChatServiceState
import com.jessica.core.modules.codex.chat.classifyChatService
import org.junit.Assert.assertEquals
import org.junit.Test

/** Explicitly distinguish OS permission from AccessibilityService binding. */
class ChatServiceDiagnosticsTest {
    @Test fun enabledPermissionWithNoBoundServiceMustNeverBeCalledDisabled() {
        assertEquals(
            ChatServiceState.ENABLED_NOT_CONNECTED,
            classifyChatService(registered = true, enabled = true, connected = false)
        )
    }

    @Test fun enabledStatusWinsWhenAndroidServiceListingIsUnavailable() {
        assertEquals(
            ChatServiceState.ENABLED_NOT_CONNECTED,
            classifyChatService(registered = false, enabled = true, connected = false)
        )
    }

    @Test fun connectedServiceUnlocksPhoneProbe() {
        assertEquals(
            ChatServiceState.CONNECTED,
            classifyChatService(registered = true, enabled = true, connected = true)
        )
    }

    @Test fun disabledAndMissingServicesHaveDistinctDiagnostics() {
        assertEquals(
            ChatServiceState.DISABLED,
            classifyChatService(registered = true, enabled = false, connected = false)
        )
        assertEquals(
            ChatServiceState.NOT_REGISTERED,
            classifyChatService(registered = false, enabled = false, connected = false)
        )
    }
}

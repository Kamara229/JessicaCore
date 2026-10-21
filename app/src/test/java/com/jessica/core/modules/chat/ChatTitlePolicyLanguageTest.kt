package com.jessica.core.modules.chat

import org.junit.Assert.assertEquals
import org.junit.Test

class ChatTitlePolicyLanguageTest {
    @Test fun englishExplicitNamesKeepProperNouns() {
        assertEquals("Project Aurora", ChatTitlePolicy.explicit("Name this chat \"Project Aurora\""))
        assertEquals("Jessica Core", ChatTitlePolicy.explicit("Chat title: Jessica Core"))
    }

    @Test fun russianExplicitNamesStillWork() {
        assertEquals("Проект Алмаз", ChatTitlePolicy.explicit("Назови этот чат «Проект Алмаз»"))
    }
}

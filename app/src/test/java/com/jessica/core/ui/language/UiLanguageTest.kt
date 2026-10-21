package com.jessica.core.ui.language

import org.junit.Assert.assertEquals
import org.junit.Test

class UiLanguageTest {
    @Test fun russianIsDefaultAndUnknownCodesAreSafe() {
        assertEquals(AppLanguage.RUSSIAN, AppLanguage.fromCode(null))
        assertEquals(AppLanguage.RUSSIAN, AppLanguage.fromCode("de"))
        assertEquals(AppLanguage.RUSSIAN, AppLanguage.fromCode("ru"))
    }

    @Test fun languageChoiceDoesNotTranslateUserNames() {
        assertEquals(AppLanguage.ENGLISH, AppLanguage.fromCode("en"))
        assertEquals("Chat history", localeText(AppLanguage.ENGLISH, "История диалогов", "Chat history"))
        assertEquals("История диалогов", localeText(AppLanguage.RUSSIAN, "История диалогов", "Chat history"))
        assertEquals("Jessica Core", localeText(AppLanguage.ENGLISH, "Jessica Core", "Jessica Core"))
    }
}

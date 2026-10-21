package com.jessica.core.ui.language

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

enum class AppLanguage(val code: String, val label: String) {
    RUSSIAN("ru", "Русский"), ENGLISH("en", "English");
    companion object {
        fun fromCode(value: String?): AppLanguage =
            entries.firstOrNull { it.code == value } ?: RUSSIAN
    }
}

/** Language preference is device-local and survives updates and app restarts. */
class AppLanguageStore(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences("jessica_ui", Context.MODE_PRIVATE)
    private val mutable = MutableStateFlow(AppLanguage.fromCode(prefs.getString("language", "ru")))
    val language = mutable.asStateFlow()

    fun current(): AppLanguage = AppLanguage.fromCode(prefs.getString("language", "ru"))

    fun set(language: AppLanguage): Boolean {
        if (!prefs.edit().putString("language", language.code).commit()) return false
        mutable.value = language
        return true
    }
}

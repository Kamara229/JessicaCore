package com.jessica.core.modules.codex

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.net.Uri

/**
 * Local, explicit user-controlled handoff. No Render, tokens, third-party
 * screen scraping, ChatGPT message extraction, or AccessibilityService.
 * ChatGPT doesn't expose a supported API to join an existing private chat.
 */
object PhoneCodexHandoff {
    private val goals = mapOf(
        "language" to "Проверь корректность языковых ответов Jessica Core: русский/английский интерфейс, языки исходных заданий, имена, кодовые блоки и ссылки.",
        "context" to "Проверь непрерывность контекста Jessica Core: отсылки к предыдущему ответу, корректное определение текущей задачи и исключение чужих диалогов.",
        "foundations" to "Проверь Foundations Jessica Core: обработку языков программирования, данных, источников, безопасность анализа кода и качество тестов."
    )

    fun supportedGoal(goal: String) = goals.containsKey(goal)

    fun buildPrompt(goal: String): String {
        val objective = goals[goal] ?: throw IllegalArgumentException("unknown-goal")
        return listOf(
            "Продолжи существующую разработку проекта Jessica Core (Kamara229/JessicaCore). Не начинай проект заново.",
            "Задание из локального мобильного Codex: $objective",
            "Проверь действующие исходники и предложи конкретные небольшие модульные исправления под фасадом.",
            "Не выдумывай результаты тестов, коммиты или развёртывания. Не изменяй продакшен без проверки.",
            "Сохраняй приватные чаты, ключи и файлы вне публичных GitHub Issues и PR.",
            "Автономное Experience/Learning не требует ручного одобрения навыков, но положительный опыт допустим только после доказательной проверки.",
            "Если инструменты GitHub доступны в этом чате, внеси исправления и проверь CI. Иначе объясни, какой именно доступ отсутствует."
        ).joinToString("\n\n")
    }

    /**
     * Private chat links vary across ChatGPT versions. Only allow the
     * account's canonical /c/<UUID> URL; never treat shared public links
     * as writable private conversations.
     */
    fun normalizePrivateChatLink(raw: String): String? {
        val trimmed = raw.trim()
        if (trimmed.isEmpty()) return null
        val uri = try { java.net.URI(trimmed) } catch (_: Exception) { return null }
        if (uri.scheme != "https" || uri.host !in setOf("chatgpt.com","www.chatgpt.com")) return null
        if (uri.port != -1 || uri.rawUserInfo != null || uri.rawQuery != null || uri.rawFragment != null) return null
        val match = Regex("^/c/([a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12})$").matchEntire(uri.rawPath.orEmpty())
            ?: return null
        return "https://chatgpt.com/c/" + match.groupValues[1]
    }

    /** Open ChatGPT on this device; Android decides how the private URL opens. */
    fun openChatGPT(context: Context, privateChatLink: String?): Boolean {
        val appIntent = context.packageManager.getLaunchIntentForPackage("com.openai.chatgpt")
        val link = privateChatLink?.let(::normalizePrivateChatLink)
        val intent = when {
            link != null -> Intent(Intent.ACTION_VIEW, Uri.parse(link)).apply {
                addCategory(Intent.CATEGORY_BROWSABLE)
                if (appIntent != null) setPackage("com.openai.chatgpt")
            }
            appIntent != null -> appIntent
            else -> Intent(Intent.ACTION_VIEW, Uri.parse("https://chatgpt.com/"))
        }
        return try {
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            context.startActivity(intent)
            true
        } catch (_: ActivityNotFoundException) {
            // Some ChatGPT releases do not handle account-private deep links.
            if (appIntent == null) return false
            return try {
                appIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(appIntent)
                true
            } catch (_: Exception) { false }
        } catch (_: SecurityException) { false }
    }
}

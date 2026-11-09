package com.jessica.core.modules.codex.relay

/**
 * Device-only deterministic continuation for strictly passive CALIBRI
 * progress. Works even when the remote interpretation service is unavailable.
 * Never sends a test, never marks a result verified, and does not end Codex.
 *
 * Only the owner's already-selected ChatGPT conversation is used by the
 * existing selected-chat/marker/composer guards downstream.
 */
object CodexPassiveLocalHandoff {
    fun response(message: String): RelayReply? {
        if (message.isBlank() || message.length > 1400) return null
        val lines = message.trim().lines().map { it.trim() }.filter { it.isNotEmpty() }
        val opening = Regex("""^JESSICA_REPLY_([0-9a-fA-F-]{36})$""").matchEntire(lines.firstOrNull().orEmpty())
        val body = if (opening != null) {
            if (lines.size < 3 ||
                lines.last() != "JESSICA_END_${opening.groupValues[1]}") return null
            lines.subList(1, lines.size - 1).joinToString(" ")
        } else {
            if (message.contains("JESSICA_REPLY_") || message.contains("JESSICA_END_"))
                return null
            lines.joinToString(" ")
        }.replace(Regex("""\s+"""), " ").trim()
        if (body.length > 750 || body.isEmpty()) return null

        val stage = Regex("""^CALIBRI\s+S(\d{1,3})(?=[:.,;\s—–-])""",
            RegexOption.IGNORE_CASE).find(body)?.groupValues?.get(1) ?: return null
        if (stage.toIntOrNull() !in 1..999) return null
        if (!Regex("""(?:TRAIN|CPU|GPU|GitHub|Green\s+Tunnel|bpp|EOS)""",
                RegexOption.IGNORE_CASE).containsMatchIn(body)) return null
        if (!Regex("""Следующ(?:ий|ая)\s+шаг\s*[—–:-]\s*\S""",
                RegexOption.IGNORE_CASE).containsMatchIn(body)) return null
        if (!Regex("""LIVE\s+не\s+менять""", RegexOption.IGNORE_CASE)
                .containsMatchIn(body)) return null

        // Fail closed on requests requiring remote interpretation, execution,
        // credentials, user action, or an explicit plan termination.
        if (Regex(
                """(?:выполни|запусти|проведи|повтори|отправь)\s+(?:(?:один|новый|следующий|этот)\s+)?тест(?=$|[^\p{L}\p{N}])|(?:создай|открой)\s+(?:новый\s+)?тестовый\s+чат|(?:заверши|останови)\s+(?:сессию|Кодекс)|(?:план|исследование)\s+(?:полностью\s+)?заверш[её]н|https?://|[<>`«»]|(?:парол|секрет|API[- ]?ключ)|(?:deploy|merge|APK)""",
                RegexOption.IGNORE_CASE
            ).containsMatchIn(body)) return null

        return RelayReply(
            "CALIBRI S$stage: сообщение о ходе исследования принято. " +
            "Тест Jessica не запускала. Продолжай проверку через GitHub и сообщи следующий шаг. " +
            "Я остаюсь в этом чате."
        )
    }
}

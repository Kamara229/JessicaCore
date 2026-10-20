package com.jessica.core.modules.chat

/** Chat titles are display metadata, not instructions for Jessica's executor. */
object ChatTitlePolicy {
    private val explicitPattern = Regex(
        """(?:назови|назвать|переименуй|переименовать)\s+(?:этот\s+|текущий\s+)?чат\s*(?::|—|-|\s+как\b|\s+в\b)?\s*["«]?([^\n"»!?]{3,100})|(?:название|имя)\s+(?:этого\s+|текущего\s+)?чата\s*[:—-]\s*["«]?([^\n"»!?]{3,100})""",
        RegexOption.IGNORE_CASE
    )
    fun explicit(text: String): String? {
        val candidate = explicitPattern.find(text)?.groupValues
            ?.drop(1)?.firstOrNull { it.isNotBlank() } ?: return null
        return clean(candidate.trimEnd('.', ':', ';', ' '), 100).takeIf { it.length >= 3 }
    }
    fun clean(value: String, maxLength: Int = 64): String = value
        .replace(Regex("""[\r\n\t]+"""), " ")
        .replace(Regex("""\s+"""), " ")
        .trim(' ', '"', '«', '»', '.', ':', '—', '-')
        .take(maxLength)

    /** Immediate, offline fallback until a one-time AI title is returned. */
    fun fallback(text: String): String {
        val normalized = text.replace(Regex("""\s+"""), " ").trim()
        val cut = normalized
            .replace(Regex("""^(?:пожалуйста[,:]?\s*)?(?:мне\s+)?(?:сделай|напиши|найди|проверь|объясни|расскажи|помоги|создай|сравни|посмотри|покажи)\s+""", RegexOption.IGNORE_CASE), "")
            .trim()
        val words = cut.split(Regex("""\s+""")).take(7).joinToString(" ")
        return clean(words).ifBlank { "Новый чат" }
    }
}

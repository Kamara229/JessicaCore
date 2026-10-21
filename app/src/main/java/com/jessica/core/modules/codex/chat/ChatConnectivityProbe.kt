package com.jessica.core.modules.codex.chat

/** A local-only read/write smoke test of the ordinary ChatGPT Android UI. */
object ChatConnectivityProbe {
    fun prompt(id: String): String = listOf(
        "Это одноразовая проверка связи с обычным чатом ChatGPT.",
        "Ответь ТОЛЬКО тремя строками, без объяснений и Markdown:",
        "JESSICA_REPLY_$id",
        """{"nonce":"$id","ok":true}""",
        "JESSICA_END_$id"
    ).joinToString("\n")

    /** Not a cryptographic proof: only confirms that the current UI round trip was captured. */
    fun accepts(body: String, id: String): Boolean {
        val input = body.trim()
        return input.startsWith("{") && input.endsWith("}") &&
            Regex(""""nonce"\\s*:\\s*"${Regex.escape(id)}"""").containsMatchIn(input) &&
            Regex(""""ok"\\s*:\\s*true\\b""").containsMatchIn(input)
    }
}

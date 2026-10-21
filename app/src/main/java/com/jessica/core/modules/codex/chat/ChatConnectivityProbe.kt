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

    /** Only confirms the current UI round trip. No account data, keys or GitHub writes. */
    fun accepts(body: String, id: String): Boolean {
        val input = body.trim()
        val nonce = Regex("\"nonce\"\\s*:\\s*\"${Regex.escape(id)}\"")
        val ok = Regex("\"ok\"\\s*:\\s*true\\b")
        return input.startsWith("{") && input.endsWith("}") &&
            nonce.containsMatchIn(input) && ok.containsMatchIn(input)
    }
}

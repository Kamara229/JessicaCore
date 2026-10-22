package com.jessica.core.modules.codex.chat

/** A local-only read/write smoke test of the ordinary ChatGPT Android UI. */
object ChatConnectivityProbe {
    fun prompt(id: String): String = listOf(
        "Одноразовая проверка связи с обычным ChatGPT. Не выполняй никаких других действий.",
        "Верни ровно три строки, без форматирования и пояснений:",
        "Первая строка должна содержать только маркер JESSICA_REPLY_$id;",
        """вторая — только JSON {"nonce":"$id","ok":true};""",
        "третья — только маркер JESSICA_END_$id.",
        "Это описание желаемого ответа. Не повторяй текст инструкции."
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

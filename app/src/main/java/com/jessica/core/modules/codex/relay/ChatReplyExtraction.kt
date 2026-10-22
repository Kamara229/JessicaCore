package com.jessica.core.modules.codex.relay

/**
 * Reads an explicitly delimited, completed ChatGPT response from accessibility
 * text. Markdown and short line wraps around a marker are tolerated, but a
 * marker mentioned inside an instruction or quotation is never a delimiter.
 */
object ChatReplyExtraction {
    private val uuid = Regex("[a-f0-9-]{36}")
    private fun markerPiece(line: String): String =
        line.trim().trim('`', '*', ' ').replace("\u00a0", "").trim()

    private fun markerEnd(lines: List<String>, at: Int, label: String): Int? {
        val line = markerPiece(lines[at])
        if (line == label) return at + 1
        // ChatGPT sometimes renders the marker prefix and UUID as separate
        // text nodes, which accessibility exposes on adjacent lines.
        val underscore = label.lastIndexOf('_')
        val prefix = label.substring(0, underscore + 1)
        val suffix = label.substring(underscore + 1)
        if (line == prefix && at + 1 < lines.size &&
            markerPiece(lines[at + 1]) == suffix) return at + 2
        return null
    }

    /** In-progress response starting at its fresh marker; may lack END yet. */
    fun openingTail(screen: String, id: String): String? {
        if (screen.length > 120_000 || !uuid.matches(id)) return null
        val lines = screen.replace("\r\n", "\n").lines()
        var start = -1
        for (i in lines.indices) {
            if (markerEnd(lines, i, "JESSICA_REPLY_$id") != null) start = i
        }
        return if (start < 0) null else lines.subList(start, lines.size)
            .joinToString("\n").take(55_000)
    }

    fun read(screen: String, id: String): String? {
        if (screen.length > 120_000 || !uuid.matches(id)) return null
        val lines = screen.replace("\r\n", "\n").lines()
        val firstLabel = "JESSICA_REPLY_$id"
        val endLabel = "JESSICA_END_$id"
        var latest: String? = null
        for (index in lines.indices) {
            val bodyStart = markerEnd(lines, index, firstLabel) ?: continue
            for (last in bodyStart until lines.size) {
                val end = markerEnd(lines, last, endLabel) ?: continue
                val answer = lines.subList(bodyStart, last).joinToString("\n").trim()
                if (answer.isNotEmpty() && answer.length <= 50_000) latest = answer
                break
            }
        }
        return latest
    }
}

package com.jessica.core.modules.codex.relay

/**
 * Reads an explicitly delimited, completed ChatGPT response from accessibility
 * text. Markdown and short line wraps around a marker are tolerated, but a
 * marker mentioned inside an instruction or quotation is never a delimiter.
 */
object ChatReplyExtraction {
    private val uuid = Regex("[a-f0-9-]{36}")
    private fun markerPiece(line: String): String =
        line.replace("\u00a0", " ").replace("\u200b", "").replace("\u2060", "")
            .trim().trim('>', '-', '*', '\u0060', ' ', ':', ';').trim()

    /**
     * ChatGPT accessibility may flatten visible line breaks into a single node.
     * A marker embedded inside the owner's instructions is NEVER a delimiter.
     */
    private fun accessibilityLines(screen: String, id: String): List<String> {
        val reply = "JESSICA_REPLY_$id"
        val end = "JESSICA_END_$id"
        val original = screen.replace("\r\n", "\n").lines()
        val output = mutableListOf<String>()
        for (line in original) {
            val cleaned = line.trim().trimStart('>', '-', '*', '\u0060', ' ', '\t').trim()
            if (cleaned.startsWith(reply) &&
                (cleaned.length == reply.length ||
                    cleaned[reply.length] in listOf(' ', ':', ';', '*', '\u0060'))) {
                output.add(reply)
                val rest = cleaned.substring(reply.length).trimStart(' ', ':', ';', '*', '\u0060')
                val position = rest.lastIndexOf(end)
                if (position > 0 && rest.substring(position + end.length)
                    .trim(' ', '*', '\u0060', ';', ':').isEmpty()) {
                    output.add(rest.substring(0, position).trimEnd())
                    output.add(end)
                } else if (rest.isNotBlank()) output.add(rest)
            } else if (cleaned.startsWith(end) &&
                cleaned.substring(end.length).trim(' ', '*', '\u0060', ';', ':').isEmpty()) {
                output.add(end)
            } else output.add(line)
        }
        return output
    }

    private fun markerEnd(lines: List<String>, at: Int, label: String): Int? {
        val first = markerPiece(lines[at])
        if (first == label) return at + 1
        // ChatGPT may wrap a 36-character UUID in the *middle* of a marker.
        // Join only 2–4 adjacent marker-only fragments, never free-form text.
        // The owner's instruction "Начни ответ строкой JESSICA_REPLY_..."
        // cannot match, because that line does not START with this marker.
        if (!first.startsWith("JESSICA_") || !label.startsWith(first)) return null
        var joined = first
        for (next in at + 1 until minOf(lines.size, at + 4)) {
            val part = markerPiece(lines[next])
            if (part.isBlank() || !part.all { ch ->
                ch.isLetterOrDigit() || ch == '-' || ch == '_'
            }) return null
            joined += part
            if (joined == label) return next + 1
            if (!label.startsWith(joined)) return null
        }
        return null
    }

    /** A quoted marker in the owner's request is not a standalone answer end. */
    fun hasStandaloneEnd(screen: String, id: String): Boolean {
        if (screen.length > 120_000 || !uuid.matches(id)) return false
        val lines = accessibilityLines(screen, id)
        return lines.indices.any { markerEnd(lines, it, "JESSICA_END_$id") != null }
    }

    /** In-progress response starting at its fresh marker; may lack END yet. */
    fun openingTail(screen: String, id: String): String? {
        if (screen.length > 120_000 || !uuid.matches(id)) return null
        val lines = accessibilityLines(screen, id)
        var start = -1
        for (i in lines.indices) {
            if (markerEnd(lines, i, "JESSICA_REPLY_$id") != null) start = i
        }
        return if (start < 0) null else lines.subList(start, lines.size)
            .joinToString("\n").take(55_000)
    }

    fun read(screen: String, id: String): String? {
        if (screen.length > 120_000 || !uuid.matches(id)) return null
        val lines = accessibilityLines(screen, id)
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

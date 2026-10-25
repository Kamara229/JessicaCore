package com.jessica.core.modules.chat

import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.security.MessageDigest

/** Validated, bounded JSON wire format. Raw document content stays in the chat request. */
object DocumentPageWire {
    fun decode(value: JSONArray?, name: String): List<DocumentPage> {
        if (value == null) return emptyList() // older backend: legacy excerpt
        if (value.length() !in 1..16) throw IOException("Неполный набор страниц документа")
        val pages = ArrayList<DocumentPage>(value.length())
        val combined = StringBuilder()
        var next = 0
        var total = -1
        var hash = ""
        for (i in 0 until value.length()) {
            val json = value.getJSONObject(i)
            val text = json.getString("text")
            val len = json.getInt("extractedChars")
            val offset = json.getInt("offset")
            val hasMore = json.getBoolean("hasMore")
            val nextOffset = if (json.isNull("nextOffset")) null else json.getInt("nextOffset")
            val fingerprint = json.getString("sourceSha256")
            if (text.isEmpty() || text.length > 16000 ||
                offset != next || len !in 1..256000 || (total != -1 && total != len) ||
                !Regex("^[a-f0-9]{64}$").matches(fingerprint) ||
                (hash.isNotEmpty() && hash != fingerprint))
                throw IOException("Нарушена целостность страниц")
            val end = offset + text.length
            if (end > len || hasMore != (i < value.length() - 1) ||
                nextOffset != (if (hasMore) end else null) ||
                (!hasMore && end != len))
                throw IOException("Страницы документа неполны")
            combined.append(text)
            pages.add(DocumentPage(name, text, json.optBoolean("truncated"),
                len, offset, nextOffset, hasMore, fingerprint))
            next = end
            total = len
            hash = fingerprint
        }
        val actual = MessageDigest.getInstance("SHA-256")
            .digest(combined.toString().toByteArray(Charsets.UTF_8))
            .joinToString("") { (it.toInt() and 255).toString(16).padStart(2, '0') }
        if (actual != hash) throw IOException("Контрольная сумма файла не совпадает")
        return pages
    }

    fun encode(pages: List<DocumentPage>): JSONArray = JSONArray().apply {
        pages.forEach { p ->
            put(JSONObject().apply {
                put("name", p.name)
                put("text", p.text)
                put("truncated", p.truncated)
                put("extractedChars", p.extractedChars)
                put("offset", p.offset)
                put("nextOffset", p.nextOffset ?: JSONObject.NULL)
                put("hasMore", p.hasMore)
                put("sourceSha256", p.sourceSha256)
            })
        }
    }
}

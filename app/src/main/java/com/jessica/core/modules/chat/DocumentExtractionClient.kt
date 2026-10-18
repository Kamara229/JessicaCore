package com.jessica.core.modules.chat

import com.jessica.core.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import android.net.Uri
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * Upload only when a user sends a message with selected documents.
 * The server extracts a bounded text excerpt; original binary remains on device.
 */
class DocumentExtractionClient(private val store: ChatAttachmentStore) {
    suspend fun extract(attachments: List<ChatAttachment>): List<DocumentContext> =
        withContext(Dispatchers.IO) {
            if (attachments.size > 3) throw IOException("За один раз можно отправить до 3 файлов")
            attachments.map { extractOne(it) }
        }

    private fun extractOne(attachment: ChatAttachment): DocumentContext {
        val file = store.file(attachment)
        if (file.length() > 5L * 1024L * 1024L) throw IOException("Слишком большой файл")
        val connection = URL("https://jessicacore.onrender.com/api/files/extract")
            .openConnection() as HttpURLConnection
        return try {
            connection.requestMethod = "POST"
            connection.doOutput = true
            connection.connectTimeout = 45_000
            connection.readTimeout = 90_000
            connection.setFixedLengthStreamingMode(file.length())
            connection.setRequestProperty("Content-Type", "application/octet-stream")
            connection.setRequestProperty("X-Jessica-Token", BuildConfig.JESSICA_APP_TOKEN)
            connection.setRequestProperty("X-Jessica-Filename", Uri.encode(attachment.name))
            file.inputStream().use { input ->
                connection.outputStream.use { output -> input.copyTo(output) }
            }
            val response = if (connection.responseCode in 200..299)
                connection.inputStream.bufferedReader().use { it.readText() }
            else connection.errorStream?.bufferedReader()?.use { it.readText() }.orEmpty()
            val json = try { JSONObject(response) } catch (_: Exception) { JSONObject() }
            if (connection.responseCode !in 200..299 || !json.optBoolean("success")) {
                throw IOException(json.optString("text").ifBlank {
                    "Ошибка чтения файла ${attachment.name}: HTTP ${connection.responseCode}"
                })
            }
            DocumentContext(
                name = attachment.name,
                text = json.getString("text"),
                truncated = json.optBoolean("truncated", false)
            )
        } finally {
            connection.disconnect()
        }
    }
}

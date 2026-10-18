package com.jessica.core.modules.chat

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.UUID

/** Keep original uploaded files in the private Jessica app folder. */
class ChatAttachmentStore(context: Context) {
    private val resolver = context.applicationContext.contentResolver
    private val dir = File(context.applicationContext.filesDir, "jessica/chat/attachments")
    private val allowed = setOf("pdf", "docx", "xlsx", "txt", "md", "csv", "json")
    private val maxSize = 5L * 1024L * 1024L

    suspend fun importUri(uri: Uri): ChatAttachment = withContext(Dispatchers.IO) {
        val name = filename(uri)
        val extension = name.substringAfterLast(".", "").lowercase()
        if (extension !in allowed) {
            throw IOException("Формат $extension не поддерживается. Выбери PDF, DOCX, XLSX, TXT, MD, CSV или JSON")
        }
        if (!dir.exists() && !dir.mkdirs()) throw IOException("Не удалось создать папку файлов")
        val id = UUID.randomUUID().toString()
        val file = File(dir, id)
        try {
            var size = 0L
            (resolver.openInputStream(uri) ?: throw IOException("Не удалось открыть файл")).use { input ->
                FileOutputStream(file).use { output ->
                    val buffer = ByteArray(32 * 1024)
                    while (true) {
                        val n = input.read(buffer)
                        if (n < 0) break
                        size += n
                        if (size > maxSize) throw IOException("Максимальный размер файла — 5 МБ")
                        output.write(buffer, 0, n)
                    }
                    output.fd.sync()
                }
            }
            if (size == 0L) throw IOException("Пустой файл")
            ChatAttachment(id, name, resolver.getType(uri) ?: "application/octet-stream", size)
        } catch (error: Exception) {
            file.delete()
            throw error
        }
    }

    fun file(attachment: ChatAttachment): File {
        if (UUID.fromString(attachment.id).toString() != attachment.id) {
            throw IOException("Неверный идентификатор файла")
        }
        val result = File(dir, attachment.id)
        if (!result.isFile) throw IOException("Файл не найден на телефоне: ${attachment.name}")
        return result
    }

    fun discard(attachment: ChatAttachment) {
        try { file(attachment).delete() } catch (_: Exception) { }
    }

    private fun filename(uri: Uri): String {
        var filename: String? = null
        resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) filename = c.getString(0)
        }
        return (filename ?: uri.lastPathSegment ?: "document.txt")
            .substringAfterLast('/').substringAfterLast('\\')
            .replace(Regex("[\\r\\n\\t]"), " ")
            .take(100)
            .ifBlank { "document.txt" }
    }
}

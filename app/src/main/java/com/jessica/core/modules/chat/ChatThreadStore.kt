package com.jessica.core.modules.chat

import android.content.Context
import android.util.AtomicFile
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.UUID

data class ChatThread(
    val id: String,
    val title: String,
    val updatedAt: Long,
    val titleLocked: Boolean = false,
    val autoTitleFinal: Boolean = false,
    val deletedAt: Long? = null
)

/**
 * Local chat storage. Each conversation is isolated from other conversations.
 * Old history.jsonl is COPIED into its own thread during the first migration.
 * The original file is never deleted, so upgrading cannot destroy old history.
 */
data class ChatSearchResult(val thread: ChatThread, val snippet: String)

class ChatThreadStore(context: Context) {
    private val appContext = context.applicationContext
    private val root = File(appContext.filesDir, "jessica/chat")
    private val threadsDir = File(root, "threads")
    private val index = AtomicFile(File(root, "threads_index.json"))
    private val preferences = appContext.getSharedPreferences("jessica_chats", Context.MODE_PRIVATE)

    @Synchronized
    fun list(): List<ChatThread> {
        ensureMigrated()
        purgeExpired()
        return readIndex().filter { it.deletedAt == null }.sortedByDescending { it.updatedAt }
    }

    @Synchronized
    fun deleted(): List<ChatThread> {
        ensureMigrated()
        purgeExpired()
        return readIndex().filter { it.deletedAt != null }
            .sortedByDescending { it.deletedAt ?: 0L }
    }

    @Synchronized
    fun activeId(): String? = preferences.getString("active_id", null)

    @Synchronized
    fun setActive(id: String) {
        if (list().none { it.id == id }) throw IOException("Диалог не найден")
        if (!preferences.edit().putString("active_id", id).commit()) {
            throw IOException("Не удалось сохранить выбранный диалог")
        }
    }

    @Synchronized
    fun create(): ChatThread {
        val existing = list()
        val thread = ChatThread(UUID.randomUUID().toString(), "Новый чат", 0L)
        writeIndex(existing + thread)
        return thread
    }

    @Synchronized
    fun messages(id: String): List<ChatMessage> {
        if (list().none { it.id == id }) throw IOException("Диалог не найден")
        val file = threadFile(id)
        if (!file.exists()) return emptyList()
        return readMessages(file)
    }

    /**
     * Read-only access for a soft-deleted chat. Does not reactivate it,
     * update last activity or alter the active conversation.
     */
    @Synchronized
    fun deletedMessages(id: String): List<ChatMessage> {
        ensureMigrated()
        purgeExpired()
        val thread = readIndex().firstOrNull { it.id == id }
            ?: throw IOException("Удалённый чат не найден")
        if (!ChatTrashPolicy.canRestore(thread.deletedAt))
            throw IOException("Срок восстановления чата истёк")
        val file = threadFile(id)
        return if (file.isFile) readMessages(file) else emptyList()
    }

    @Synchronized
    fun append(id: String, message: ChatMessage) {
        val threads = list()
        val thread = threads.find { it.id == id } ?: throw IOException("Диалог не найден")
        if (!threadsDir.isDirectory && !threadsDir.mkdirs()) {
            throw IOException("Не удалось открыть папку диалогов")
        }
        val json = JSONObject().apply {
            put("text", message.text)
            put("isUser", message.isUser)
            put("timestamp", message.timestamp)
            val attached = JSONArray()
            message.attachments.forEach { a ->
                attached.put(JSONObject().apply {
                    put("id", a.id)
                    put("name", a.name)
                    put("mime", a.mime)
                    put("sizeBytes", a.sizeBytes)
                })
            }
            put("attachments", attached)
        }
        FileOutputStream(threadFile(id), true).use { stream ->
            stream.write((json.toString() + "\n").toByteArray(Charsets.UTF_8))
            stream.fd.sync()
        }
        val requestedName = if (message.isUser) ChatTitlePolicy.explicit(message.text) else null
        val firstMessage = message.isUser && thread.title == "Новый чат" && !thread.titleLocked
        val newTitle = requestedName ?: if (firstMessage) ChatTitlePolicy.fallback(message.text) else thread.title
        val newTimestamp = maxOf(System.currentTimeMillis(), (threads.maxOfOrNull { it.updatedAt } ?: 0L) + 1L)
        val updated = thread.copy(
            title = newTitle,
            updatedAt = newTimestamp,
            titleLocked = thread.titleLocked || requestedName != null,
            autoTitleFinal = if (requestedName != null) true else thread.autoTitleFinal
        )
        writeIndex(threads.map { if (it.id == id) updated else it })
    }

    @Synchronized
    fun applyAutomaticTitle(id: String, rawTitle: String): Boolean {
        val threads = list()
        val thread = threads.find { it.id == id } ?: return false
        if (thread.titleLocked || thread.autoTitleFinal) return false
        val title = ChatTitlePolicy.clean(rawTitle)
        if (title.length < 3) return false
        writeIndex(threads.map { if (it.id == id) thread.copy(title = title, autoTitleFinal = true) else it })
        return true
    }

    /** Soft deletion: neither messages nor attachments are touched for 60 minutes. */
    @Synchronized
    fun delete(id: String): List<ChatThread> {
        val visible = list()
        if (visible.none { it.id == id }) throw IOException("Чат не найден")
        val all = readIndex()
        val timestamp = System.currentTimeMillis()
        writeIndex(all.map { if (it.id == id) it.copy(deletedAt = timestamp) else it })
        if (activeId() == id) {
            val next = list().firstOrNull()?.id
            if (!preferences.edit().putString("active_id", next).commit())
                throw IOException("Чат скрыт, но не удалось сменить активный чат")
        }
        // WorkManager runs after one hour, even if the app leaves the foreground.
        // Android may defer execution; list/search also prune on next access.
        try { ChatTrashScheduler.schedule(appContext, id) } catch (_: Exception) { }
        return list()
    }

    @Synchronized
    fun restore(id: String): List<ChatThread> {
        ensureMigrated()
        purgeExpired()
        val all = readIndex()
        val target = all.find { it.id == id } ?: throw IOException("Удалённый чат не найден")
        if (!ChatTrashPolicy.canRestore(target.deletedAt))
            throw IOException("Срок восстановления чата истёк")
        writeIndex(all.map { if (it.id == id) it.copy(deletedAt = null) else it })
        try { ChatTrashScheduler.cancel(appContext, id) } catch (_: Exception) { }
        return list()
    }

    /** Idempotent permanent cleanup; called by WorkManager and while opening the app. */
    @Synchronized
    fun purgeExpired(): Int {
        ensureMigrated()
        var deleted = 0
        val now = System.currentTimeMillis()
        for (thread in readIndex().filter { ChatTrashPolicy.isExpired(it.deletedAt, now) }) {
            val original = threadFile(thread.id)
            val staged = File(threadsDir, "${thread.id}.deleting")
            val legacyCopy = isMigratedLegacyCopy(original)
            if (original.exists() && !original.renameTo(staged))
                throw IOException("Невозможно подготовить окончательное удаление чата")
            try {
                writeIndex(readIndex().filterNot { it.id == thread.id && it.deletedAt != null })
            } catch (error: Exception) {
                if (staged.exists()) staged.renameTo(original)
                throw error
            }
            val attachments = if (staged.isFile) readMessages(staged).flatMap { it.attachments } else emptyList()
            if (!staged.exists() || staged.delete()) {
                attachments.forEach { ChatAttachmentStore(appContext).discard(it) }
            }
            if (legacyCopy) File(root, "history.jsonl").delete()
            deleted++
        }
        // Recover after a crash between atomic index replacement and file unlink.
        // Never erase an orphan whose index still points to an existing chat.
        threadsDir.listFiles()?.filter { it.name.endsWith(".deleting") }?.forEach { staged ->
            val id = staged.name.removeSuffix(".deleting")
            val valid = try { UUID.fromString(id).toString() == id }
                catch (_: Exception) { false }
            if (!valid || !staged.isFile) return@forEach
            val original = threadFile(id)
            if (readIndex().any { it.id == id }) {
                if (!original.exists()) staged.renameTo(original)
                return@forEach
            }
            val attachments = readMessages(staged).flatMap { it.attachments }
            if (staged.delete()) attachments.forEach { ChatAttachmentStore(appContext).discard(it) }
        }
        return deleted
    }

    @Synchronized
    fun search(query: String): List<ChatSearchResult> {
        val needle = query.trim()
        if (needle.isEmpty()) return emptyList()
        ensureMigrated()
        purgeExpired()
        val trashSearch = needle.lowercase() in setOf("удаленные", "удалённые", "корзина", "deleted", "trash")
        return readIndex().sortedByDescending { it.updatedAt }.mapNotNull { thread ->
            if (trashSearch && thread.deletedAt == null) return@mapNotNull null
            if (trashSearch) return@mapNotNull ChatSearchResult(thread, "")
            val titleMatch = thread.title.contains(needle, ignoreCase = true)
            val file = threadFile(thread.id)
            val match = if (titleMatch || !file.isFile) null else readMessages(file)
                .firstOrNull { it.text.contains(needle, ignoreCase = true) }
            if (!titleMatch && match == null) return@mapNotNull null
            val text = match?.text.orEmpty()
            val at = text.indexOf(needle, ignoreCase = true).coerceAtLeast(0)
            val snippet = if (match == null) "" else text
                .substring((at - 42).coerceAtLeast(0), (at + needle.length + 86).coerceAtMost(text.length))
                .replace(Regex("\\s+"), " ")
            ChatSearchResult(thread, snippet)
        }
    }

    /** The original pre-thread archive is a copy of the migrated conversation.
     * It is removed only when that exact conversation is explicitly deleted.
     */
    private fun isMigratedLegacyCopy(file: File): Boolean {
        val legacy = File(root, "history.jsonl")
        if (!file.isFile || !legacy.isFile || legacy.length() <= 0 ||
            file.length() < legacy.length()) return false
        return try {
            legacy.inputStream().use { a ->
                file.inputStream().use { b ->
                    val left = ByteArray(8192)
                    val right = ByteArray(8192)
                    while (true) {
                        val count = a.read(left)
                        if (count < 0) break
                        var offset = 0
                        while (offset < count) {
                            val n = b.read(right, offset, count - offset)
                            if (n <= 0) return false
                            offset += n
                        }
                        if (!left.copyOf(count).contentEquals(right.copyOf(count))) return false
                    }
                    true
                }
            }
        } catch (_: IOException) { false }
    }

    private fun threadFile(id: String): File {
        // Never use arbitrary input as a filesystem path.
        val valid = try { UUID.fromString(id).toString() == id } catch (_: Exception) { false }
        if (!valid) throw IOException("Некорректный идентификатор диалога")
        return File(threadsDir, "$id.jsonl")
    }

    private fun ensureMigrated() {
        if (index.baseFile.exists()) return
        if (!root.isDirectory && !root.mkdirs()) throw IOException("Не удалось создать папку Jessica")
        if (!threadsDir.isDirectory && !threadsDir.mkdirs()) throw IOException("Не удалось создать папку чатов")
        val legacy = File(root, "history.jsonl")
        val migrated = mutableListOf<ChatThread>()
        if (legacy.isFile && legacy.length() > 0L) {
            val history = readMessages(legacy)
            if (history.isEmpty()) throw IOException("Старая история повреждена; оригинал сохранён")
            val id = UUID.randomUUID().toString()
            // Write the copy BEFORE the index, so migration can retry safely.
            legacy.copyTo(threadFile(id), overwrite = true)
            migrated.add(ChatThread(
                id,
                history.firstOrNull { it.isUser }?.text?.trim()
                    ?.replace(Regex("\\s+"), " ")?.take(48)?.ifBlank { "Первый чат" }
                    ?: "Первый чат",
                history.last().timestamp.coerceAtLeast(System.currentTimeMillis() - 1000L)
            ))
        }
        writeIndex(migrated)
    }

    private fun readIndex(): List<ChatThread> {
        if (!index.baseFile.exists()) return emptyList()
        val text = index.openRead().bufferedReader(Charsets.UTF_8).use { it.readText() }
        val json = JSONArray(text)
        return (0 until json.length()).map { i ->
            val obj = json.getJSONObject(i)
            ChatThread(
                id = obj.getString("id"),
                title = obj.getString("title"),
                updatedAt = obj.getLong("updatedAt"),
                titleLocked = obj.optBoolean("titleLocked", false),
                autoTitleFinal = obj.optBoolean("autoTitleFinal", false),
                deletedAt = obj.optLong("deletedAt", 0L).takeIf { it > 0L }
            )
        }
    }

    private fun writeIndex(threads: List<ChatThread>) {
        val array = JSONArray()
        threads.forEach { thread ->
            array.put(JSONObject().apply {
                put("id", thread.id)
                put("title", thread.title)
                put("updatedAt", thread.updatedAt)
                put("titleLocked", thread.titleLocked)
                put("autoTitleFinal", thread.autoTitleFinal)
                if (thread.deletedAt != null) put("deletedAt", thread.deletedAt)
            })
        }
        val stream = index.startWrite()
        try {
            stream.write(array.toString().toByteArray(Charsets.UTF_8))
            index.finishWrite(stream)
        } catch (error: Exception) {
            index.failWrite(stream)
            throw error
        }
    }

    private fun parseAttachments(array: JSONArray?): List<ChatAttachment> {
        if (array == null) return emptyList()
        return (0 until array.length()).mapNotNull { index ->
            try {
                val obj = array.getJSONObject(index)
                ChatAttachment(
                    id = obj.getString("id"),
                    name = obj.getString("name"),
                    mime = obj.optString("mime", "application/octet-stream"),
                    sizeBytes = obj.optLong("sizeBytes", 0L)
                )
            } catch (_: Exception) {
                null
            }
        }
    }

    private fun readMessages(file: File): List<ChatMessage> {
        val result = mutableListOf<ChatMessage>()
        file.bufferedReader(Charsets.UTF_8).use { reader ->
            reader.forEachLine { line ->
                try {
                    val json = JSONObject(line)
                    if (json.has("isUser") && json.optString("text").isNotBlank()) {
                        result.add(ChatMessage(
                            text = json.getString("text"),
                            isUser = json.getBoolean("isUser"),
                            timestamp = json.optLong("timestamp", 0L),
                            attachments = parseAttachments(json.optJSONArray("attachments"))
                        ))
                    }
                } catch (_: Exception) {
                    // The final partially written line does not erase valid records.
                }
            }
        }
        return result
    }
}

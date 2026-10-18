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
    val updatedAt: Long
)

/**
 * Local chat storage. Each conversation is isolated from other conversations.
 * Old history.jsonl is COPIED into its own thread during the first migration.
 * The original file is never deleted, so upgrading cannot destroy old history.
 */
class ChatThreadStore(context: Context) {
    private val appContext = context.applicationContext
    private val root = File(appContext.filesDir, "jessica/chat")
    private val threadsDir = File(root, "threads")
    private val index = AtomicFile(File(root, "threads_index.json"))
    private val preferences = appContext.getSharedPreferences("jessica_chats", Context.MODE_PRIVATE)

    @Synchronized
    fun list(): List<ChatThread> {
        ensureMigrated()
        return readIndex().sortedByDescending { it.updatedAt }
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
        val thread = ChatThread(UUID.randomUUID().toString(), "Новый чат", System.currentTimeMillis())
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
        }
        FileOutputStream(threadFile(id), true).use { stream ->
            stream.write((json.toString() + "\n").toByteArray(Charsets.UTF_8))
            stream.fd.sync()
        }
        val newTitle = if (message.isUser && thread.title == "Новый чат") {
            message.text.trim().replace(Regex("\\s+"), " ").take(48).ifBlank { "Новый чат" }
        } else thread.title
        val updated = thread.copy(title = newTitle, updatedAt = System.currentTimeMillis())
        writeIndex(threads.map { if (it.id == id) updated else it })
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
                updatedAt = obj.getLong("updatedAt")
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
                            timestamp = json.optLong("timestamp", 0L)
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

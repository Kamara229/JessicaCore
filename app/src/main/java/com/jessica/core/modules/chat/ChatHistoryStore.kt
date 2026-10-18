package com.jessica.core.modules.chat

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/** App-private archive: files/jessica/chat/history.jsonl; survives upgrades. */
class ChatHistoryStore(context: Context) {
    private val directory = File(context.applicationContext.filesDir, "jessica/chat")
    private val archive = File(directory, "history.jsonl")

    @Synchronized
    fun load(): List<ChatMessage> {
        if (!archive.exists()) return emptyList()
        val messages = mutableListOf<ChatMessage>()
        archive.bufferedReader(Charsets.UTF_8).use { reader ->
            reader.forEachLine { line ->
                try {
                    val item = JSONObject(line)
                    if (item.has("isUser") && item.optString("text").isNotBlank()) {
                        messages.add(
                            ChatMessage(
                                text = item.getString("text"),
                                isUser = item.getBoolean("isUser"),
                                timestamp = item.optLong("timestamp", 0L)
                            )
                        )
                    }
                } catch (_: Exception) {
                    // A truncated final record must not destroy earlier messages.
                }
            }
        }
        return messages
    }

    @Synchronized
    fun append(message: ChatMessage) {
        if (!directory.isDirectory && !directory.mkdirs()) {
            throw IOException("Не удалось создать папку истории Jessica")
        }
        val record = JSONObject().apply {
            put("text", message.text)
            put("isUser", message.isUser)
            put("timestamp", message.timestamp)
        }.toString() + "\n"
        FileOutputStream(archive, true).use { output ->
            output.write(record.toByteArray(Charsets.UTF_8))
            output.fd.sync()
        }
    }
}

package com.jessica.core.modules.chat

import com.jessica.core.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.io.IOException

/** Private test-screen mirror of all local chats. No attachment bytes. */
class ChatDebugSyncClient {
    suspend fun sync(store: ChatThreadStore): String? = withContext(Dispatchers.IO) {
        if (BuildConfig.JESSICA_APP_TOKEN.isBlank()) return@withContext "Отсутствует токен диагностики"
        val threads = store.list()
        val data = JSONArray()
        for (thread in threads) {
            val content = JSONArray()
            for (message in store.messages(thread.id)) {
                content.put(JSONObject().apply {
                    put("text", message.text)
                    put("isUser", message.isUser)
                    put("timestamp", message.timestamp)
                })
            }
            data.put(JSONObject().apply {
                put("id", thread.id)
                put("title", thread.title)
                put("updatedAt", thread.updatedAt)
                put("messages", content)
            })
        }
        val body=JSONObject().apply {
            put("threads", data)
            put("activeThreadId", store.activeId())
        }.toString().toByteArray(Charsets.UTF_8)
        if (body.size > 850_000) return@withContext "История слишком велика для тестового экрана, ничего не обрезано"
        val connection=URL("https://jessicacore.onrender.com/api/chat/debug/sync")
            .openConnection() as HttpURLConnection
        try {
            connection.requestMethod="POST"
            connection.connectTimeout=8000
            connection.readTimeout=12000
            connection.doOutput=true
            connection.setRequestProperty("Content-Type","application/json; charset=UTF-8")
            connection.setRequestProperty("X-Jessica-Token",BuildConfig.JESSICA_APP_TOKEN)
            connection.outputStream.use { it.write(body) }
            if(connection.responseCode !in 200..299) {
                val detail=connection.errorStream?.bufferedReader()?.use { it.readText() }.orEmpty()
                return@withContext "Диагностика не синхронизирована: HTTP " + connection.responseCode + " " + detail.take(80)
            }
            val response=JSONObject(connection.inputStream.bufferedReader().use {it.readText()})
            if(!response.optBoolean("success")) "Ошибка синхронизации диагностики" else null
        } catch (error: IOException) {
            "Нет соединения с тестовым экраном: " + error.message.orEmpty().take(90)
        } finally {
            connection.disconnect()
        }
    }
}

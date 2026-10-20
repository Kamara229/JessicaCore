package com.jessica.core.modules.chat

import com.jessica.core.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/** One short AI title, after a new chat receives its first answer. */
class ChatTitleClient {
    suspend fun suggest(firstTask: String, firstAnswer: String): String? = withContext(Dispatchers.IO) {
        if (BuildConfig.JESSICA_APP_TOKEN.isBlank()) return@withContext null
        val connection = URL("https://jessicacore.onrender.com/api/chat/title")
            .openConnection() as HttpURLConnection
        try {
            connection.requestMethod = "POST"
            connection.connectTimeout = 8000
            connection.readTimeout = 12000
            connection.doOutput = true
            connection.setRequestProperty("Content-Type", "application/json; charset=UTF-8")
            connection.setRequestProperty("X-Jessica-Token", BuildConfig.JESSICA_APP_TOKEN)
            val body = JSONObject().apply {
                put("task", firstTask.take(1800))
                put("answer", firstAnswer.take(1200))
            }.toString()
            connection.outputStream.use { it.write(body.toByteArray(Charsets.UTF_8)) }
            if (connection.responseCode !in 200..299) return@withContext null
            val json = JSONObject(connection.inputStream.bufferedReader().use { it.readText() })
            ChatTitlePolicy.clean(json.optString("title")).takeIf { it.length >= 3 }
        } catch (_: Exception) {
            null
        } finally {
            connection.disconnect()
        }
    }
}

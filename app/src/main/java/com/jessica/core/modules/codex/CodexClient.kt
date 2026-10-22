package com.jessica.core.modules.codex

import com.jessica.core.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/** Uses Jessica's existing understanding service. No GitHub, Render, owner key or paid Codex API. */
class CodexClient {
    suspend fun interpret(payload: JSONObject): JSONObject = withContext(Dispatchers.IO) {
        // Interpretation has no side effects. Only transient transport failures may be retried;
        // executing a test and sending a ChatGPT message are never retried here.
        for (attempt in 0..2) {
            try { return@withContext request(payload) }
            catch (e: TransientFailure) {
                if (attempt == 2) throw e
                delay((attempt + 1) * 3000L)
            }
        }
        error("Понимание ответа недоступно")
    }

    private class TransientFailure(message: String, cause: Throwable? = null) : IOException(message, cause)

    private fun request(payload: JSONObject): JSONObject {
        val connection = URL("https://jessicacore.onrender.com/api/codex/relay/interpret").openConnection() as HttpURLConnection
        try {
            connection.requestMethod = "POST"
            connection.connectTimeout = 45000
            connection.readTimeout = 90000
            connection.doOutput = true
            connection.setRequestProperty("X-Jessica-Token", BuildConfig.JESSICA_APP_TOKEN)
            connection.setRequestProperty("Content-Type", "application/json; charset=UTF-8")
            connection.outputStream.use { it.write(payload.toString().toByteArray(Charsets.UTF_8)) }
            val code = connection.responseCode
            val raw = (if (code in 200..299) connection.inputStream else connection.errorStream)
                ?.bufferedReader()?.use { it.readText() }.orEmpty()
            if (code in setOf(429, 502, 504) || (code == 503 && !raw.contains("relay-understanding-unavailable")))
                throw TransientFailure("Сервер временно недоступен: HTTP $code")
            val json = runCatching { JSONObject(raw) }.getOrElse { throw IOException("Сервер вернул непонятный ответ") }
            if (code !in 200..299 || !json.optBoolean("success"))
                throw IOException(json.optString("text", "Понимание ответа недоступно: HTTP $code"))
            return json
        } catch (e: java.net.SocketTimeoutException) {
            throw TransientFailure("Сервер не ответил вовремя", e)
        } catch (e: java.net.ConnectException) {
            throw TransientFailure("Не удалось подключиться к серверу", e)
        } finally { connection.disconnect() }
    }
}

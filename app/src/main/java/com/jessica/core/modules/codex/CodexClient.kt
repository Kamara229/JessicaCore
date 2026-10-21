package com.jessica.core.modules.codex

import com.jessica.core.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/** No GitHub or OpenAI credentials are ever packaged into the Android APK. */
class CodexClient {
    private val origin = "https://jessicacore.onrender.com"

    suspend fun configuration(): JSONObject = request("/api/codex/config", null)

    /** Start an API-free day shift; no OpenAI/Codex billing key is sent or used. */
    suspend fun startDay(ownerKey: String, goal: String = "auto"): JSONObject =
        request("/api/codex/day/start", JSONObject().apply {
            put("ownerKey", ownerKey)
            put("goal", goal)
        })

    suspend fun start(ownerKey: String, goal: String): JSONObject =
        request("/api/codex/start", JSONObject().apply {
            put("ownerKey", ownerKey)
            put("goal", goal)
        })

    suspend fun status(ownerKey: String, issue: Int): JSONObject =
        request("/api/codex/status", JSONObject().apply {
            put("ownerKey", ownerKey)
            put("issue", issue)
        })

    suspend fun stop(ownerKey: String, issue: Int): JSONObject =
        request("/api/codex/stop", JSONObject().apply {
            put("ownerKey", ownerKey)
            put("issue", issue)
        })

    private suspend fun request(path: String, payload: JSONObject?): JSONObject =
        withContext(Dispatchers.IO) {
            val connection = URL(origin + path).openConnection() as HttpURLConnection
            try {
                connection.requestMethod = if (payload == null) "GET" else "POST"
                connection.connectTimeout = 12_000
                connection.readTimeout = 20_000
                connection.setRequestProperty("X-Jessica-Token", BuildConfig.JESSICA_APP_TOKEN)
                connection.setRequestProperty("Accept", "application/json")
                connection.setRequestProperty("Cache-Control", "no-store")
                if (payload != null) {
                    connection.doOutput = true
                    connection.setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                    connection.outputStream.use { it.write(payload.toString().toByteArray(Charsets.UTF_8)) }
                }
                val code = connection.responseCode
                val body = (if (code in 200..299) connection.inputStream else connection.errorStream)
                    ?.bufferedReader()?.use { it.readText() }.orEmpty()
                val json = try { JSONObject(body) } catch (_: Exception) { JSONObject() }
                if (code !in 200..299) {
                    val reason = json.optString("reason").ifBlank { "HTTP " + code }
                    throw IOException(reason)
                }
                json
            } finally {
                connection.disconnect()
            }
        }
}

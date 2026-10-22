package com.jessica.core.modules.codex.chat

import com.jessica.core.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * Best-effort, content-free observability. A telemetry failure NEVER blocks
 * a relay turn, causes a retry or triggers an unconfirmed send.
 */
object PhoneChatTelemetry {
    suspend fun send(session: String, phase: String, turn: Int, elapsedSeconds: Int) {
        withContext(Dispatchers.IO) {
            val connection = URL("https://jessicacore.onrender.com/api/codex/relay/phase")
                .openConnection() as HttpURLConnection
            try {
                connection.requestMethod = "POST"
                connection.connectTimeout = 3000
                connection.readTimeout = 3000
                connection.doOutput = true
                connection.setRequestProperty("X-Jessica-Token", BuildConfig.JESSICA_APP_TOKEN)
                connection.setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                val payload = JSONObject()
                    .put("session", session)
                    .put("phase", phase)
                    .put("turn", turn)
                    .put("elapsedSeconds", elapsedSeconds)
                connection.outputStream.use { it.write(payload.toString().toByteArray(Charsets.UTF_8)) }
                connection.responseCode
            } finally {
                connection.disconnect()
            }
        }
    }
}

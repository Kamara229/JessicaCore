package com.jessica.core.modules.codex.chat

import android.content.Context
import com.jessica.core.modules.codex.CodexClient
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONObject
import java.io.File
import java.util.UUID

data class PhoneChatStatus(val active: Boolean = false, val text: String = "Не запущено", val url: String = "")
class PhoneChatLaunch(val id: String, val ownerKey: String, var prompt: String, val chatUrl: String, val diagnostic: Boolean = false)

/** Only the active process holds the owner key. Never put it in a prompt, log, saved state or Intent. */
object PhoneChatRuntime {
    private val mutableStatus = MutableStateFlow(PhoneChatStatus())
    val status = mutableStatus.asStateFlow()
    var launch: PhoneChatLaunch? = null; private set
    var controller: ChatGptAccessibilityService? = null
    fun start(context: Context, prepared: JSONObject, ownerKey: String, url: String, diagnostic: Boolean = false) {
        check(controller != null) { "Сначала включи службу Джессики в специальных возможностях" }
        check(launch == null) { "Сессия уже запущена" }
        launch = PhoneChatLaunch(prepared.getString("id"), ownerKey, prepared.getString("prompt"), url, diagnostic)
        update(context, "Открываю обычный чат. Выбери режим «Чат», если он ещё не выбран.", true, "")
        controller?.begin()
    }
    fun checkConnection(context: Context, url: String) {
        val id = UUID.randomUUID().toString()
        val prompt = "Проверка связи с Джессикой в обычном чате. Не включай Работу/Codex и не меняй код. " +
            "Ответь одним блоком: первая отдельная строка JESSICA_REPLY_$id, затем JSON " +
            "{\"nonce\":\"$id\",\"status\":\"connected\"}, последняя отдельная строка JESSICA_END_$id."
        start(context, JSONObject().put("id", id).put("prompt", prompt), "", url, diagnostic = true)
    }
    fun update(context: Context, text: String, active: Boolean = true, url: String = status.value.url) {
        mutableStatus.value = PhoneChatStatus(active, text, url)
        context.getSharedPreferences("jessica_phone_chat", Context.MODE_PRIVATE).edit()
            .putString("last_status", text).putString("last_url", url).apply()
    }
    fun saveReply(context: Context, id: String, reply: String) {
        val folder = File(context.filesDir, "phone-chat").apply { mkdirs() }
        File(folder, "$id.json").writeText(reply)
        // Bound local retention; no screen dumps or unrelated conversations are saved.
        folder.listFiles()?.sortedByDescending { it.lastModified() }?.drop(10)?.forEach { it.delete() }
    }
    fun clear() { launch = null }
    fun stop() { controller?.stopByUser() }
    suspend fun prepare(ownerKey: String, goal: String, autoApply: Boolean): JSONObject =
        CodexClient().phone("prepare", ownerKey, JSONObject().put("goal", goal).put("autoApply", autoApply))
}

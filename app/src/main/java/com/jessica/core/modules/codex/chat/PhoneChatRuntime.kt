package com.jessica.core.modules.codex.chat

import android.content.Context
import com.jessica.core.modules.codex.CodexClient
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONObject
import java.io.File

data class PhoneChatStatus(val active: Boolean = false, val text: String = "Не запущено", val url: String = "")
class PhoneChatLaunch(val id: String, val ownerKey: String, var prompt: String, val chatUrl: String,
                      val connectivityOnly: Boolean = false) {
    // A repair keeps the backend session but must receive a fresh assistant answer.
    var replyId: String = id
    var previousReplyId: String = id
}

/** Only the active process holds the owner key. Never put it in a prompt, log, saved state or Intent. */
object PhoneChatRuntime {
    private val mutableStatus = MutableStateFlow(PhoneChatStatus())
    val status = mutableStatus.asStateFlow()
    var launch: PhoneChatLaunch? = null; private set
    @Volatile var controller: ChatGptAccessibilityService? = null; private set
    fun connected(service: ChatGptAccessibilityService) { controller = service }
    fun disconnected(service: ChatGptAccessibilityService) {
        if (controller === service) controller = null
    }
    private fun requireConnected(context: Context) {
        if (controller != null) return
        val diagnostics = ChatServiceDiagnostics.inspect(context)
        val problem = when (diagnostics.state) {
            ChatServiceState.ENABLED_NOT_CONNECTED ->
                "Разрешение уже включено, но служба Jessica не подключилась. " +
                "Проверь раздел «Диагностика службы»."
            ChatServiceState.NOT_REGISTERED ->
                "Android не видит службу Jessica. Обнови приложение поверх установленного."
            ChatServiceState.DISABLED ->
                "Включи службу Jessica Core в специальных возможностях Android."
            ChatServiceState.CONNECTED ->
                "Подключение службы изменилось. Проверь состояние ещё раз."
        }
        throw IllegalStateException(problem)
    }

    fun start(context: Context, prepared: JSONObject, ownerKey: String, url: String) {
        requireConnected(context)
        check(launch == null) { "Сессия уже запущена" }
        launch = PhoneChatLaunch(prepared.getString("id"), ownerKey, prepared.getString("prompt"), url)
        update(context, "Открываю обычный чат. Выбери режим «Чат», если он ещё не выбран.", true, "")
        controller?.begin()
    }
    /** One opted-in on-device send/receive check; NO owner key, GitHub or backend request. */
    fun startConnectivityProbe(context: Context, url: String) {
        requireConnected(context)
        check(launch == null) { "Сессия уже запущена" }
        val id = java.util.UUID.randomUUID().toString()
        launch = PhoneChatLaunch(id, "", ChatConnectivityProbe.prompt(id), url, connectivityOnly = true)
        update(context, "Пробная сессия без ключей: выбери обычный чат и подтверди его на панели.", true, "")
        controller?.begin()
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

package com.jessica.core.modules.codex.chat

import android.content.Context
import com.jessica.core.modules.codex.relay.RelayConversation
import com.jessica.core.modules.codex.relay.RelayProtocol
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.UUID

data class PhoneChatStatus(val active: Boolean = false, val text: String = "Не запущено")
class PhoneChatLaunch(context: Context) {
    val conversation = RelayConversation(context)
    // Random per-run diagnostics ID; no account, device or chat identifiers.
    val sessionId: String = UUID.randomUUID().toString()
    var replyId: String = UUID.randomUUID().toString()
    var previousReplyId: String = ""
    var message: String = RelayProtocol.introduction
    var prompt: String = RelayProtocol.wrap(message, replyId)
    var endAfterSend: Boolean = false
}

object PhoneChatRuntime {
    private val mutableStatus = MutableStateFlow(PhoneChatStatus())
    val status = mutableStatus.asStateFlow()
    var launch: PhoneChatLaunch? = null; private set
    @Volatile var controller: ChatGptAccessibilityService? = null; private set
    fun connected(service: ChatGptAccessibilityService) { controller = service }
    fun disconnected(service: ChatGptAccessibilityService) { if (controller === service) controller = null }
    fun start(context: Context) {
        check(controller != null) { ChatServiceDiagnostics.inspect(context).message }
        check(launch == null) { "Кодекс уже работает" }
        launch = PhoneChatLaunch(context.applicationContext)
        update(context, "Открой подготовленный чат ChatGPT и нажми «Чат выбран» на панели Джессики.")
        controller?.begin()
    }
    fun update(context: Context, text: String, active: Boolean = true) {
        mutableStatus.value = PhoneChatStatus(active, text)
        context.getSharedPreferences("jessica_phone_chat", Context.MODE_PRIVATE).edit().putString("last_status", text).apply()
    }
    fun clear() { launch = null }
    fun stop() { controller?.stopByUser() }
}

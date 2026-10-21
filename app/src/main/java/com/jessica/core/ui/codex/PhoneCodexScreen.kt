package com.jessica.core.ui.codex

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.codex.PhoneCodexHandoff
import com.jessica.core.ui.language.uiText
import com.jessica.core.ui.language.LocalAppLanguage
import com.jessica.core.ui.language.localeText

/**
 * On-device (not Render) launch and handoff to the installed ChatGPT app.
 * This is intentionally NOT an AccessibilityService / autonomous app scraper.
 */
@Composable
fun PhoneCodexScreen() {
    val context = LocalContext.current
    val language = LocalAppLanguage.current
    val prefs = remember { context.getSharedPreferences("jessica_codex_phone", Context.MODE_PRIVATE) }
    var goal by remember { mutableStateOf(prefs.getString("goal","language") ?: "language") }
    var url by remember { mutableStateOf(prefs.getString("chat_link","") ?: "") }
    var running by remember { mutableStateOf(false) }
    var notice by remember { mutableStateOf("") }
    val prompt = remember(goal) { PhoneCodexHandoff.buildPrompt(goal) }

    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(uiText("Codex на телефоне", "Codex on this phone"),
            style = MaterialTheme.typography.headlineSmall)
        Text(uiText(
            "Без Render. Открывает ChatGPT и подготавливает запрос прямо на телефоне. " +
                "Дальнейшая отправка и чтение ответа выполняются пользователем: " +
                "автоматического управления чужим приложением здесь нет.",
            "No Render. Opens ChatGPT and prepares a request on your device. " +
                "You send the message and read the answer; the app does not autonomously control ChatGPT."
        ),style=MaterialTheme.typography.bodySmall)
        Row(horizontalArrangement=Arrangement.spacedBy(7.dp)) {
            listOf(
                "language" to uiText("Язык","Language"),
                "context" to uiText("Контекст","Context"),
                "foundations" to uiText("Основа","Foundations")
            ).forEach { (value,label) ->
                FilterChip(
                    selected=goal==value,
                    enabled=!running,
                    onClick={ goal=value; prefs.edit().putString("goal",value).apply() },
                    label={ Text(label) }
                )
            }
        }
        OutlinedTextField(
            value=url,
            onValueChange={ url=it },
            enabled=!running,
            label={ Text(uiText("Ссылка на нужный приватный чат (необязательно)",
                                "Target private chat link (optional)")) },
            supportingText={ Text(uiText(
                "Можно вставить https://chatgpt.com/c/…; хранится только на телефоне. " +
                    "Если ссылка не указана, откроется обычное приложение ChatGPT.",
                "Paste https://chatgpt.com/c/…; stored on-device only. " +
                    "Without a link, ChatGPT opens normally."
            )) },
            maxLines=2,modifier=Modifier.fillMaxWidth()
        )
        if(!running) {
            Button(onClick={
                val normalized = PhoneCodexHandoff.normalizePrivateChatLink(url)
                if(url.isNotBlank() && normalized==null) {
                    notice=localeText(language,"Неверная ссылка на приватный чат ChatGPT", "Invalid private ChatGPT conversation link")
                    return@Button
                }
                prefs.edit().putString("chat_link",normalized ?: "").apply()
                val clip=context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                clip.setPrimaryClip(ClipData.newPlainText("Jessica Codex request",prompt))
                running=true
                val opened=PhoneCodexHandoff.openChatGPT(context,normalized)
                notice=if(opened)
                    localeText(language, "Запрос скопирован. В ChatGPT открой нужный чат, вставь текст и отправь.", "Prompt copied. Open the intended ChatGPT chat, paste the text and send it.")
                else localeText(language, "Запрос скопирован. ChatGPT открыть не удалось — открой его вручную.", "Prompt copied. ChatGPT could not be opened; please launch it manually.")
            },modifier=Modifier.fillMaxWidth()){
                Text(uiText("▶ Запустить на телефоне","▶ Start on device"))
            }
        }else{
            Text(uiText("Локальная сессия активна · запрос в буфере обмена",
                        "Local handoff active · prompt copied to clipboard"),
                color=MaterialTheme.colorScheme.primary)
            Row(horizontalArrangement=Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick={
                    val clip=context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                    clip.setPrimaryClip(ClipData.newPlainText("Jessica Codex request",prompt))
                    notice=localeText(language,"Запрос снова скопирован", "Prompt copied again")
                }){Text(uiText("Копировать запрос","Copy prompt"))}
                OutlinedButton(onClick={
                    PhoneCodexHandoff.openChatGPT(
                        context,PhoneCodexHandoff.normalizePrivateChatLink(url))
                }){Text(uiText("Открыть ChatGPT","Open ChatGPT"))}
            }
            Button(onClick={
                running=false
                notice=localeText(language,"Локальная сессия остановлена. Сообщения в ChatGPT не удаляются.", "Local handoff stopped. Messages in ChatGPT are unchanged.")
            },modifier=Modifier.fillMaxWidth()){
                Text(uiText("■ Остановить","■ Stop"))
            }
        }
        if(notice.isNotEmpty()) Text(notice,style=MaterialTheme.typography.bodySmall)
        Text(uiText("Текст подготовленного задания:","Prepared request:"),
            style=MaterialTheme.typography.titleSmall)
        Text(prompt,style=MaterialTheme.typography.bodySmall)
    }
}

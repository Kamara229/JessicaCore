package com.jessica.core.ui.codex

import android.content.Context
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.codex.PhoneCodexHandoff
import com.jessica.core.modules.codex.chat.PhoneChatRuntime
import com.jessica.core.modules.codex.chat.ChatServiceDiagnostics
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

@Composable
fun OrdinaryChatScreen() {
    val context = LocalContext.current
    val prefs = remember { context.getSharedPreferences("jessica_phone_chat", Context.MODE_PRIVATE) }
    val status by PhoneChatRuntime.status.collectAsState()
    val scope = rememberCoroutineScope()
    var ownerKey by remember { mutableStateOf("") }
    var goal by remember { mutableStateOf("language") }
    var chatUrl by remember { mutableStateOf(prefs.getString("chat_url", "") ?: "") }
    var autoApply by remember { mutableStateOf(false) }
    var consent by remember { mutableStateOf(false) }
    var connection by remember { mutableStateOf(ChatServiceDiagnostics.inspect(context)) }
    val ready = connection.connected
    var diagnostics by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf("") }
    LaunchedEffect(Unit) {
        while (true) { connection = ChatServiceDiagnostics.inspect(context); delay(1000) }
    }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Сборка Jessica: v${com.jessica.core.BuildConfig.VERSION_NAME}", style = MaterialTheme.typography.labelSmall)
        Text("Обычный ChatGPT · экспериментальный режим", style = MaterialTheme.typography.titleLarge)
        Text("Оставь телефон разблокированным и подключённым к питанию. Джессика отправит задание в выбранный чат, получит ответ и передаст исправление на тесты. До трёх сообщений и одного исправления за запуск.")
        Text("Сначала выбери в ChatGPT именно «Чат», затем нажми на плавающей панели «Чат выбран — начать». Режим «Работа» включать не нужно. Совместимость интерфейса проверяется на твоём телефоне.", style = MaterialTheme.typography.bodySmall)
        if (!ready) OutlinedButton(onClick = {
            context.startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
        }, modifier = Modifier.fillMaxWidth()) { Text("Открыть настройки службы Jessica") }
        Text("Разрешение Android: ${if (connection.enabled) "включено" else "выключено"}", style = MaterialTheme.typography.labelLarge)
        Text(connection.message, style = MaterialTheme.typography.labelLarge)
        if (!ready && connection.enabled) {
            Text("Разрешение уже выдано. Нажми проверку ниже: при отсутствии подключения появится отчёт о запуске службы.", style = MaterialTheme.typography.bodySmall)
        }
        if (!status.active) {
            Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf("language" to "Язык", "context" to "Контекст", "foundations" to "Основа").forEach { (value, label) ->
                    FilterChip(selected = goal == value, onClick = { goal = value }, label = { Text(label) }, enabled = !busy)
                }
            }
            OutlinedTextField(value = chatUrl, onValueChange = { chatUrl = it }, enabled = !busy,
                label = { Text("Ссылка на личный чат — необязательно") }, modifier = Modifier.fillMaxWidth(),
                supportingText = { Text("https://chatgpt.com/c/… . Без ссылки выбери нужный обычный чат после открытия ChatGPT.") })
            OutlinedTextField(value = ownerKey, onValueChange = { ownerKey = it }, enabled = !busy,
                visualTransformation = PasswordVisualTransformation(), singleLine = true,
                label = { Text("Ключ владельца Джессики") }, modifier = Modifier.fillMaxWidth(),
                supportingText = { Text("Разрешает запись в твой GitHub. В ChatGPT не передаётся и на телефоне не сохраняется. Это не платный ключ OpenAI.") })
            Row {
                Checkbox(checked = autoApply, onCheckedChange = { autoApply = it }, enabled = !busy)
                Text("Применять исправление в main автоматически после независимых тестов. Обычные процессы обновления сервера и APK запустятся вслед за коммитом.", modifier = Modifier.padding(top = 10.dp))
            }
            Row {
                Checkbox(checked = consent, onCheckedChange = { consent = it }, enabled = !busy)
                Text("Разрешаю Джессике отправлять сообщения в выбранном обычном чате и читать ответы. При запуске исправлений дополнительно разрешаю передачу подготовленного кода на проверку в мой GitHub. Проверка связи ничего не меняет в репозитории.", modifier = Modifier.padding(top = 10.dp))
            }
            OutlinedButton(
                enabled = consent && !busy, modifier = Modifier.fillMaxWidth(),
                onClick = {
                    connection = ChatServiceDiagnostics.inspect(context)
                    if (!connection.connected) {
                        error = connection.message + " Сообщение в ChatGPT не отправлено."
                        diagnostics = ChatServiceDiagnostics.report(context)
                        return@OutlinedButton
                    }
                    val url = PhoneCodexHandoff.normalizePrivateChatLink(chatUrl)
                    if (chatUrl.isNotBlank() && url == null) {
                        error = "Нужна ссылка на личный чат вида https://chatgpt.com/c/…"
                    } else {
                        try {
                            prefs.edit().putString("chat_url", url ?: "").apply()
                            PhoneChatRuntime.startConnectivityProbe(context, url ?: "")
                            error = ""
                        } catch (e: Exception) {
                            error = "Проверка не запущена: ${e.message}"
                            diagnostics = ChatServiceDiagnostics.report(context)
                        }
                    }
                }
            ) { Text("🔎 Проверить связь с обычным чатом — без ключей") }
            if (!consent) Text("Для проверки поставь галочку разрешения на отправку тестового сообщения.", style = MaterialTheme.typography.bodySmall)
            Text("Одно тестовое сообщение. Не использует GitHub, не применяет исправлений и не требует платного API.", style = MaterialTheme.typography.bodySmall)
            Button(enabled = ready && consent && ownerKey.length >= 16 && !busy, modifier = Modifier.fillMaxWidth(), onClick = {
                val url = PhoneCodexHandoff.normalizePrivateChatLink(chatUrl)
                if (chatUrl.isNotBlank() && url == null) { error = "Нужна ссылка на личный чат вида https://chatgpt.com/c/…"; return@Button }
                busy = true; error = ""
                scope.launch {
                    try {
                        val prepared = PhoneChatRuntime.prepare(ownerKey, goal, autoApply)
                        prefs.edit().putString("chat_url", url ?: "").apply()
                        PhoneChatRuntime.start(context, prepared, ownerKey, url ?: "")
                        ownerKey = ""
                    } catch (e: Exception) {
                        error = when (e.message) {
                            "owner-key-required" -> "Неверный ключ владельца Джессики."
                            "github-owner-setup-required" -> "На сервере ещё не настроен доступ владельца к GitHub. Платный API OpenAI не нужен."
                            "too-many-active-sessions" -> "Есть незавершённые сессии. Дождись их завершения."
                            else -> "Запуск не выполнен: ${e.message}"
                        }
                    } finally { busy = false }
                }
            }) { Text(if (busy) "Подготавливаю исходники…" else "▶ Запустить через обычный чат") }
        } else {
            Button(onClick = { PhoneChatRuntime.stop() }, modifier = Modifier.fillMaxWidth()) { Text("■ Остановить") }
        }
        if (error.isNotBlank()) Text(error, color = MaterialTheme.colorScheme.error)
        OutlinedButton(onClick = { diagnostics = ChatServiceDiagnostics.report(context) }) {
            Text("Показать диагностику службы")
        }
        if (diagnostics.isNotBlank()) {
            Text(diagnostics, style = MaterialTheme.typography.bodySmall)
            TextButton(onClick = {
                context.getSystemService(ClipboardManager::class.java)
                    .setPrimaryClip(ClipData.newPlainText("Jessica: диагностика службы", diagnostics))
            }) { Text("Скопировать диагностику") }
        }
        Text(if (status.text != "Не запущено") status.text else prefs.getString("last_status", "Не запущено") ?: "Не запущено")
        val resultUrl = status.url.ifBlank { prefs.getString("last_url", "") ?: "" }
        if (resultUrl.startsWith("https://github.com/Kamara229/JessicaCore/pull/"))
            OutlinedButton(onClick = { context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(resultUrl))) }) { Text("Открыть исправление и результаты тестов") }
        Text("Кнопка «Стоп Джессика» также будет поверх ChatGPT. При перезапуске приложения сессия сама не возобновляется. Ответы хранятся в папке приложения; история других чатов не выгружается.", style = MaterialTheme.typography.bodySmall)
    }
}

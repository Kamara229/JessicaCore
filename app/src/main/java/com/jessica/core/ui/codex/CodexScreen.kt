package com.jessica.core.ui.codex

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.provider.Settings
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.jessica.core.BuildConfig
import com.jessica.core.modules.codex.chat.ChatServiceDiagnostics
import com.jessica.core.modules.codex.chat.PhoneChatRuntime
import kotlinx.coroutines.delay

@Composable
fun CodexScreen() {
    val context = LocalContext.current
    val status by PhoneChatRuntime.status.collectAsState()
    var consent by remember { mutableStateOf(false) }
    var service by remember { mutableStateOf(ChatServiceDiagnostics.inspect(context)) }
    var report by remember { mutableStateOf("") }
    var error by remember { mutableStateOf("") }
    LaunchedEffect(Unit) { while (true) { service = ChatServiceDiagnostics.inspect(context); delay(1000) } }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Text("Кодекс", style = MaterialTheme.typography.headlineMedium)
        Text("Джессика участвует в переписке с ChatGPT вместо тебя и проверяет результаты через свой чат.")
        Text("1. Подготовь в ChatGPT план улучшения backend или исследования CALIBRI и попроси его спросить: «Джессика, ты готова?»\n\n" +
            "2. Нажми «Запустить Кодекс», открой эту переписку и нажми «Чат выбран» на панели Джессики.\n\n" +
            "3. Джессика сообщит о готовности, прочитает ответы, выполнит предложенные тесты и вернёт фактические результаты.")
        Text("Код и эксперименты ведёт ChatGPT через свои подключения. CALIBRI — только изолированные research-ветки; main, Render, Supabase и APK не меняем. Джессика отвечает в выбранной переписке и при необходимости выполняет отдельные тесты.", style = MaterialTheme.typography.bodySmall)
        Text(service.message, style = MaterialTheme.typography.labelLarge)
        if (!service.connected) OutlinedButton(onClick = { context.startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)) }) {
            Text("Открыть настройки службы Jessica")
        }
        if (!status.active) {
            Row {
                Checkbox(checked = consent, onCheckedChange = { consent = it })
                Text("Разрешаю читать выбранную переписку ChatGPT, передавать её контекст Джессике для понимания, отправлять ответы и выполнять тесты в её чате.")
            }
            Button(enabled = consent && service.connected, onClick = {
                try { PhoneChatRuntime.start(context); error = "" } catch (e: Exception) { error = e.message ?: "Запуск не выполнен" }
            }, modifier = Modifier.fillMaxWidth()) { Text("Запустить Кодекс") }
        } else Button(onClick = { PhoneChatRuntime.stop() }, modifier = Modifier.fillMaxWidth()) { Text("■ Остановить Кодекс") }
        if (error.isNotBlank()) Text(error, color = MaterialTheme.colorScheme.error)
        Text(if (status.text != "Не запущено") status.text else context.getSharedPreferences("jessica_phone_chat", Context.MODE_PRIVATE).getString("last_status", "Не запущено") ?: "Не запущено")
        Text("Оставь экран разблокированным, ChatGPT открытым, а телефон подключённым к питанию. Панель Джессики запрашивает удержание экрана включённым, но на конкретном Android это ещё нужно проверить. Обычная автопрокрутка запрещена. Кнопка «Мини-прокрутка» на панели Кодекса — отдельная экспериментальная функция: раз в 180 секунд ТОЛЬКО при генерации ответа и проверке выбранного чата; по умолчанию выключена. Она не гарантирует защиту от сна и требует проверки на реальном телефоне. Сессия — до 8 часов или 60 сообщений, есть кнопка СТОП. После перезапуска сама не возобновляется.", style = MaterialTheme.typography.bodySmall)
        Text("Переписка и тесты сохраняются в списке чатов Джессики под названиями «Кодекс — …».", style = MaterialTheme.typography.bodySmall)
        TextButton(onClick = { report = ChatServiceDiagnostics.report(context) }) { Text("Диагностика службы · v${BuildConfig.VERSION_NAME}") }
        if (report.isNotBlank()) {
            Text(report, style = MaterialTheme.typography.bodySmall)
            TextButton(onClick = { context.getSystemService(ClipboardManager::class.java).setPrimaryClip(ClipData.newPlainText("Jessica", report)) }) { Text("Скопировать диагностику") }
        }
    }
}

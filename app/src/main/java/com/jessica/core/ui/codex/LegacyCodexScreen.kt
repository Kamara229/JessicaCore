package com.jessica.core.ui.codex

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.codex.CodexClient
import com.jessica.core.ui.language.uiText
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.json.JSONObject

private data class AgentLine(val role: String, val text: String, val at: String)

@Composable
fun LegacyCodexScreen() {
    val context = LocalContext.current
    val preferences = remember {
        context.applicationContext.getSharedPreferences("jessica_codex", 0)
    }
    val client = remember { CodexClient() }
    val scope = rememberCoroutineScope()
    var configured by remember { mutableStateOf<Boolean?>(null) }
    // Owner key lives only in volatile Compose state; never saved to disk or logs.
    var ownerKey by remember { mutableStateOf("") }
    var authenticated by remember { mutableStateOf(false) }
    var goal by remember { mutableStateOf("language") }
    var issue by remember { mutableIntStateOf(preferences.getInt("issue", 0)) }
    var state by remember { mutableStateOf("idle") }
    var error by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var messages by remember { mutableStateOf<List<AgentLine>>(emptyList()) }

    LaunchedEffect(Unit) {
        try { configured = client.configuration().optBoolean("ready") }
        catch (_: Exception) { configured = false }
    }

    LaunchedEffect(issue, ownerKey, authenticated) {
        if (issue <= 0 || !authenticated || ownerKey.length < 16) return@LaunchedEffect
        while (true) {
            try {
                val data = client.status(ownerKey, issue)
                state = data.optString("state", "unknown")
                val all = data.optJSONArray("conversation")
                messages = (0 until (all?.length() ?: 0)).mapNotNull { n ->
                    all?.optJSONObject(n)?.let { line ->
                        AgentLine(
                            role = line.optString("role", "System"),
                            text = line.optString("text", ""),
                            at = line.optString("at", "")
                        )
                    }
                }
                error = ""
                if (state == "closed") break
            } catch (ex: Exception) {
                error = ex.message ?: "status-unavailable"
                if (error == "owner-key-required") authenticated = false
            }
            delay(12_000)
        }
    }

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Text(
            uiText("Codex ↔ Jessica", "Codex ↔ Jessica"),
            style = MaterialTheme.typography.headlineSmall
        )
        Text(
            uiText(
                "Два ИИ-агента обсуждают улучшение публичного кода, проверяют изменения " +
                    "и создают отдельный Pull Request. Основная версия не меняется автоматически.",
                "Two AI agents review public source, discuss improvements, run tests, " +
                    "and create a pull request. Production is not modified automatically."
            ),
            style = MaterialTheme.typography.bodySmall
        )
        if (configured == false) {
            Text(
                uiText(
                    "Codex ещё не настроен: необходимы ключ владельца и GitHub на сервере, а также ключ OpenAI в GitHub Actions.",
                    "Codex needs an owner key and GitHub credential on the server, plus an OpenAI key in GitHub Actions."
                ),
                color = MaterialTheme.colorScheme.error
            )
        }
        OutlinedTextField(
            value = ownerKey,
            onValueChange = { ownerKey = it; authenticated = false },
            label = { Text(uiText("Ключ владельца Codex", "Codex owner key")) },
            placeholder = { Text(uiText("Не хранится в приложении", "Not saved to the app")) },
            visualTransformation = PasswordVisualTransformation(),
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )
        if (issue == 0 || state == "closed") {
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                listOf(
                    "language" to uiText("Язык", "Language"),
                    "context" to uiText("Контекст", "Context"),
                    "foundations" to uiText("Основа", "Foundations")
                ).forEach { (id, name) ->
                    FilterChip(
                        selected = goal == id,
                        onClick = { goal = id },
                        label = { Text(name) }
                    )
                }
            }
            Button(
                onClick = {
                    busy = true
                    error = ""
                    scope.launch {
                        try {
                            val result = client.start(ownerKey, goal)
                            issue = result.optInt("issue")
                            if (issue > 0) preferences.edit().putInt("issue", issue).apply()
                            authenticated = true
                            state = "open"
                            messages = emptyList()
                        } catch (ex: Exception) { error = ex.message ?: "start-failed" }
                        finally { busy = false }
                    }
                },
                enabled = configured == true && !busy && ownerKey.length >= 16,
                modifier = Modifier.fillMaxWidth()
            ) { Text(uiText("▶ Запустить диалог и улучшение", "▶ Start agent collaboration")) }
        } else {
            Text(
                uiText("Сессия #", "Session #") + issue + " · " + state,
                style = MaterialTheme.typography.titleMedium
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(
                    enabled = !busy && ownerKey.length >= 16,
                    onClick = { authenticated = true }
                ) { Text(uiText("Обновить", "Refresh")) }
                OutlinedButton(
                    enabled = !busy && ownerKey.length >= 16 && state != "closed",
                    onClick = {
                        busy = true
                        scope.launch {
                            try {
                                client.stop(ownerKey, issue)
                                state = "closed"
                                authenticated = true
                            } catch (ex: Exception) { error = ex.message ?: "stop-failed" }
                            finally { busy = false }
                        }
                    }
                ) { Text(uiText("■ Остановить", "■ Stop")) }
            }
        }
        if (error.isNotBlank()) {
            Text(uiText("Ошибка: ", "Error: ") + error,
                color = MaterialTheme.colorScheme.error)
        }
        HorizontalDivider()
        Text(uiText("Диалог агентов", "Agent conversation"),
            style = MaterialTheme.typography.titleMedium)
        LazyColumn(
            modifier = Modifier.fillMaxWidth().weight(1f),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(messages) { item ->
                Column(modifier = Modifier.fillMaxWidth()) {
                    Text(item.role + " · " + item.at.take(19),
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.primary)
                    Text(item.text, style = MaterialTheme.typography.bodyMedium)
                }
                HorizontalDivider()
            }
        }
    }
}

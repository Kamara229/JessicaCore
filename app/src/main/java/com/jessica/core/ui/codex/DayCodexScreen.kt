package com.jessica.core.ui.codex

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

/**
 * One owner-authenticated daytime run continues on GitHub Actions without a
 * model API or requiring the Android app to stay in the foreground.
 * Only the latest issue number and goal are stored locally, never the owner key.
 */
@Composable
fun DayCodexScreen() {
    val context=LocalContext.current
    val prefs=remember { context.applicationContext.getSharedPreferences("jessica_day",Context.MODE_PRIVATE) }
    val client=remember { CodexClient() }
    val scope=rememberCoroutineScope()
    var ownerKey by remember { mutableStateOf("") }
    var configured by remember { mutableStateOf<Boolean?>(null) }
    var goal by remember { mutableStateOf(prefs.getString("goal","auto") ?: "auto") }
    var issue by remember { mutableIntStateOf(prefs.getInt("issue",0)) }
    var status by remember { mutableStateOf("idle") }
    var messages by remember { mutableStateOf<List<Pair<String,String>>>(emptyList()) }
    var busy by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf("") }

    LaunchedEffect(Unit) {
        try { configured=client.configuration().optBoolean("apiFreeDayReady",false) }
        catch (e:Exception) { configured=false; error=e.message ?: "configuration-unavailable" }
    }
    LaunchedEffect(issue,ownerKey) {
        if(issue<=0 || ownerKey.length<16) return@LaunchedEffect
        while(true) {
            try {
                val result=client.status(ownerKey,issue)
                status=result.optString("state","unknown")
                val transcript=result.optJSONArray("conversation")
                messages=(0 until (transcript?.length() ?: 0)).mapNotNull { index ->
                    transcript?.optJSONObject(index)?.let { entry ->
                        entry.optString("role","System") to entry.optString("text","")
                    }
                }
                error=""
                if(status=="closed")break
            } catch(e:Exception) {
                error=e.message ?: "day-status-unavailable"
                if(error=="owner-key-required")break
            }
            delay(12_000L)
        }
    }

    Column(
        modifier=Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement=Arrangement.spacedBy(12.dp)
    ) {
        Text(uiText("☀ Самопроверка Джессики днём","☀ Jessica daytime self-audit"),
            style=MaterialTheme.typography.headlineSmall)
        Text(uiText(
            "Без платного ИИ API. Запускает GitHub Actions: проверяет исходники, выполняет " +
                "регрессионные тесты и при необходимости создаёт небольшой проверенный PR. " +
                "Работа продолжается после закрытия приложения. Код не сливается в main автоматически.",
            "No paid AI API. Runs GitHub Actions: checks source code, executes regression tests, " +
                "and may open one reviewable PR. Work continues after you close the app. " +
                "Code is never automatically merged into main."
        ),style=MaterialTheme.typography.bodyMedium)
        Text(uiText(
            "Это не запуск модели Codex: без API невозможно автоматически генерировать " +
                "произвольные исправления. Сейчас выполняются безопасные детерминированные улучшения тестов.",
            "This does not run a Codex model: arbitrary AI-generated code changes require a model. " +
                "This mode performs deterministic regression improvements only."
        ),style=MaterialTheme.typography.bodySmall)
        if(configured==false) Text(uiText(
            "Для запуска один раз настрой в Render ключ владельца и GitHub-токен. OPENAI_API_KEY не нужен.",
            "One-time Render setup: owner secret and scoped GitHub token. OPENAI_API_KEY is not needed."
        ),color=MaterialTheme.colorScheme.error)
        OutlinedTextField(
            value=ownerKey,
            onValueChange={ownerKey=it},
            label={Text(uiText("Ключ владельца","Owner key"))},
            supportingText={Text(uiText("Хранится только до закрытия экрана","Kept in memory only"))},
            visualTransformation=PasswordVisualTransformation(),
            singleLine=true,
            modifier=Modifier.fillMaxWidth()
        )
        if(issue<=0 || status=="closed") {
            Row(horizontalArrangement=Arrangement.spacedBy(6.dp)) {
                listOf(
                    "auto" to uiText("Авто","Auto"),
                    "language" to uiText("Языки","Language"),
                    "context" to uiText("Контекст","Context"),
                    "foundations" to uiText("Основа","Foundations")
                ).forEach { (key,label) ->
                    FilterChip(selected=goal==key,onClick={
                        goal=key
                        prefs.edit().putString("goal",key).apply()
                    },label={Text(label)})
                }
            }
            Button(onClick={
                busy=true
                error=""
                scope.launch {
                    try {
                        val result=client.startDay(ownerKey,goal)
                        issue=result.optInt("issue",0)
                        if(issue>0) prefs.edit().putInt("issue",issue).apply()
                        status="open"
                        messages=emptyList()
                    }catch(e:Exception){error=e.message ?: "day-start-failed"}
                    finally{busy=false}
                }
            },enabled=configured==true && ownerKey.length>=16 && !busy,
                modifier=Modifier.fillMaxWidth()){
                Text(uiText("▶ Запустить дневную самопроверку","▶ Start daytime self-audit"))
            }
        }else{
            Text(uiText("Сессия #","Session #")+issue+" · "+status,
                style=MaterialTheme.typography.titleMedium)
            Row(horizontalArrangement=Arrangement.spacedBy(8.dp)) {
                OutlinedButton(enabled=!busy && ownerKey.length>=16,
                    onClick={
                        scope.launch {
                            try {
                                val result=client.status(ownerKey,issue)
                                status=result.optString("state","unknown")
                            }catch(e:Exception){error=e.message ?: "day-status-unavailable"}
                        }
                    }){Text(uiText("Обновить","Refresh"))}
                OutlinedButton(enabled=!busy && ownerKey.length>=16,
                    onClick={
                        busy=true
                        scope.launch {
                            try { client.stop(ownerKey,issue); status="closed" }
                            catch(e:Exception){error=e.message ?: "day-stop-failed"}
                            finally{busy=false}
                        }
                    }){Text(uiText("■ Остановить","■ Stop"))}
            }
        }
        if(issue>0) Text(uiText("Последняя сессия #","Latest session #")+issue,
            style=MaterialTheme.typography.labelMedium)
        if(error.isNotBlank()) Text(uiText("Ошибка: ","Error: ")+error,
            color=MaterialTheme.colorScheme.error)
        messages.forEach { (role,line) ->
            Text(role,style=MaterialTheme.typography.labelMedium,
                color=MaterialTheme.colorScheme.primary)
            Text(line,style=MaterialTheme.typography.bodySmall)
        }
    }
}

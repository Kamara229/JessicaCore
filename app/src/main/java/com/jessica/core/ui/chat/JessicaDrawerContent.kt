package com.jessica.core.ui.chat

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import com.jessica.core.ui.language.AppLanguage
import com.jessica.core.ui.language.uiText
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.Row
import androidx.compose.ui.Alignment
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.chat.ChatThread
import com.jessica.core.modules.chat.ChatSearchResult
import com.jessica.core.modules.chat.ChatTrashPolicy
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics

/** Navigation only: thread data and chat persistence stay in the ViewModel. */
@Composable
fun JessicaDrawerContent(
    threads: List<ChatThread>,
    activeThreadId: String?,
    searchQuery: String,
    searchResults: List<ChatSearchResult>,
    diagnosticError: String?,
    selectedLanguage: AppLanguage,
    onLanguageChange: (AppLanguage) -> Boolean,
    onSearch: (String) -> Unit,
    onDiagnosticSync: () -> Unit,
    busy: Boolean,
    onNewChat: () -> Unit,
    onThread: (String) -> Unit,
    onPreviewDeleted: (String) -> Unit,
    onRestore: (String) -> Unit,
    onTasks: () -> Unit,
    onReports: () -> Unit,
    onSettings: () -> Unit
) {
    var languageExpanded by remember { mutableStateOf(false) }
    ModalDrawerSheet {
        Column(modifier = Modifier.fillMaxSize()) {
            Text(
                "Jessica",
                color = MaterialTheme.colorScheme.primary,
                style = MaterialTheme.typography.headlineSmall,
                modifier = Modifier.padding(start = 26.dp, top = 30.dp, bottom = 12.dp)
            )
            LazyColumn(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                item {
                    NavigationDrawerItem(
                        label = { Text(uiText("＋  Новый чат", "＋  New chat")) },
                        selected = false,
                        onClick = { if (!busy) onNewChat() },
                        modifier = Modifier.padding(horizontal = 12.dp)
                    )
                }
                item {
                    OutlinedTextField(
                        value = searchQuery,
                        onValueChange = onSearch,
                        label = { Text(uiText("Поиск по чатам", "Search chats")) },
                        placeholder = { Text(uiText("Название, текст или «Удалённые»", "Title, messages or “Deleted”")) },
                        singleLine = true,
                        trailingIcon = {
                            if (searchQuery.isNotEmpty()) IconButton(onClick = { onSearch("") }) {
                                Text("×")
                            }
                        },
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 6.dp)
                    )
                }
                item {
                    androidx.compose.foundation.layout.Box(modifier = Modifier.padding(horizontal = 12.dp)) {
                        NavigationDrawerItem(
                            label = { Text(uiText("🌐  Язык", "🌐  Language") + ": " + selectedLanguage.label + " ▾") },
                            selected = false,
                            onClick = { languageExpanded = true }
                        )
                        DropdownMenu(expanded = languageExpanded, onDismissRequest = { languageExpanded = false }) {
                            AppLanguage.entries.forEach { option ->
                                DropdownMenuItem(
                                    text = { Text((if (selectedLanguage == option) "✓  " else "") + option.label) },
                                    onClick = {
                                        onLanguageChange(option)
                                        languageExpanded = false
                                    }
                                )
                            }
                        }
                    }
                }
                item {
                    HorizontalDivider(modifier = Modifier.padding(vertical = 10.dp))
                    Text(
                        uiText("История диалогов", "Chat history"),
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(start = 28.dp, bottom = 6.dp)
                    )
                }
                if (searchQuery.isNotBlank() && searchResults.isEmpty()) {
                    item { Text(uiText("Ничего не найдено", "No matches found"),
                        modifier = Modifier.padding(horizontal = 28.dp, vertical = 12.dp),
                        color = MaterialTheme.colorScheme.onSurfaceVariant) }
                }
                items(
                    if (searchQuery.isBlank()) threads.map { ChatSearchResult(it, "") }
                    else searchResults,
                    key = { it.thread.id }
                ) { match ->
                    val deleted = match.thread.deletedAt != null
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth().padding(end = 6.dp)
                    ) {
                        NavigationDrawerItem(
                            label = {
                                Column {
                                    Text(if (match.thread.title == "Новый чат") uiText("Новый чат", "New chat") else match.thread.title, maxLines = 1,
                                        overflow = TextOverflow.Ellipsis)
                                    if (deleted) {
                                        Text(
                                            uiText("Удалён · восстановление ещё ", "Deleted · recoverable for ") +
                                                ChatTrashPolicy.minutesLeft(match.thread.deletedAt) + uiText(" мин", " min"),
                                            color = MaterialTheme.colorScheme.error,
                                            style = MaterialTheme.typography.labelSmall
                                        )
                                    }
                                    if (match.snippet.isNotBlank()) {
                                        Text(match.snippet, maxLines = 2,
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            overflow = TextOverflow.Ellipsis)
                                    }
                                }
                            },
                            selected = !deleted && activeThreadId == match.thread.id,
                            onClick = {
                                if (!busy) {
                                    when (chatSearchSelection(match.thread)) {
                                        ChatSearchSelection.PREVIEW_DELETED ->
                                            onPreviewDeleted(match.thread.id)
                                        ChatSearchSelection.OPEN_ACTIVE ->
                                            onThread(match.thread.id)
                                    }
                                }
                            },
                            modifier = Modifier.weight(1f).padding(start = 12.dp)
                        )
                        if (deleted) {
                            IconButton(
                                enabled = !busy,
                                onClick = { onRestore(match.thread.id) },
                                modifier = Modifier.semantics {
                                    contentDescription = "Восстановить удалённый чат"
                                }
                            ) { Text("↶", style = MaterialTheme.typography.titleLarge) }
                        }
                    }
                }
                if (diagnosticError != null) {
                    item {
                        Row(verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.padding(horizontal = 24.dp)) {
                            Text(uiText("Тестовый экран: ", "Diagnostics: ") + diagnosticError,
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.error,
                                modifier = Modifier.weight(1f))
                            IconButton(onClick = onDiagnosticSync) { Text("↻") }
                        }
                    }
                }
                item {
                    HorizontalDivider(modifier = Modifier.padding(vertical = 10.dp))
                    Text(
                        "Jessica Core",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(start = 28.dp, bottom = 6.dp)
                    )
                }
                item {
                    NavigationDrawerItem(
                        label = { Text(uiText("Задачи", "Tasks")) },
                        selected = false,
                        onClick = onTasks,
                        modifier = Modifier.padding(horizontal = 12.dp)
                    )
                }
                item {
                    NavigationDrawerItem(
                        label = { Text(uiText("Отчёты", "Reports")) },
                        selected = false,
                        onClick = onReports,
                        modifier = Modifier.padding(horizontal = 12.dp)
                    )
                }
                item {
                    NavigationDrawerItem(
                        label = { Text(uiText("Настройки и обновление", "Settings and updates")) },
                        selected = false,
                        onClick = onSettings,
                        modifier = Modifier.padding(horizontal = 12.dp)
                    )
                }
            }
        }
    }
}

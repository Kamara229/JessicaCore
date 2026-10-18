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
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.chat.ChatThread

/** Navigation only: thread data and chat persistence stay in the ViewModel. */
@Composable
fun JessicaDrawerContent(
    threads: List<ChatThread>,
    activeThreadId: String?,
    busy: Boolean,
    onNewChat: () -> Unit,
    onThread: (String) -> Unit,
    onTasks: () -> Unit,
    onReports: () -> Unit,
    onSettings: () -> Unit
) {
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
                        label = { Text("＋  Новый чат") },
                        selected = false,
                        enabled = !busy,
                        onClick = onNewChat,
                        modifier = Modifier.padding(horizontal = 12.dp)
                    )
                }
                item {
                    HorizontalDivider(modifier = Modifier.padding(vertical = 10.dp))
                    Text(
                        "История диалогов",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(start = 28.dp, bottom = 6.dp)
                    )
                }
                items(threads, key = { it.id }) { thread ->
                    NavigationDrawerItem(
                        label = {
                            Text(
                                thread.title,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        },
                        selected = activeThreadId == thread.id,
                        enabled = !busy,
                        onClick = { onThread(thread.id) },
                        modifier = Modifier.padding(horizontal = 12.dp)
                    )
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
                        label = { Text("Задачи") },
                        selected = false,
                        onClick = onTasks,
                        modifier = Modifier.padding(horizontal = 12.dp)
                    )
                }
                item {
                    NavigationDrawerItem(
                        label = { Text("Отчёты") },
                        selected = false,
                        onClick = onReports,
                        modifier = Modifier.padding(horizontal = 12.dp)
                    )
                }
                item {
                    NavigationDrawerItem(
                        label = { Text("Настройки и обновление") },
                        selected = false,
                        onClick = onSettings,
                        modifier = Modifier.padding(horizontal = 12.dp)
                    )
                }
            }
        }
    }
}

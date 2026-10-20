package com.jessica.core.ui.chat

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.AssistChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.chat.ChatMessage

@Composable
fun ChatScreen(
    state: ChatState,
    onInputChange: (TextFieldValue) -> Unit,
    onSend: () -> Unit,
    onAttach: () -> Unit,
    onRemoveAttachment: (String) -> Unit,
    onRetry: (ChatMessage) -> Unit = {}
) {
    val clipboard = LocalClipboardManager.current
    val listState = rememberLazyListState()

    LaunchedEffect(state.activeThreadId, state.previewDeletedThreadId, state.messages.size, state.isRunning) {
        val size = state.messages.size + if (state.isRunning) 1 else 0
        if (size > 0) listState.scrollToItem(size - 1)
    }

    Column(modifier = Modifier.fillMaxSize().imePadding()) {
        if (state.previewDeletedThreadId != null) {
            Text(
                "Удалённый чат «" + (state.previewDeletedTitle ?: "Чат") +
                    "» · Только просмотр. Для восстановления нажми ↶ вверху справа.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.fillMaxWidth().padding(horizontal = 18.dp, vertical = 12.dp)
            )
        }
        if (state.messages.isEmpty() && !state.isRunning) {
            Box(
                modifier = Modifier.weight(1f).fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    modifier = Modifier.padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Text(
                        "Jessica",
                        color = MaterialTheme.colorScheme.primary,
                        style = MaterialTheme.typography.headlineLarge
                    )
                    Text("Чем займёмся?", style = MaterialTheme.typography.titleLarge)
                    Text(
                        "Расскажи о задаче. Я помогу разобраться и сохранить контекст.",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }
        } else {
            LazyColumn(
                modifier = Modifier.weight(1f).fillMaxWidth(),
                state = listState,
                contentPadding = androidx.compose.foundation.layout.PaddingValues(
                    horizontal = 16.dp, vertical = 12.dp
                )
            ) {
                items(state.messages) { message ->
                    MessageBubble(
                        message = message,
                        onCopy = { clipboard.setText(AnnotatedString(it)) },
                        onRetry = if (state.previewDeletedThreadId != null) { _ -> } else onRetry
                    )
                }
                if (state.isRunning) {
                    item { ThinkingIndicator() }
                }
            }
        }

        if (state.isLoading) {
            Text(
                "Загружаю историю…",
                modifier = Modifier.padding(horizontal = 20.dp),
                style = MaterialTheme.typography.bodySmall
            )
        }
        if (state.storageError != null) {
            Text(
                state.storageError,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(horizontal = 20.dp, vertical = 4.dp),
                style = MaterialTheme.typography.bodySmall
            )
        }
        if (state.pendingAttachments.isNotEmpty()) {
            Row(
                modifier = Modifier.fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
                    .padding(horizontal = 12.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                state.pendingAttachments.forEach { attachment ->
                    AssistChip(
                        onClick = { onRemoveAttachment(attachment.id) },
                        label = { Text("📎 ${attachment.name}  ×", maxLines = 1) },
                        enabled = !state.isRunning && !state.isImporting
                    )
                }
            }
        }
        if (state.isImporting) {
            Text(
                "Прикрепляю файл…",
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(horizontal = 20.dp)
            )
        }
        if (state.previewDeletedThreadId == null) ChatInput(
            value = state.input,
            onValueChange = onInputChange,
            onSend = onSend,
            onAttach = onAttach,
            enabled = !state.isRunning && !state.isLoading && !state.isImporting,
            modifier = Modifier.fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp)
        )
    }
}

package com.jessica.core.ui.chat

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.chat.ChatMessage
import com.jessica.core.ui.language.uiText
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun MessageBubble(
    message: ChatMessage,
    onCopy: (String) -> Unit = {},
    onRetry: (ChatMessage) -> Unit = {}
) {
    var actionsVisible by remember { mutableStateOf(false) }
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp),
        horizontalArrangement = if (message.isUser) Arrangement.End else Arrangement.Start
    ) {
        Box {
            Surface(
                modifier = Modifier.widthIn(max = 344.dp)
                    .clickable { actionsVisible = true },
                color = if (message.isUser)
                    MaterialTheme.colorScheme.primaryContainer
                else Color.Transparent,
                shape = RoundedCornerShape(
                    topStart = 22.dp,
                    topEnd = 22.dp,
                    bottomEnd = if (message.isUser) 6.dp else 22.dp,
                    bottomStart = if (message.isUser) 22.dp else 6.dp
                )
            ) {
                Column(modifier = Modifier.padding(
                    horizontal = if (message.isUser) 16.dp else 4.dp,
                    vertical = 11.dp
                )) {
                    if (!message.isUser) {
                        Text(
                            "Jessica",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.padding(bottom = 6.dp)
                        )
                    }
                    if (message.attachments.isNotEmpty()) {
                        message.attachments.forEach { attachment ->
                            Text(
                                "📎 ${attachment.name} (${(attachment.sizeBytes + 1023) / 1024} " + uiText("КБ", "KB") + ")",
                                style = MaterialTheme.typography.labelLarge,
                                color = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.padding(bottom = 6.dp)
                            )
                        }
                    }
                    SelectionContainer {
                        Text(message.text, style = MaterialTheme.typography.bodyLarge)
                    }
                    Text(
                        SimpleDateFormat("HH:mm", Locale.getDefault())
                            .format(Date(message.timestamp)),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 6.dp)
                    )
                }
            }
            MessageActions(
                message = message,
                expanded = actionsVisible,
                onDismiss = { actionsVisible = false },
                onCopy = onCopy,
                onRetry = onRetry
            )
        }
    }
}

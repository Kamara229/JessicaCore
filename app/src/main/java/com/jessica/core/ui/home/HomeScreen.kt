package com.jessica.core.ui.home

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.updater.UpdateState
import com.jessica.core.ui.status.JessicaStatusCard
import com.jessica.core.ui.update.UpdateCard
import com.jessica.core.ui.language.uiText
import com.jessica.core.BuildConfig

/** Settings and app health, now reachable from the navigation drawer. */
@Composable
fun HomeScreen(
    message: String,
    updateState: UpdateState,
    onUpdate: () -> Unit,
    onBetaUpdate: () -> Unit
) {
    LazyColumn(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item { JessicaStatusCard() }
        item { UpdateCard(state = updateState, onUpdateClick = onUpdate, onBetaUpdateClick = onBetaUpdate) }
        item {
            Text(
                uiText("О Jessica", "About Jessica"),
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(horizontal = 12.dp)
            )
            HorizontalDivider(modifier = Modifier.padding(top = 8.dp))
        }
        item {
            Text(
                uiText("Jessica Core v${BuildConfig.VERSION_NAME} запущена", "Jessica Core v${BuildConfig.VERSION_NAME} is running"),
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp)
            )
            Text(
                uiText("История диалогов хранится на телефоне автоматически. " +
                    "Обновление приложения сохраняет локальные данные.",
                    "Chat history is stored on your phone automatically. Updates preserve local data."),
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp)
            )
            Text(
                uiText("Experience и Skills продолжают работать независимо от локальной истории.",
                    "Experience and Skills work independently of local chat history."),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 12.dp)
            )
        }
    }
}

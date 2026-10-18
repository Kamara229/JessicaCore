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

/** Settings and app health, now reachable from the navigation drawer. */
@Composable
fun HomeScreen(
    message: String,
    updateState: UpdateState,
    onUpdate: () -> Unit
) {
    LazyColumn(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item { JessicaStatusCard() }
        item { UpdateCard(state = updateState, onUpdateClick = onUpdate) }
        item {
            Text(
                "О Jessica",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(horizontal = 12.dp)
            )
            HorizontalDivider(modifier = Modifier.padding(top = 8.dp))
        }
        item {
            Text(
                message,
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp)
            )
            Text(
                "История диалогов хранится на телефоне автоматически. " +
                    "Обновление приложения сохраняет локальные данные.",
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp)
            )
            Text(
                "Experience и Skills продолжают работать независимо от локальной истории.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 12.dp)
            )
        }
    }
}

package com.jessica.core.ui.home

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.updater.UpdateState
import com.jessica.core.ui.status.JessicaStatusCard
import com.jessica.core.ui.update.UpdateCard

@Composable
fun HomeScreen(
    message: String,
    updateState: UpdateState,
    onUpdate: () -> Unit,
    onChat: () -> Unit,
    onReports: () -> Unit,
    onTasks: () -> Unit
) {
    LazyColumn(
        modifier = Modifier.fillMaxSize().padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        item { JessicaStatusCard() }
        item { UpdateCard(state = updateState, onUpdateClick = onUpdate) }
        item { Text(message) }
        item { Text("История переписки сохраняется на телефоне автоматически.") }
        item {
            Button(onClick = onChat, modifier = Modifier.fillMaxWidth()) {
                Text("Чат Jessica")
            }
        }
        item {
            Button(onClick = onReports, modifier = Modifier.fillMaxWidth()) {
                Text("Отчёты")
            }
        }
        item {
            Button(onClick = onTasks, modifier = Modifier.fillMaxWidth()) {
                Text("Задачи")
            }
        }
    }
}

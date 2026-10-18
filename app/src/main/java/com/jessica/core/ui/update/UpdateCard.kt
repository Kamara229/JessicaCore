package com.jessica.core.ui.update

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.jessica.core.BuildConfig
import com.jessica.core.modules.updater.UpdateState

@Composable
fun UpdateCard(
    state: UpdateState,
    onUpdateClick: () -> Unit
) {
    val busy = state is UpdateState.Checking ||
        state is UpdateState.Downloading ||
        state is UpdateState.Verifying ||
        state is UpdateState.WaitingPermission ||
        state is UpdateState.Installing

    Card(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text("Обновление Jessica", style = MaterialTheme.typography.titleMedium)
            val message = when (state) {
                UpdateState.Idle -> "Установлена v${BuildConfig.VERSION_NAME}. Проверить наличие новой версии"
                UpdateState.Checking -> "Этап 1/4 · Проверка обновлений на GitHub…"
                is UpdateState.UpToDate -> "Установлена актуальная версия: ${state.version}"
                is UpdateState.Downloading -> "Этап 2/4 · Загрузка обновления: ${state.progress}%"
                UpdateState.Verifying -> "Этап 3/4 · Проверка файла и контрольной суммы SHA-256…"
                is UpdateState.ReadyToInstall ->
                    "Этап 4/4 · Версия ${state.version} готова. Открываю установщик Android…"
                is UpdateState.WaitingPermission ->
                    "Этап 4/4 · Для версии ${state.version} требуется разрешение Android."
                is UpdateState.Installing ->
                    "Этап 4/4 · Установка версии ${state.version}. Подтвердите замену в Android."
                is UpdateState.Installed ->
                    "✅ Все 4 этапа завершены. Установлена версия ${state.version}"
                is UpdateState.Error -> "Ошибка: ${state.message}"
            }
            Text(message, style = MaterialTheme.typography.bodyMedium)
            if (state is UpdateState.Downloading) {
                if (state.totalBytes > 0) {
                    LinearProgressIndicator(
                        progress = { state.progress / 100f },
                        modifier = Modifier.fillMaxWidth()
                    )
                    Text(
                        "${formatMegabytes(state.downloadedBytes)} / ${formatMegabytes(state.totalBytes)} МБ",
                        style = MaterialTheme.typography.bodySmall
                    )
                } else {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    Text(
                        "Загружено: ${formatMegabytes(state.downloadedBytes)} МБ",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
            if (state is UpdateState.Checking || state is UpdateState.Verifying) {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }
            val buttonText = when (state) {
                is UpdateState.Error -> "Повторить обновление"
                is UpdateState.UpToDate, is UpdateState.Installed -> "Проверить снова"
                else -> "🔄 Обновить Jessica"
            }
            Button(
                onClick = onUpdateClick,
                enabled = !busy && state !is UpdateState.ReadyToInstall,
                modifier = Modifier.fillMaxWidth().padding(top = 8.dp)
            ) {
                Text(buttonText)
            }
        }
    }
}

private fun formatMegabytes(bytes: Long): String =
    String.format(java.util.Locale.getDefault(), "%.1f", bytes / (1024.0 * 1024.0))

package com.jessica.core.ui.update

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.jessica.core.BuildConfig
import com.jessica.core.modules.updater.UpdateState
import com.jessica.core.ui.language.uiText

@Composable
fun UpdateCard(
    state: UpdateState,
    onUpdateClick: () -> Unit,
    onBetaUpdateClick: () -> Unit
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
            Text(uiText("Обновление Jessica", "Jessica update"), style = MaterialTheme.typography.titleMedium)
            val message = when (state) {
                UpdateState.Idle -> uiText("Установлена v${BuildConfig.VERSION_NAME}. Проверить наличие новой версии", "Installed v${BuildConfig.VERSION_NAME}. Check for updates")
                UpdateState.Checking -> uiText("Этап 1/4 · Проверка обновлений на GitHub…", "Step 1/4 · Checking GitHub for updates…")
                is UpdateState.UpToDate -> uiText("Установлена актуальная версия: ${state.version}", "You have the latest version: ${state.version}")
                is UpdateState.Downloading -> uiText("Этап 2/4 · Загрузка обновления: ${state.progress}%", "Step 2/4 · Downloading update: ${state.progress}%")
                UpdateState.Verifying -> uiText("Этап 3/4 · Проверка файла и контрольной суммы SHA-256…", "Step 3/4 · Verifying file and SHA-256 checksum…")
                is UpdateState.ReadyToInstall ->
                    uiText("Этап 4/4 · Версия ${state.version} готова. Открываю установщик Android…", "Step 4/4 · Version ${state.version} ready. Opening Android installer…")
                is UpdateState.WaitingPermission ->
                    uiText("Этап 4/4 · Для версии ${state.version} требуется разрешение Android.", "Step 4/4 · Android permission required to install ${state.version}.")
                is UpdateState.Installing ->
                    uiText("Этап 4/4 · Установка версии ${state.version}. Подтвердите замену в Android.", "Step 4/4 · Installing ${state.version}. Confirm replacement in Android.")
                is UpdateState.Installed ->
                    uiText("✅ Все 4 этапа завершены. Установлена версия ${state.version}", "✅ All four steps completed. Installed ${state.version}")
                is UpdateState.Error -> uiText("Ошибка: ", "Error: ") + state.message
            }
            Text(message, style = MaterialTheme.typography.bodyMedium)
            if (state is UpdateState.Downloading) {
                if (state.totalBytes > 0) {
                    LinearProgressIndicator(
                        progress = { state.progress / 100f },
                        modifier = Modifier.fillMaxWidth()
                    )
                    Text(
                        "${formatMegabytes(state.downloadedBytes)} / ${formatMegabytes(state.totalBytes)}" + uiText(" МБ", " MB"),
                        style = MaterialTheme.typography.bodySmall
                    )
                } else {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    Text(
                        uiText("Загружено: ", "Downloaded: ") + formatMegabytes(state.downloadedBytes) + uiText(" МБ", " MB"),
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
            if (state is UpdateState.Checking || state is UpdateState.Verifying) {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }
            Button(
                onClick = onUpdateClick,
                enabled = !busy && state !is UpdateState.ReadyToInstall,
                modifier = Modifier.fillMaxWidth().padding(top = 8.dp)
            ) {
                Text(uiText("🔄 Обновить · стабильная версия", "🔄 Update · stable"))
            }
            OutlinedButton(
                onClick = onBetaUpdateClick,
                enabled = !busy && state !is UpdateState.ReadyToInstall,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(uiText("🧪 Обновить · тестовая Beta", "🧪 Update · test beta"))
            }
            Text(
                uiText("Beta — экспериментальная версия Кодекса и CALIBRI. " +
                    "Обновление устанавливается поверх Jessica без удаления чатов. " +
                    "Нажимайте Beta только если хотите испытывать новые функции.",
                    "Beta includes experimental Codex and CALIBRI features. " +
                    "Updates install over Jessica without deleting chats. " +
                    "Choose Beta only to test unfinished features."),
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}

private fun formatMegabytes(bytes: Long): String =
    String.format(java.util.Locale.getDefault(), "%.1f", bytes / (1024.0 * 1024.0))

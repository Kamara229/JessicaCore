package com.jessica.core.ui.chat

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue

@Composable
fun ChatRoute(viewModel: ChatViewModel) {
    val state by viewModel.state.collectAsState()
    val picker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenMultipleDocuments()
    ) { uris -> viewModel.attachFiles(uris) }
    ChatScreen(
        state = state,
        onInputChange = viewModel::updateInput,
        onSend = viewModel::sendMessage,
        onAttach = { picker.launch(arrayOf("*/*")) },
        onRemoveAttachment = viewModel::removeAttachment,
        onRetry = viewModel::retryMessage
    )
}

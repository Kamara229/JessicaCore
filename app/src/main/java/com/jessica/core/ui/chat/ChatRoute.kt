package com.jessica.core.ui.chat

import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue

@Composable
fun ChatRoute(viewModel: ChatViewModel) {
    val state by viewModel.state.collectAsState()
    ChatScreen(
        state = state,
        onInputChange = viewModel::updateInput,
        onSend = viewModel::sendMessage,
        onRetry = viewModel::retryMessage
    )
}

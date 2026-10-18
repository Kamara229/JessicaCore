package com.jessica.core.ui.chat

import androidx.compose.ui.text.input.TextFieldValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.jessica.core.modules.chat.ChatContextWindow
import com.jessica.core.modules.chat.ChatHistoryStore
import com.jessica.core.modules.chat.ChatMessage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class ChatState(
    val messages: List<ChatMessage> = emptyList(),
    val input: TextFieldValue = TextFieldValue(),
    val isRunning: Boolean = false,
    val isLoading: Boolean = true,
    val storageError: String? = null
)

class ChatViewModel(
    private val historyStore: ChatHistoryStore,
    private val messageExecutor: suspend (String, List<ChatMessage>) -> String
) : ViewModel() {

    private val _state = MutableStateFlow(ChatState())
    val state: StateFlow<ChatState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            try {
                val restored = withContext(Dispatchers.IO) { historyStore.load() }
                _state.value = _state.value.copy(messages = restored, isLoading = false)
            } catch (error: Exception) {
                _state.value = _state.value.copy(
                    isLoading = false,
                    storageError = "Не удалось прочитать историю: ${error.message}"
                )
            }
        }
    }

    fun updateInput(value: TextFieldValue) {
        _state.value = _state.value.copy(input = value)
    }

    fun sendMessage() {
        val current = _state.value
        val text = current.input.text.trim()
        if (text.isEmpty() || current.isRunning || current.isLoading) return
        val history = ChatContextWindow.select(current.messages)
        _state.value = current.copy(isRunning = true, storageError = null)
        submit(ChatMessage(text = text, isUser = true), history, clearInput = true)
    }

    fun retryMessage(message: ChatMessage) {
        val current = _state.value
        if (!message.isUser || message.text.isBlank() ||
            current.isRunning || current.isLoading) return
        val history = ChatContextWindow.select(current.messages)
        _state.value = current.copy(isRunning = true, storageError = null)
        submit(ChatMessage(text = message.text.trim(), isUser = true), history, clearInput = false)
    }

    private fun submit(
        userMessage: ChatMessage,
        previous: List<ChatMessage>,
        clearInput: Boolean
    ) {
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) { historyStore.append(userMessage) }
            } catch (error: Exception) {
                _state.value = _state.value.copy(
                    isRunning = false,
                    storageError = "Не удалось записать сообщение: ${error.message}"
                )
                return@launch
            }

            _state.value = _state.value.copy(
                messages = _state.value.messages + userMessage,
                input = if (clearInput) TextFieldValue() else _state.value.input
            )
            val result = try {
                messageExecutor(userMessage.text, previous)
            } catch (error: Exception) {
                "Не удалось выполнить запрос: ${error.message ?: "неизвестная ошибка"}"
            }
            val answer = ChatMessage(text = result, isUser = false)
            val writeError = try {
                withContext(Dispatchers.IO) { historyStore.append(answer) }
                null
            } catch (error: Exception) {
                "Ответ отображён, но не сохранён: ${error.message}"
            }
            _state.value = _state.value.copy(
                messages = _state.value.messages + answer,
                isRunning = false,
                storageError = writeError
            )
        }
    }
}

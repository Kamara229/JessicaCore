package com.jessica.core.ui.chat

import androidx.compose.ui.text.input.TextFieldValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.jessica.core.modules.chat.ChatContextWindow
import com.jessica.core.modules.chat.ChatMessage
import com.jessica.core.modules.chat.ChatThread
import com.jessica.core.modules.chat.ChatThreadStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class ChatState(
    val messages: List<ChatMessage> = emptyList(),
    val threads: List<ChatThread> = emptyList(),
    val activeThreadId: String? = null,
    val input: TextFieldValue = TextFieldValue(),
    val isRunning: Boolean = false,
    val isLoading: Boolean = true,
    val storageError: String? = null
)

class ChatViewModel(
    private val threadStore: ChatThreadStore,
    private val messageExecutor: suspend (String, List<ChatMessage>) -> String
) : ViewModel() {

    private val _state = MutableStateFlow(ChatState())
    val state: StateFlow<ChatState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            try {
                val restored = withContext(Dispatchers.IO) {
                    val threads = threadStore.list()
                    val preferred = threadStore.activeId()
                    val chosen = threads.firstOrNull { it.id == preferred } ?: threads.firstOrNull()
                    if (chosen != null) threadStore.setActive(chosen.id)
                    Triple(threads, chosen?.id, chosen?.let { threadStore.messages(it.id) } ?: emptyList())
                }
                _state.value = _state.value.copy(
                    threads = restored.first,
                    activeThreadId = restored.second,
                    messages = restored.third,
                    isLoading = false
                )
            } catch (error: Exception) {
                _state.value = _state.value.copy(
                    isLoading = false,
                    storageError = "Не удалось загрузить чаты: ${error.message}"
                )
            }
        }
    }

    fun updateInput(value: TextFieldValue) {
        _state.value = _state.value.copy(input = value)
    }

    fun newChat() {
        val previous = _state.value
        if (previous.isRunning || previous.isLoading) return
        _state.value = previous.copy(isLoading = true, storageError = null)
        viewModelScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val created = threadStore.create()
                    threadStore.setActive(created.id)
                    Pair(created.id, threadStore.list())
                }
                _state.value = _state.value.copy(
                    activeThreadId = result.first,
                    threads = result.second,
                    messages = emptyList(),
                    input = TextFieldValue(),
                    isLoading = false
                )
            } catch (error: Exception) {
                _state.value = previous.copy(
                    isLoading = false,
                    storageError = "Не удалось создать чат: ${error.message}"
                )
            }
        }
    }

    fun openChat(id: String) {
        val previous = _state.value
        if (previous.isRunning || previous.isLoading || id == previous.activeThreadId) return
        if (previous.threads.none { it.id == id }) return
        _state.value = previous.copy(isLoading = true, storageError = null)
        viewModelScope.launch {
            try {
                val messages = withContext(Dispatchers.IO) {
                    val history = threadStore.messages(id)
                    threadStore.setActive(id)
                    history
                }
                _state.value = _state.value.copy(
                    activeThreadId = id,
                    messages = messages,
                    input = TextFieldValue(),
                    isLoading = false
                )
            } catch (error: Exception) {
                _state.value = previous.copy(
                    isLoading = false,
                    storageError = "Не удалось открыть чат: ${error.message}"
                )
            }
        }
    }

    fun sendMessage() {
        val snapshot = _state.value
        val text = snapshot.input.text.trim()
        if (text.isBlank() || snapshot.isRunning || snapshot.isLoading) return
        _state.value = snapshot.copy(isRunning = true, storageError = null)
        submit(ChatMessage(text = text, isUser = true),
            ChatContextWindow.select(snapshot.messages), clearInput = true)
    }

    fun retryMessage(message: ChatMessage) {
        val snapshot = _state.value
        if (!message.isUser || message.text.isBlank() || snapshot.isRunning || snapshot.isLoading) return
        _state.value = snapshot.copy(isRunning = true, storageError = null)
        submit(ChatMessage(text = message.text.trim(), isUser = true),
            ChatContextWindow.select(snapshot.messages), clearInput = false)
    }

    private fun submit(
        userMessage: ChatMessage,
        previousMessages: List<ChatMessage>,
        clearInput: Boolean
    ) {
        viewModelScope.launch {
            val id: String
            val threads: List<ChatThread>
            try {
                val saved = withContext(Dispatchers.IO) {
                    val target = _state.value.activeThreadId ?: threadStore.create().id.also {
                        threadStore.setActive(it)
                    }
                    threadStore.append(target, userMessage)
                    Pair(target, threadStore.list())
                }
                id = saved.first
                threads = saved.second
            } catch (error: Exception) {
                _state.value = _state.value.copy(
                    isRunning = false,
                    storageError = "Сообщение не отправлено: ${error.message}"
                )
                return@launch
            }

            _state.value = _state.value.copy(
                activeThreadId = id,
                threads = threads,
                messages = _state.value.messages + userMessage,
                input = if (clearInput) TextFieldValue() else _state.value.input
            )

            val text = try {
                messageExecutor(userMessage.text, previousMessages)
            } catch (error: Exception) {
                "Не удалось выполнить запрос: ${error.message ?: "неизвестная ошибка"}"
            }
            val answer = ChatMessage(text = text, isUser = false)
            val stored = try {
                withContext(Dispatchers.IO) {
                    threadStore.append(id, answer)
                    threadStore.list()
                }
            } catch (error: Exception) {
                _state.value = _state.value.copy(
                    storageError = "Ответ отображён, но не сохранён: ${error.message}"
                )
                null
            }
            _state.value = _state.value.copy(
                messages = _state.value.messages + answer,
                threads = stored ?: _state.value.threads,
                isRunning = false
            )
        }
    }
}

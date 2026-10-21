package com.jessica.core.ui.chat

import android.net.Uri
import com.jessica.core.ui.language.AppLanguage
import com.jessica.core.ui.language.localeText
import androidx.compose.ui.text.input.TextFieldValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.jessica.core.modules.chat.ChatAttachment
import com.jessica.core.modules.chat.ChatAttachmentStore
import com.jessica.core.modules.chat.ChatContextWindow
import com.jessica.core.modules.chat.ChatMessage
import com.jessica.core.modules.chat.ChatThread
import com.jessica.core.modules.chat.ChatThreadStore
import com.jessica.core.modules.chat.ChatTitlePolicy
import com.jessica.core.modules.chat.ChatTitleClient
import com.jessica.core.modules.chat.ChatSearchResult
import com.jessica.core.modules.chat.ChatDebugSyncClient
import com.jessica.core.modules.chat.ChatTrashPolicy
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

data class ChatState(
    val messages: List<ChatMessage> = emptyList(),
    val threads: List<ChatThread> = emptyList(),
    val activeThreadId: String? = null,
    val searchQuery: String = "",
    val searchResults: List<ChatSearchResult> = emptyList(),
    val undoThreadId: String? = null,
    val previewDeletedThreadId: String? = null,
    val previewDeletedTitle: String? = null,
    val debugSyncError: String? = null,
    val input: TextFieldValue = TextFieldValue(),
    val pendingAttachments: List<ChatAttachment> = emptyList(),
    val isImporting: Boolean = false,
    val isRunning: Boolean = false,
    val isLoading: Boolean = true,
    val storageError: String? = null
)

class ChatViewModel(
    private val threadStore: ChatThreadStore,
    private val attachmentStore: ChatAttachmentStore,
    private val messageExecutor: suspend (String, List<ChatMessage>, List<ChatAttachment>) -> String,
    private val languageProvider: () -> AppLanguage = { AppLanguage.RUSSIAN }
) : ViewModel() {
    private val _state = MutableStateFlow(ChatState())
    val state: StateFlow<ChatState> = _state.asStateFlow()
    private val syncMutex = Mutex()
    private var searchJob: Job? = null
    private val titleClient = ChatTitleClient { languageProvider().code }
    private val syncClient = ChatDebugSyncClient()
    private fun tr(ru: String, en: String): String = localeText(languageProvider(), ru, en)

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
                    isLoading = false,
                    undoThreadId = withContext(Dispatchers.IO) { threadStore.deleted().firstOrNull()?.id }
                )
                scheduleDebugSync()
            } catch (error: Exception) {
                _state.value = _state.value.copy(
                    isLoading = false,
                    storageError = tr("Не удалось загрузить чаты: ", "Could not load chats: ") + error.message
                )
            }
        }
    }

    init {
        // Refresh UI expiry and purge if the app stays open beyond 60 minutes.
        viewModelScope.launch {
            while (true) {
                delay(15_000L)
                val current = _state.value
                if (current.undoThreadId == null && current.previewDeletedThreadId == null && current.searchQuery.isBlank()) continue
                val pending = withContext(Dispatchers.IO) { threadStore.deleted() }
                val freshUndo = pending.firstOrNull()?.id
                val expiredPreview = current.previewDeletedThreadId != null &&
                    pending.none { it.id == current.previewDeletedThreadId }
                if (freshUndo != current.undoThreadId || expiredPreview) {
                    val activeMessages = if (expiredPreview) withContext(Dispatchers.IO) {
                        current.activeThreadId?.let { threadStore.messages(it) } ?: emptyList()
                    } else null
                    _state.value = _state.value.copy(
                        undoThreadId = freshUndo,
                        previewDeletedThreadId = if (expiredPreview) null else current.previewDeletedThreadId,
                        previewDeletedTitle = if (expiredPreview) null else current.previewDeletedTitle,
                        messages = activeMessages ?: current.messages
                    )
                    scheduleDebugSync()
                }
                if (current.searchQuery.isNotBlank()) updateSearch(current.searchQuery)
            }
        }
    }

    fun updateSearch(query: String) {
        _state.value = _state.value.copy(searchQuery = query)
        searchJob?.cancel()
        if (query.isBlank()) {
            _state.value = _state.value.copy(searchResults = emptyList())
            return
        }
        searchJob = viewModelScope.launch {
            delay(180)
            val results = withContext(Dispatchers.IO) { threadStore.search(query) }
            if (_state.value.searchQuery == query)
                _state.value = _state.value.copy(searchResults = results)
        }
    }

    private fun scheduleDebugSync() {
        viewModelScope.launch {
            syncMutex.withLock {
                val error = syncClient.sync(threadStore)
                _state.value = _state.value.copy(debugSyncError = error)
            }
        }
    }

    fun refreshChatDiagnostics() = scheduleDebugSync()

    fun updateInput(value: TextFieldValue) {
        if (_state.value.previewDeletedThreadId != null) return
        _state.value = _state.value.copy(input = value)
    }

    fun attachFiles(uris: List<Uri>) {
        val current = _state.value
        if (current.isRunning || current.isLoading || current.isImporting || current.previewDeletedThreadId != null) return
        if (uris.isEmpty()) return
        _state.value = current.copy(isImporting = true, storageError = null)
        viewModelScope.launch {
            val list = current.pendingAttachments.toMutableList()
            var problem: String? = null
            for (uri in uris) {
                if (list.size >= 3) {
                    problem = tr("Можно приложить не более 3 файлов к одному сообщению", "You can attach up to three files per message")
                    break
                }
                try {
                    list.add(attachmentStore.importUri(uri))
                } catch (error: Exception) {
                    problem = error.message ?: tr("Не удалось добавить файл", "Could not attach file")
                    break
                }
            }
            _state.value = _state.value.copy(
                pendingAttachments = list,
                isImporting = false,
                storageError = problem
            )
        }
    }

    fun removeAttachment(id: String) {
        val current = _state.value
        if (current.isRunning || current.isImporting || current.isLoading || current.previewDeletedThreadId != null) return
        val removed = current.pendingAttachments.find { it.id == id } ?: return
        _state.value = current.copy(
            pendingAttachments = current.pendingAttachments.filterNot { it.id == id },
            storageError = null
        )
        viewModelScope.launch(Dispatchers.IO) { attachmentStore.discard(removed) }
    }

    private fun discardPending() {
        val old = _state.value.pendingAttachments
        _state.value = _state.value.copy(pendingAttachments = emptyList())
        viewModelScope.launch(Dispatchers.IO) {
            old.forEach(attachmentStore::discard)
        }
    }

    fun newChat() {
        val previous = _state.value
        if (previous.isRunning || previous.isLoading || previous.isImporting) return
        discardPending()
        _state.value = _state.value.copy(isLoading = true, storageError = null)
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
                    isLoading = false,
                    previewDeletedThreadId = null,
                    previewDeletedTitle = null
                )
                scheduleDebugSync()
            } catch (error: Exception) {
                _state.value = _state.value.copy(
                    isLoading = false,
                    storageError = tr("Не удалось создать чат: ", "Could not create chat: ") + error.message
                )
            }
        }
    }

    fun openChat(id: String) {
        val current = _state.value
        if (current.isRunning || current.isLoading || current.isImporting || (id == current.activeThreadId && current.previewDeletedThreadId == null)) return
        if (current.threads.none { it.id == id }) return
        discardPending()
        _state.value = _state.value.copy(isLoading = true, storageError = null)
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
                    previewDeletedThreadId = null,
                    previewDeletedTitle = null,
                    isLoading = false
                )
            } catch (error: Exception) {
                _state.value = _state.value.copy(
                    isLoading = false,
                    storageError = tr("Не удалось открыть чат: ", "Could not open chat: ") + error.message
                )
            }
        }
    }


    /** Opening a deleted search result previews its history without undoing deletion. */
    fun previewDeletedChat(id: String) {
        val current = _state.value
        if (current.isRunning || current.isLoading || current.isImporting) return
        discardPending()
        _state.value = _state.value.copy(isLoading = true, storageError = null)
        viewModelScope.launch {
            try {
                val preview = withContext(Dispatchers.IO) {
                    val thread = threadStore.deleted().firstOrNull { it.id == id }
                        ?: throw IllegalStateException(tr("Время восстановления этого чата истекло", "This chat can no longer be restored"))
                    Pair(thread.title, threadStore.deletedMessages(id))
                }
                _state.value = _state.value.copy(
                    previewDeletedThreadId = id,
                    previewDeletedTitle = preview.first,
                    messages = preview.second,
                    input = TextFieldValue(),
                    isLoading = false
                )
            } catch (error: Exception) {
                _state.value = _state.value.copy(
                    isLoading = false,
                    storageError = tr("Не удалось открыть удалённый чат: ", "Could not open deleted chat: ") + error.message
                )
                updateSearch(_state.value.searchQuery)
            }
        }
    }

    fun deleteActiveChat() {
        val current = _state.value
        val id = current.activeThreadId ?: return
        if (current.isRunning || current.isLoading || current.isImporting || current.previewDeletedThreadId != null) return
        _state.value = current.copy(isLoading = true, storageError = null)
        viewModelScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    val remaining = threadStore.delete(id)
                    val active = threadStore.activeId()
                    Triple(remaining, active, active?.let { threadStore.messages(it) } ?: emptyList())
                }
                // The last deleted chat can be undone with the curved arrow.
                // Its content and attachment files are NOT erased until expiry.
                _state.value = _state.value.copy(
                    threads = result.first,
                    activeThreadId = result.second,
                    messages = result.third,
                    undoThreadId = id,
                    previewDeletedThreadId = null,
                    previewDeletedTitle = null,
                    input = TextFieldValue(),
                    isLoading = false
                )
                updateSearch(_state.value.searchQuery)
                scheduleDebugSync()
            } catch (error: Exception) {
                _state.value = _state.value.copy(
                    isLoading = false,
                    storageError = tr("Не удалось удалить чат: ", "Could not delete chat: ") + error.message
                )
            }
        }
    }

    fun undoDelete() {
        // Undo must apply only to a deleted chat that the user is viewing.
        _state.value.previewDeletedThreadId?.let { restoreChat(it) }
    }

    fun restoreChat(id: String) {
        val current = _state.value
        if (current.isRunning || current.isLoading || current.isImporting) return
        _state.value = current.copy(isLoading = true, storageError = null)
        viewModelScope.launch {
            try {
                val restored = withContext(Dispatchers.IO) {
                    threadStore.restore(id)
                    threadStore.setActive(id)
                    val threads = threadStore.list()
                    val messages = threadStore.messages(id)
                    val pending = threadStore.deleted()
                    Triple(threads, messages, pending.firstOrNull()?.id)
                }
                discardPending()
                _state.value = _state.value.copy(
                    threads = restored.first,
                    activeThreadId = id,
                    messages = restored.second,
                    undoThreadId = restored.third,
                    previewDeletedThreadId = null,
                    previewDeletedTitle = null,
                    input = TextFieldValue(),
                    isLoading = false
                )
                updateSearch(_state.value.searchQuery)
                scheduleDebugSync()
            } catch (error: Exception) {
                _state.value = _state.value.copy(
                    isLoading = false,
                    storageError = tr("Не удалось восстановить чат: ", "Could not restore chat: ") + error.message
                )
                updateSearch(_state.value.searchQuery)
            }
        }
    }

    fun sendMessage() {
        val current = _state.value
        val files = current.pendingAttachments
        val entered = current.input.text.trim()
        if ((entered.isEmpty() && files.isEmpty()) ||
            current.isRunning || current.isLoading || current.isImporting || current.previewDeletedThreadId != null) return
        val task = entered.ifBlank {
            if (languageProvider() == AppLanguage.ENGLISH)
                "Analyze the attached files and explain their contents."
            else "Проанализируй приложенные файлы и расскажи об их содержании."
        }
        val history = ChatContextWindow.select(current.messages)
        _state.value = current.copy(isRunning = true, storageError = null)
        submit(ChatMessage(text = task, isUser = true, attachments = files),
            history, clearInput = true)
    }

    fun retryMessage(message: ChatMessage) {
        val current = _state.value
        if (!message.isUser || message.text.isBlank() ||
            current.isRunning || current.isLoading || current.isImporting || current.previewDeletedThreadId != null) return
        val history = ChatContextWindow.select(current.messages)
        _state.value = current.copy(isRunning = true, storageError = null)
        submit(message.copy(timestamp = System.currentTimeMillis()),
            history, clearInput = false)
    }

    private fun submit(
        userMessage: ChatMessage,
        history: List<ChatMessage>,
        clearInput: Boolean
    ) {
        val firstMessage = _state.value.messages.none { it.isUser }
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
                    storageError = tr("Сообщение не отправлено: ", "Message not sent: ") + error.message
                )
                return@launch
            }

            _state.value = _state.value.copy(
                activeThreadId = id,
                threads = threads,
                messages = _state.value.messages + userMessage,
                pendingAttachments = emptyList(),
                input = if (clearInput) TextFieldValue() else _state.value.input
            )

            val result = try {
                messageExecutor(userMessage.text, history, userMessage.attachments)
            } catch (error: Exception) {
                tr("Не удалось выполнить запрос: ", "Could not complete the request: ") + (error.message ?: tr("неизвестная ошибка", "unknown error"))
            }
            val answer = ChatMessage(text = result, isUser = false)
            val stored = try {
                withContext(Dispatchers.IO) {
                    threadStore.append(id, answer)
                    threadStore.list()
                }
            } catch (error: Exception) {
                _state.value = _state.value.copy(
                    storageError = tr("Ответ отображён, но не сохранён: ", "The answer was shown but not saved: ") + error.message
                )
                null
            }
            _state.value = _state.value.copy(
                messages = _state.value.messages + answer,
                threads = stored ?: _state.value.threads,
                isRunning = false
            )
            scheduleDebugSync()
            if (firstMessage && ChatTitlePolicy.explicit(userMessage.text) == null) {
                viewModelScope.launch {
                    val title = titleClient.suggest(userMessage.text, result) ?: return@launch
                    val changed = withContext(Dispatchers.IO) {
                        if (threadStore.applyAutomaticTitle(id, title)) threadStore.list() else null
                    }
                    if (changed != null) {
                        _state.value = _state.value.copy(threads = changed)
                        updateSearch(_state.value.searchQuery)
                        scheduleDebugSync()
                    }
                }
            }
        }
    }
}

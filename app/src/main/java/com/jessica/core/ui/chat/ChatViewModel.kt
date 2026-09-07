package com.jessica.core.ui.chat

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.jessica.core.modules.chat.ChatMessage
import com.jessica.core.modules.chat.ChatMessageRole
import com.jessica.core.modules.chat.ChatState
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.util.UUID


/*
 * =========================================================
 * JESSICA CHAT VIEW MODEL
 * =========================================================
 *
 * Посредник между UI и AI Engine.
 *
 * UI ничего не знает:
 *
 * - откуда приходит ответ;
 * - сколько времени выполняется задача;
 * - какие модули используются.
 *
 * Это будет подключено к JessicaAIEngine позже.
 */


class ChatViewModel : ViewModel() {


    private val _state =
        MutableStateFlow(
            ChatState(
                messages =
                    listOf(
                        ChatMessage(
                            id =
                                UUID.randomUUID()
                                    .toString(),

                            role =
                                ChatMessageRole.SYSTEM,

                            text =
                                "Jessica Core v0.1 готова к работе"
                        )
                    )
            )
        )


    val state:
            StateFlow<ChatState> =
        _state



    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    fun updateInput(
        text: String
    ) {

        _state.value =
            _state.value.copy(
                inputText = text
            )

    }



    /*
     * =====================================================
     * SEND
     * =====================================================
     */


    fun sendMessage() {

        val text =
            _state.value.inputText
                .trim()


        if (
            text.isBlank() ||
            _state.value.isRunning
        ) {

            return

        }


        val userMessage =
            ChatMessage(

                id =
                    UUID.randomUUID()
                        .toString(),

                role =
                    ChatMessageRole.USER,

                text =
                    text

            )


        _state.value =
            _state.value.copy(

                messages =
                    _state.value.messages +
                            userMessage,

                inputText =
                    ""

            )


        executeTask(
            text
        )

    }



    /*
     * =====================================================
     * EXECUTION
     * =====================================================
     *
     * Временно имитация.
     *
     * Следующим шагом сюда подключим
     * настоящий JessicaAIEngine.
     */


    private fun executeTask(
        task: String
    ) {

        viewModelScope.launch {


            _state.value =
                _state.value.copy(
                    isRunning = true
                )


            delay(
                800
            )


            val answer =
                ChatMessage(

                    id =
                        UUID.randomUUID()
                            .toString(),

                    role =
                        ChatMessageRole.JESSICA,

                    text =
                        "Получила задачу:\n\n$task\n\nAI Engine будет подключён на следующем этапе."

                )


            _state.value =
                _state.value.copy(

                    messages =
                        _state.value.messages +
                                answer,

                    isRunning =
                        false

                )

        }

    }

}

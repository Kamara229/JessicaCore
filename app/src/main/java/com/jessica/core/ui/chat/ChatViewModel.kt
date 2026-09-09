package com.jessica.core.ui.chat


import androidx.compose.ui.text.input.TextFieldValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope

import com.jessica.core.modules.chat.ChatMessage

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch



/*
 * =========================================================
 * JESSICA CHAT STATE
 * =========================================================
 *
 * Состояние экрана чата.
 *
 * Хранит только данные UI:
 *
 * - сообщения
 * - текст ввода
 * - состояние выполнения запроса
 *
 * =========================================================
 */


data class ChatState(

    val messages: List<ChatMessage> = emptyList(),

    val input: TextFieldValue = TextFieldValue(),

    val isRunning: Boolean = false

)



/*
 * =========================================================
 * JESSICA CHAT VIEW MODEL
 * =========================================================
 *
 * Управляет состоянием чата.
 *
 * Отвечает за:
 *
 * - ввод текста
 * - отправку сообщения
 * - добавление сообщения пользователя
 * - получение ответа Jessica
 * - обработку ошибки
 * - повтор запроса
 *
 * Не содержит:
 *
 * - Planner
 * - Tool Registry
 * - сетевую реализацию
 * - AI Engine
 *
 * Выполнение запроса передаётся через messageExecutor.
 *
 * =========================================================
 */


class ChatViewModel(

    private val messageExecutor:
        suspend (String) -> String = {

            "Я получила сообщение. Ядро Jessica ещё не подключено к чату."

        }

) : ViewModel() {



    private val _state =

        MutableStateFlow(
            ChatState()
        )



    val state: StateFlow<ChatState> =

        _state.asStateFlow()



    /*
     * =====================================================
     * INPUT
     * =====================================================
     */


    fun updateInput(

        value: TextFieldValue

    ) {


        _state.value =

            _state.value.copy(

                input = value

            )


    }



    /*
     * =====================================================
     * SEND
     * =====================================================
     */


    fun sendMessage() {


        val text =

            _state.value
                .input
                .text
                .trim()



        if (text.isBlank()) {

            return

        }



        if (_state.value.isRunning) {

            return

        }



        val userMessage =

            ChatMessage(

                text = text,

                isUser = true

            )



        _state.value =

            _state.value.copy(

                messages =

                    _state.value.messages +
                        userMessage,

                input =
                    TextFieldValue(),

                isRunning =
                    true

            )



        executeMessage(
            text
        )


    }



    /*
     * =====================================================
     * EXECUTION
     * =====================================================
     */


    private fun executeMessage(

        text: String

    ) {


        viewModelScope.launch {


            try {


                val result =

                    messageExecutor(
                        text
                    )



                addJessicaMessage(

                    text =
                        result

                )


            } catch (

                error: Exception

            ) {


                addJessicaMessage(

                    text =
                        error.message
                            ?.takeIf {
                                it.isNotBlank()
                            }
                            ?.let {
                                "Не удалось выполнить запрос: $it"
                            }
                            ?: "Не удалось выполнить запрос."

                )


            }


        }


    }



    /*
     * =====================================================
     * JESSICA MESSAGE
     * =====================================================
     */


    private fun addJessicaMessage(

        text: String

    ) {


        val answer =

            ChatMessage(

                text = text,

                isUser = false

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



    /*
     * =====================================================
     * RETRY
     * =====================================================
     */


    fun retryMessage(

        message: ChatMessage

    ) {


        if (_state.value.isRunning) {

            return

        }



        val text =

            findPreviousUserMessage(
                message
            )



        if (text == null) {

            return

        }



        _state.value =

            _state.value.copy(

                isRunning = true

            )



        executeMessage(
            text
        )


    }



    /*
     * =====================================================
     * FIND SOURCE MESSAGE
     * =====================================================
     */


    private fun findPreviousUserMessage(

        message: ChatMessage

    ): String? {


        val messages =

            _state.value.messages



        val messageIndex =

            messages.indexOf(
                message
            )



        if (messageIndex <= 0) {

            return null

        }



        for (

            index in
                messageIndex - 1 downTo 0

        ) {


            val previousMessage =

                messages[index]



            if (previousMessage.isUser) {


                return previousMessage
                    .text
                    .takeIf {
                        it.isNotBlank()
                    }


            }


        }



        return null


    }


}

package com.jessica.core.ui.chat


import androidx.compose.ui.text.input.TextFieldValue
import androidx.lifecycle.ViewModel

import com.jessica.core.modules.chat.ChatMessage

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow



data class ChatState(

    val messages: List<ChatMessage> = emptyList(),

    val input: TextFieldValue = TextFieldValue(),

    val isRunning: Boolean = false

)



class ChatViewModel : ViewModel() {



    private val _state =

        MutableStateFlow(
            ChatState()
        )



    val state: StateFlow<ChatState> =

        _state.asStateFlow()




    fun updateInput(

        value: TextFieldValue

    ) {


        _state.value =

            _state.value.copy(

                input = value

            )


    }




    fun sendMessage() {


        val text =

            _state.value
                .input
                .text



        if (text.isBlank()) {

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


                isRunning = true

            )



        generateResponse()



    }




    private fun generateResponse() {



        /*
         * Пока тестовый ответ.
         *
         * Позже здесь будет:
         *
         * ChatViewModel
         *        |
         *        ↓
         * JessicaAIEngine
         *        |
         *        ↓
         * Render API
         *
         */



        val answer =

            ChatMessage(

                text =
                    "Я получила сообщение. Модуль ответа готов к подключению AI Engine.",


                isUser = false

            )



        _state.value =

            _state.value.copy(

                messages =

                    _state.value.messages +
                            answer,


                isRunning = false

            )


    }


}

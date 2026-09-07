package com.jessica.core.ui.chat


import androidx.compose.ui.text.input.TextFieldValue
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import com.jessica.core.modules.chat.ChatMessage



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
            _state.value.input.text


        if (text.isBlank())
            return



        val message =
            ChatMessage(

                text = text,

                isUser = true

            )



        _state.value =
            _state.value.copy(

                messages =
                    _state.value.messages +
                            message,


                input =
                    TextFieldValue()

            )


    }



}

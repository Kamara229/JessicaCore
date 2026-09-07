package com.jessica.core.ui.chat

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.chat.ChatMessage
import com.jessica.core.modules.chat.ChatState


/*
 * =========================================================
 * JESSICA CHAT SCREEN
 * =========================================================
 *
 * Главный экран общения с Jessica.
 *
 * Сейчас отвечает только за UI:
 *
 * - список сообщений;
 * - прокрутку;
 * - поле ввода;
 * - отображение состояния.
 *
 * Логика AI подключится позже.
 */


@Composable
fun ChatScreen(

    state: ChatState,

    onInputChange: (String) -> Unit,

    onSend: () -> Unit,

    modifier: Modifier = Modifier

) {


    val listState =
        rememberLazyListState()


    /*
     * Автоматически прокручиваем вниз
     * при появлении нового сообщения.
     */

    LaunchedEffect(
        state.messages.size
    ) {

        if (
            state.messages.isNotEmpty()
        ) {

            listState.animateScrollToItem(
                state.messages.size - 1
            )

        }

    }


    Column(

        modifier =
            modifier
                .fillMaxSize(),

    ) {


        /*
         * =================================================
         * MESSAGE LIST
         * =================================================
         */


        LazyColumn(

            modifier =
                Modifier
                    .weight(1f)
                    .padding(
                        horizontal = 12.dp
                    ),

            state =
                listState,

            verticalArrangement =
                Arrangement.spacedBy(
                    4.dp
                )

        ) {


            items(
                state.messages,
                key = {
                    it.id
                }
            ) { message ->


                MessageBubble(

                    message =
                        message

                )

            }


            if (
                state.messages.isEmpty()
            ) {

                item {

                    Text(

                        text =
                            "Jessica Core готова к работе",

                        style =
                            MaterialTheme
                                .typography
                                .bodyLarge,

                        modifier =
                            Modifier.padding(
                                20.dp
                            )

                    )

                }

            }


        }


        /*
         * =================================================
         * ERROR
         * =================================================
         */


        state.error?.let { error ->

            Text(

                text =
                    error,

                color =
                    MaterialTheme
                        .colorScheme
                        .error,

                modifier =
                    Modifier.padding(
                        12.dp
                    )

            )

        }


        /*
         * =================================================
         * INPUT
         * =================================================
         */


        ChatInput(

            value =
                androidx.compose.ui.text.input
                    .TextFieldValue(
                        state.inputText
                    ),

            onValueChange = {

                onInputChange(
                    it.text
                )

            },

            onSend =
                onSend,

            isRunning =
                state.isRunning

        )


    }

}

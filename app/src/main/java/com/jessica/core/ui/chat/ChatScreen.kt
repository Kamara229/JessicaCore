package com.jessica.core.ui.chat


import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding

import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState

import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect

import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp

import com.jessica.core.modules.chat.ChatMessage



/*
 * =========================================================
 * JESSICA CHAT SCREEN
 * =========================================================
 *
 * Центральный UI-слой чата.
 *
 * Соединяет:
 *
 * - MessageBubble
 * - MessageActions
 * - ThinkingIndicator
 * - ChatInput
 *
 * Отвечает:
 *
 * - за отображение сообщений
 * - копирование текста
 * - повтор запроса
 * - ввод сообщения
 * - автоматическую прокрутку чата
 *
 * Не содержит:
 *
 * - AI Engine
 * - сеть
 * - память
 * - Planner
 * - backend-логику
 *
 * =========================================================
 */


@Composable
fun ChatScreen(

    state: ChatState,

    onInputChange: (TextFieldValue) -> Unit,

    onSend: () -> Unit,

    onBack: () -> Unit,

    onRetry: (ChatMessage) -> Unit = {}

) {


    /*
     * =====================================================
     * CLIPBOARD
     * =====================================================
     */


    val clipboardManager =
        LocalClipboardManager.current



    /*
     * =====================================================
     * MESSAGE LIST STATE
     * =====================================================
     *
     * Хранит текущее положение LazyColumn.
     *
     * Используется для автоматической прокрутки
     * к последнему сообщению.
     *
     * =====================================================
     */


    val listState =
        rememberLazyListState()



    /*
     * =====================================================
     * AUTO SCROLL
     * =====================================================
     *
     * Срабатывает когда:
     *
     * - пользователь отправил сообщение;
     * - Jessica начала выполнение;
     * - Jessica закончила выполнение;
     * - появился новый ответ.
     *
     * =====================================================
     */


    LaunchedEffect(

        state.messages.size,

        state.isRunning

    ) {


        val totalItems =

            state.messages.size +

                if (state.isRunning) {

                    1

                } else {

                    0

                }



        if (totalItems > 0) {


            listState.animateScrollToItem(

                index =
                    totalItems - 1

            )


        }


    }



    /*
     * =====================================================
     * SCREEN
     * =====================================================
     */


    Column(

        modifier =
            Modifier
                .fillMaxSize()

    ) {



        /*
         * ================================
         * HEADER
         * ================================
         */


        Row(

            modifier =
                Modifier
                    .fillMaxWidth()
                    .padding(
                        horizontal = 12.dp,
                        vertical = 8.dp
                    )

        ) {



            Button(

                onClick =
                    onBack

            ) {


                Text(

                    text =
                        "Назад"

                )


            }



            Spacer(

                modifier =
                    Modifier.weight(1f)

            )



            Text(

                text =
                    "Jessica",

                style =
                    MaterialTheme
                        .typography
                        .titleMedium,

                modifier =
                    Modifier.padding(
                        top = 10.dp
                    )

            )


        }



        /*
         * ================================
         * MESSAGE LIST
         * ================================
         */


        LazyColumn(

            state =
                listState,

            modifier =
                Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .padding(
                        horizontal = 12.dp
                    )

        ) {



            /*
             * Сообщения диалога.
             */


            items(

                items =
                    state.messages

            ) { message ->



                MessageBubble(

                    message =
                        message,



                    onCopy = { text ->


                        clipboardManager
                            .setText(

                                AnnotatedString(
                                    text
                                )

                            )


                    },



                    onRetry = { retryMessage ->


                        onRetry(
                            retryMessage
                        )


                    }

                )


            }



            /*
             * Jessica выполняет запрос.
             */


            if (state.isRunning) {


                item {


                    ThinkingIndicator()


                }


            }


        }



        /*
         * ================================
         * INPUT
         * ================================
         */


        ChatInput(

            value =
                state.input,


            onValueChange =
                onInputChange,


            onSend =
                onSend,


            modifier =
                Modifier
                    .fillMaxWidth()
                    .padding(
                        12.dp
                    )

        )


    }


}

package com.jessica.core.ui.chat


import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding

import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items

import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text

import androidx.compose.runtime.Composable

import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp


/*
 * =========================================================
 * JESSICA CHAT SCREEN
 * =========================================================
 *
 * Ядро UI экрана чата.
 *
 * Соединяет:
 *
 * - MessageBubble
 * - ThinkingIndicator
 * - ChatInput
 *
 * Не содержит:
 *
 * - бизнес-логику;
 * - работу с сетью;
 * - AI-логику;
 * - хранение данных.
 *
 * =========================================================
 */


@Composable
fun ChatScreen(

    state: ChatState,

    onInputChange: (TextFieldValue) -> Unit,

    onSend: () -> Unit,

    onBack: () -> Unit

) {


    Column(

        modifier =
            Modifier
                .fillMaxSize()

    ) {


        /*
         * =================================================
         * HEADER
         * =================================================
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

                onClick = onBack

            ) {


                Text(
                    text = "Назад"
                )


            }


            /*
             * Раздвигает кнопку и заголовок.
             *
             * ВАЖНО:
             * отдельный import weight не нужен.
             */

            Spacer(

                modifier =
                    Modifier.weight(1f)

            )


            Text(

                text = "Jessica",

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
         * =================================================
         * MESSAGES
         * =================================================
         */

        LazyColumn(

            modifier =
                Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .padding(
                        horizontal = 12.dp
                    )

        ) {


            items(

                items =
                    state.messages

            ) { message ->


                MessageBubble(

                    message =
                        message

                )


            }


            /*
             * Индикатор работы Jessica.
             */

            if (state.isRunning) {


                item {


                    ThinkingIndicator()


                }


            }


        }


        /*
         * =================================================
         * INPUT
         * =================================================
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

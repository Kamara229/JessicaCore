package com.jessica.core.ui.chat


import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.weight

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
 * UI экрана чата.
 *
 * Только отображение.
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
         * Header
         */

        Row(

            modifier =
                Modifier
                    .fillMaxWidth()
                    .padding(12.dp)

        ) {


            Button(

                onClick = onBack

            ) {


                Text(
                    text = "Назад"
                )

            }



            Spacer(

                modifier =
                    Modifier.weight(1f)

            )



            Text(

                text = "Jessica",

                style =
                    MaterialTheme
                        .typography
                        .titleMedium

            )


        }





        /*
         * Messages
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

                items = state.messages

            ) { message ->


                MessageBubble(

                    message = message

                )


            }



            if (state.isRunning) {


                item {


                    ThinkingIndicator()


                }


            }


        }





        /*
         * Input
         */

        ChatInput(

            value = state.input,

            onValueChange = onInputChange,

            onSend = onSend

        )


    }


}

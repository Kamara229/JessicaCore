package com.jessica.core.ui.chat


import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp



/*
 * =========================================================
 * JESSICA CHAT SCREEN
 * =========================================================
 *
 * UI слой.
 *
 * Не содержит:
 * - ViewModel
 * - бизнес логику
 * - навигацию
 *
 * Только отображает состояние.
 *
 * =========================================================
 */


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(

    state: ChatState,

    onInputChange: (String) -> Unit,

    onSend: () -> Unit,

    onBack: () -> Unit

) {


    Scaffold(

        topBar = {

            TopAppBar(

                title = {

                    Text(
                        "Jessica Chat"
                    )

                },

                navigationIcon = {

                    Button(
                        onClick = onBack
                    ) {

                        Text(
                            "Назад"
                        )

                    }

                }

            )

        }


    ) { padding ->


        Column(

            modifier =
                Modifier
                    .padding(padding)
                    .fillMaxSize()

        ) {



            LazyColumn(

                modifier =
                    Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .padding(12.dp),

                verticalArrangement =
                    Arrangement.spacedBy(8.dp)

            ) {


                items(

                    state.messages

                ) { message ->


                    MessageBubble(

                        message = message

                    )


                }


            }



            Row(

                modifier =
                    Modifier
                        .fillMaxWidth()
                        .padding(12.dp),

                horizontalArrangement =
                    Arrangement.spacedBy(8.dp)

            ) {



                ChatInput(

                    value =
                        state.input,

                    onValueChange =
                        onInputChange,

                    modifier =
                        Modifier.weight(1f)

                )



                Button(

                    onClick = onSend

                ) {


                    Text(
                        "Отправить"
                    )


                }


            }


        }


    }


}

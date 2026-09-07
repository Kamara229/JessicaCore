package com.jessica.core.ui.chat


import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items

import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextField

import androidx.compose.runtime.Composable

import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

import androidx.compose.ui.text.input.TextFieldValue



@Composable
fun ChatScreen(

    state: ChatState,

    onInputChange: (
        TextFieldValue
    ) -> Unit,

    onSend: () -> Unit,

    onBack: () -> Unit

) {


    Column(

        modifier =
            Modifier
                .fillMaxSize()
                .padding(16.dp),

        verticalArrangement =
            Arrangement.SpaceBetween

    ) {



        Column {


            Text(

                text = "Jessica Chat",

                style =
                    MaterialTheme.typography.titleLarge

            )



            LazyColumn {


                items(

                    state.messages

                ) { message ->



                    MessageBubble(

                        message = message

                    )


                }


            }


        }



        Column {



            TextField(

                value =
                    state.input,


                onValueChange =
                    onInputChange,


                modifier =
                    Modifier
                        .padding(
                            bottom = 8.dp
                        )


            )



            Button(

                onClick =
                    onSend

            ) {


                Text(

                    text = "Отправить"

                )


            }


        }


    }

}

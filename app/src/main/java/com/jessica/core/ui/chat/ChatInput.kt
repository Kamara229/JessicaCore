package com.jessica.core.ui.chat


import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding

import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.material3.TextField

import androidx.compose.runtime.Composable

import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp



/*
 * =========================================================
 * JESSICA CHAT INPUT
 * =========================================================
 *
 * Поле ввода сообщения.
 *
 * Не содержит бизнес-логики.
 *
 * Отвечает только за UI:
 *
 * - ввод текста
 * - отображение кнопки отправки
 *
 * =========================================================
 */


@Composable
fun ChatInput(

    value: TextFieldValue,

    onValueChange: (
        TextFieldValue
    ) -> Unit,

    onSend: () -> Unit,

    modifier: Modifier = Modifier

) {


    Column(

        modifier =
            modifier
                .fillMaxWidth()

    ) {



        TextField(

            value = value,


            onValueChange = onValueChange,


            modifier =
                Modifier
                    .fillMaxWidth()
                    .padding(
                        bottom = 8.dp
                    ),


            placeholder = {

                Text(
                    "Введите сообщение"
                )

            }

        )



        Button(

            onClick = onSend,


            modifier =
                Modifier
                    .padding(
                        top = 8.dp
                    )

        ) {


            Text(
                "Отправить"
            )


        }


    }


}

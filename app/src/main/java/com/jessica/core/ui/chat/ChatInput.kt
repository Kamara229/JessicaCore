package com.jessica.core.ui.chat

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp


/*
 * =========================================================
 * JESSICA CHAT INPUT
 * =========================================================
 *
 * Нижняя панель ввода сообщения.
 *
 * Отвечает только за UI:
 *
 * - ввод текста;
 * - многострочное поле;
 * - кнопка отправки;
 * - блокировка отправки во время выполнения.
 *
 * Сам AI здесь НЕ вызывается.
 */


/*
 * =========================================================
 * CHAT INPUT
 * =========================================================
 */


@Composable
fun ChatInput(

    value: TextFieldValue,

    onValueChange: (TextFieldValue) -> Unit,

    onSend: () -> Unit,

    isRunning: Boolean,

    modifier: Modifier = Modifier

) {

    val canSend =
        value.text.isNotBlank() &&
        !isRunning


    Row(

        modifier =
            modifier
                .fillMaxWidth()
                .padding(
                    horizontal = 12.dp,
                    vertical = 10.dp
                ),

        verticalAlignment =
            Alignment.Bottom,

        horizontalArrangement =
            Arrangement.spacedBy(
                8.dp
            )

    ) {


        /*
         * =================================================
         * MESSAGE FIELD
         * =================================================
         */


        OutlinedTextField(

            value =
                value,

            onValueChange =
                onValueChange,

            modifier =
                Modifier
                    .weight(1f),

            placeholder = {

                Text(
                    "Сообщение Jessica..."
                )

            },

            enabled =
                !isRunning,

            minLines =
                1,

            maxLines =
                6

        )


        /*
         * =================================================
         * SEND
         * =================================================
         */


        Button(

            onClick =
                onSend,

            enabled =
                canSend,

            modifier =
                Modifier
                    .widthIn(
                        min = 56.dp
                    )

        ) {

            Text(
                if (isRunning) {
                    "..."
                } else {
                    "➤"
                }
            )

        }

    }

}

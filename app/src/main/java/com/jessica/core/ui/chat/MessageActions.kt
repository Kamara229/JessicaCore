package com.jessica.core.ui.chat


import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Text

import androidx.compose.runtime.Composable

import com.jessica.core.modules.chat.ChatMessage


/*
 * =========================================================
 * JESSICA MESSAGE ACTIONS
 * =========================================================
 *
 * Контекстное меню действий над сообщением.
 *
 * Само меню не решает, когда оно открывается.
 *
 * MessageBubble передаёт:
 *
 * expanded = true / false
 *
 * Доступные действия:
 *
 * - Копировать
 * - Повторить запрос
 *
 * =========================================================
 */


@Composable
fun MessageActions(

    message: ChatMessage,

    expanded: Boolean,

    onDismiss: () -> Unit,

    onCopy: (String) -> Unit,

    onRetry: (ChatMessage) -> Unit

) {


    DropdownMenu(

        expanded =
            expanded,

        onDismissRequest =
            onDismiss

    ) {


        /*
         * =================================================
         * COPY
         * =================================================
         */


        DropdownMenuItem(

            text = {

                Text(
                    text = "📋  Копировать"
                )

            },

            onClick = {


                onCopy(
                    message.text
                )


                onDismiss()


            }

        )



        /*
         * =================================================
         * RETRY
         * =================================================
         *
         * Повторяем именно запрос пользователя.
         *
         * Для ответа Jessica эту кнопку
         * пока не показываем.
         *
         * =================================================
         */


        if (message.isUser) {


            DropdownMenuItem(

                text = {

                    Text(
                        text = "🔄  Повторить запрос"
                    )

                },

                onClick = {


                    onRetry(
                        message
                    )


                    onDismiss()


                }

            )


        }


    }


}

package com.jessica.core.ui.chat


import androidx.compose.foundation.clickable

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn

import androidx.compose.foundation.text.selection.SelectionContainer

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue

import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

import com.jessica.core.modules.chat.ChatMessage

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale


/*
 * =========================================================
 * JESSICA MESSAGE BUBBLE
 * =========================================================
 *
 * Одно сообщение чата.
 *
 * Отвечает только за UI.
 *
 * Возможности:
 *
 * - сообщение пользователя справа;
 * - ответ Jessica слева;
 * - выделение текста;
 * - нажатие на сообщение открывает меню действий;
 * - копирование;
 * - повтор запроса.
 *
 * =========================================================
 */


@Composable
fun MessageBubble(

    message: ChatMessage,

    onCopy: (String) -> Unit = {},

    onRetry: (ChatMessage) -> Unit = {}

) {


    val isUser =
        message.isUser


    /*
     * Состояние контекстного меню
     */

    var actionsVisible by
        remember {

            mutableStateOf(
                false
            )

        }



    Column(

        modifier =
            Modifier
                .fillMaxWidth()
                .padding(
                    vertical = 4.dp
                )

    ) {


        Row(

            modifier =
                Modifier.fillMaxWidth(),

            horizontalArrangement =
                if (isUser) {

                    Arrangement.End

                } else {

                    Arrangement.Start

                }

        ) {


            /*
             * Box является якорем
             * для выпадающего меню.
             */

            Box {


                /*
                 * =================================================
                 * MESSAGE
                 * =================================================
                 */


                Surface(

                    modifier =
                        Modifier
                            .widthIn(
                                max = 340.dp
                            )
                            .clickable {

                                actionsVisible =
                                    true

                            },

                    shape =
                        MaterialTheme
                            .shapes
                            .large,

                    color =
                        if (isUser) {

                            MaterialTheme
                                .colorScheme
                                .primaryContainer

                        } else {

                            MaterialTheme
                                .colorScheme
                                .surfaceVariant

                        }

                ) {


                    Column(

                        modifier =
                            Modifier.padding(
                                14.dp
                            )

                    ) {


                        /*
                         * Имя ассистента.
                         */


                        if (!isUser) {


                            Text(

                                text =
                                    "Jessica",

                                style =
                                    MaterialTheme
                                        .typography
                                        .labelMedium,

                                modifier =
                                    Modifier.padding(
                                        bottom = 4.dp
                                    )

                            )


                        }



                        /*
                         * Текст сообщения.
                         *
                         * SelectionContainer оставляем,
                         * чтобы пользователь мог
                         * выделять отдельные части текста.
                         */


                        SelectionContainer {


                            Text(

                                text =
                                    message.text,

                                style =
                                    MaterialTheme
                                        .typography
                                        .bodyLarge

                            )


                        }



                        /*
                         * Время сообщения.
                         */


                        Text(

                            text =
                                formatTime(
                                    message.timestamp
                                ),

                            style =
                                MaterialTheme
                                    .typography
                                    .labelSmall,

                            color =
                                MaterialTheme
                                    .colorScheme
                                    .onSurfaceVariant,

                            modifier =
                                Modifier.padding(
                                    top = 4.dp
                                )

                        )


                    }


                }



                /*
                 * =================================================
                 * MESSAGE ACTION MENU
                 * =================================================
                 *
                 * В обычном состоянии не видно.
                 *
                 * Открывается после нажатия
                 * на сообщение.
                 *
                 * =================================================
                 */


                MessageActions(

                    message =
                        message,

                    expanded =
                        actionsVisible,

                    onDismiss = {

                        actionsVisible =
                            false

                    },

                    onCopy =
                        onCopy,

                    onRetry =
                        onRetry

                )


            }


        }


    }


}



/*
 * =========================================================
 * MESSAGE TIME
 * =========================================================
 */


private fun formatTime(

    timestamp: Long

): String {


    return SimpleDateFormat(

        "HH:mm",

        Locale.getDefault()

    ).format(

        Date(timestamp)

    )


}

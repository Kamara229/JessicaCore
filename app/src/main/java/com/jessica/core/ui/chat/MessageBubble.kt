package com.jessica.core.ui.chat


import androidx.compose.foundation.layout.Arrangement
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

import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

import com.jessica.core.modules.chat.ChatMessage

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale



@Composable
fun MessageBubble(

    message: ChatMessage

) {


    val isUser =
        message.isUser


    Row(

        modifier =
            Modifier
                .fillMaxWidth()
                .padding(
                    vertical = 4.dp
                ),

        horizontalArrangement =
            if (isUser) {

                Arrangement.End

            } else {

                Arrangement.Start

            }

    ) {


        Surface(

            modifier =
                Modifier
                    .widthIn(
                        max = 340.dp
                    ),

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
                    Modifier
                        .padding(
                            horizontal = 14.dp,
                            vertical = 10.dp
                        )

            ) {


                if (!isUser) {


                    Text(

                        text = "Jessica",

                        style =
                            MaterialTheme
                                .typography
                                .labelMedium,

                        color =
                            MaterialTheme
                                .colorScheme
                                .primary,

                        modifier =
                            Modifier.padding(
                                bottom = 4.dp
                            )

                    )


                }



                SelectionContainer {


                    Text(

                        text =
                            message.text,

                        style =
                            MaterialTheme
                                .typography
                                .bodyLarge,

                        color =
                            if (isUser) {

                                MaterialTheme
                                    .colorScheme
                                    .onPrimaryContainer

                            } else {

                                MaterialTheme
                                    .colorScheme
                                    .onSurfaceVariant

                            }

                    )


                }



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
                            top = 5.dp
                        )

                )


            }


        }


    }


}



private fun formatTime(

    timestamp: Long

): String {


    if (timestamp <= 0L) {

        return ""

    }


    return SimpleDateFormat(

        "HH:mm",

        Locale.getDefault()

    ).format(

        Date(timestamp)

    )


}

package com.jessica.core.ui.chat

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.chat.ChatMessage
import com.jessica.core.modules.chat.ChatMessageRole
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale


@Composable
fun MessageBubble(

    message: ChatMessage,

    onCopy: ((String) -> Unit)? = null,

    onRetry: ((ChatMessage) -> Unit)? = null,

    modifier: Modifier = Modifier

) {


    val isUser =
        message.role == ChatMessageRole.USER


    val alignment =
        if (isUser) {
            Alignment.End
        } else {
            Alignment.Start
        }


    val colors =
        if (isUser) {

            CardDefaults.cardColors(
                containerColor =
                    MaterialTheme
                        .colorScheme
                        .primaryContainer
            )

        } else {

            CardDefaults.cardColors(
                containerColor =
                    MaterialTheme
                        .colorScheme
                        .surfaceVariant
            )

        }


    Column(

        modifier =
            modifier
                .fillMaxWidth()
                .padding(
                    vertical = 4.dp
                ),

        horizontalAlignment =
            alignment

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


            Box(

                modifier =
                    Modifier.fillMaxWidth(
                        0.88f
                    )

            ) {


                Card(

                    colors =
                        colors

                ) {


                    Column(

                        modifier =
                            Modifier.padding(
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
                                    if (message.isError) {

                                        MaterialTheme
                                            .colorScheme
                                            .error

                                    } else {

                                        MaterialTheme
                                            .colorScheme
                                            .onSurface

                                    }

                            )

                        }


                        Spacer(
                            modifier =
                                Modifier.padding(
                                    top = 6.dp
                                )
                        )


                        Text(

                            text =
                                formatMessageTime(
                                    message.time
                                ),

                            style =
                                MaterialTheme
                                    .typography
                                    .labelSmall,

                            color =
                                MaterialTheme
                                    .colorScheme
                                    .onSurfaceVariant

                        )


                        MessageActions(

                            message = message,

                            onCopy = {

                                onCopy?.invoke(it)

                            },

                            onRetry = {

                                onRetry?.invoke(it)

                            }

                        )


                    }

                }

            }

        }

    }

}



private fun formatMessageTime(
    time: Long
): String {

    if (time <= 0) {
        return ""
    }


    return SimpleDateFormat(
        "HH:mm",
        Locale.getDefault()
    )
        .format(
            Date(time)
        )

}

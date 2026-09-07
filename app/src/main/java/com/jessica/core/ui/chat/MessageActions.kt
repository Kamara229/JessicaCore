package com.jessica.core.ui.chat

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.chat.ChatMessage
import com.jessica.core.modules.chat.ChatMessageRole


@Composable
fun MessageActions(

    message: ChatMessage,

    onCopy: (String) -> Unit,

    onRetry: (ChatMessage) -> Unit

) {


    Row(

        modifier =
            Modifier
                .padding(
                    top = 4.dp
                ),

        horizontalArrangement =
            Arrangement.End

    ) {


        IconButton(

            onClick = {
                onCopy(
                    message.text
                )
            }

        ) {

            Icon(

                imageVector =
                    Icons.Default.ContentCopy,

                contentDescription =
                    "Копировать"

            )

        }



        if (
            message.role ==
            ChatMessageRole.JESSICA
        ) {


            IconButton(

                onClick = {
                    onRetry(
                        message
                    )
                }

            ) {

                Icon(

                    imageVector =
                        Icons.Default.Refresh,

                    contentDescription =
                        "Повторить"

                )

            }

        }

    }

}

package com.jessica.core.ui.chat


import androidx.compose.material3.Text
import androidx.compose.material3.TextButton

import androidx.compose.runtime.Composable

import com.jessica.core.modules.chat.ChatMessage



@Composable
fun MessageActions(

    message: ChatMessage,

    onCopy: (String) -> Unit,

    onRetry: (ChatMessage) -> Unit

) {


    TextButton(

        onClick = {

            onCopy(
                message.text
            )

        }

    ) {


        Text(
            "📋"
        )


    }



    if (!message.isUser) {


        TextButton(

            onClick = {

                onRetry(
                    message
                )

            }

        ) {


            Text(
                "🔄"
            )


        }


    }


}

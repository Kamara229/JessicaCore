package com.jessica.core.ui.chat


import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ContentCopy

import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.TooltipBox

import androidx.compose.runtime.Composable

import androidx.compose.ui.platform.LocalClipboardManager

import androidx.compose.ui.text.AnnotatedString

import com.jessica.core.modules.chat.ChatMessage



@Composable
fun MessageActions(

    message: ChatMessage

) {


    val clipboard =
        LocalClipboardManager.current



    IconButton(

        onClick = {


            clipboard.setText(

                AnnotatedString(

                    message.text

                )

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


}

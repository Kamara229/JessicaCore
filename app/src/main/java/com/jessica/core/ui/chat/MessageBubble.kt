package com.jessica.core.ui.chat


import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding

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


    Surface(

        modifier =
            Modifier
                .padding(
                    vertical = 4.dp
                ),

        shape =
            MaterialTheme.shapes.medium

    ) {


        Column(

            modifier =
                Modifier
                    .padding(12.dp)

        ) {



            Text(

                text =
                    message.text,


                style =
                    MaterialTheme.typography.bodyLarge

            )



            Text(

                text =
                    formatTime(
                        message.timestamp
                    ),


                style =
                    MaterialTheme.typography.bodySmall

            )


        }


    }


}



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

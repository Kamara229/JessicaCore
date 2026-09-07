package com.jessica.core.ui.chat

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.TooltipBox
import androidx.compose.material3.Text
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import com.jessica.core.modules.chat.ChatMessage


/*
 * =========================================================
 * JESSICA MESSAGE ACTIONS
 * =========================================================
 *
 * Действия над сообщением.
 *
 * Сейчас:
 *
 * - копировать текст;
 * - повторить запрос.
 *
 * В дальнейшем:
 *
 * - память;
 * - блоки;
 * - экспорт;
 * - источники.
 *
 */


@Composable
fun MessageActions(

    message: ChatMessage,

    onCopy: (String) -> Unit,

    onRetry: ((ChatMessage) -> Unit)? = null,

    modifier: Modifier = Modifier

) {


    Row(

        modifier =
            modifier,

        horizontalArrangement =
            Arrangement.Start,

        verticalAlignment =
            Alignment.CenterVertically

    ) {


        /*
         * Копирование текста
         */

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
                    "Копировать",

                tint =
                    MaterialTheme
                        .colorScheme
                        .onSurfaceVariant

            )

        }



        /*
         * Повторить запрос
         *
         * Пока подключаем только если
         * callback передан.
         */

        if (
            onRetry != null
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
                        "Повторить",

                    tint =
                        MaterialTheme
                            .colorScheme
                            .onSurfaceVariant

                )

            }


        }


    }

}

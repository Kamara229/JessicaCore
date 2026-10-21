package com.jessica.core.ui.status

import com.jessica.core.BuildConfig


import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.jessica.core.ui.language.uiText



@Composable
fun JessicaStatusCard(

    version: String = "v" + BuildConfig.VERSION_NAME,

    status: String? = null

) {


    Card(

        modifier =
            Modifier
                .fillMaxWidth()
                .padding(
                    horizontal = 16.dp,
                    vertical = 8.dp
                )

    ) {


        Column(

            modifier =
                Modifier.padding(
                    16.dp
                )

        ) {


            Text(

                text = "Jessica Core",

                style =
                    MaterialTheme
                        .typography
                        .titleLarge

            )


            Text(

                text =
                    uiText("Версия: ", "Version: ") + version,

                style =
                    MaterialTheme
                        .typography
                        .bodyMedium

            )


            Text(

                text =
                    uiText("Статус: ", "Status: ") + (status ?: uiText("Готова к работе", "Ready to work")),

                style =
                    MaterialTheme
                        .typography
                        .bodyMedium

            )


        }

    }

}

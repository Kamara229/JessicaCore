package com.jessica.core.ui.status

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp


@Composable
fun JessicaStatusCard(

    version: String = "v0.1",

    status: String = "Готова к работе"

) {


    Card(

        modifier =
            Modifier
                .fillMaxWidth()
                .padding(16.dp)

    ) {


        Column(

            modifier =
                Modifier.padding(16.dp)

        ) {


            Text(

                text = "Jessica Core",

                style =
                    MaterialTheme
                        .typography
                        .titleMedium

            )


            Text(
                text = "Версия: $version"
            )


            Text(
                text = "Статус: $status"
            )


        }

    }

}

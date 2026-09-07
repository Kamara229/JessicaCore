package com.jessica.core.ui.update

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.updater.UpdateState


@Composable
fun UpdateCard(

    state: UpdateState,

    onUpdateClick: () -> Unit

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

                text =
                    when(state) {

                        UpdateState.Idle ->
                            "Версия v0.1"

                        UpdateState.Checking ->
                            "Проверяю обновления..."

                        is UpdateState.Downloading ->
                            "Загрузка: ${state.progress}%"

                        is UpdateState.Completed ->
                            "Обновлено до ${state.version}"

                        is UpdateState.Error ->
                            "Ошибка: ${state.message}"

                    }

            )


            Button(

                onClick =
                    onUpdateClick

            ) {

                Text(
                    text = "Обновить Jessica"
                )

            }


        }

    }

}

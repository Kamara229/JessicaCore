package com.jessica.core.ui.update


import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
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

                text = "Обновление Jessica",

                style =
                    MaterialTheme
                        .typography
                        .titleMedium

            )



            Text(

                text =
                    when(state) {


                        UpdateState.Idle ->

                            "Проверить наличие новой версии"



                        UpdateState.Checking ->

                            "Проверяю обновления..."



                        is UpdateState.Downloading ->

                            "Загрузка: ${state.progress}%"



                        is UpdateState.Completed ->

                            "Обновление завершено: ${state.version}"



                        is UpdateState.Error ->

                            "Ошибка: ${state.message}"

                    }

            )



            if (
                state is UpdateState.Downloading
            ) {


                LinearProgressIndicator(

                    progress =
                        state.progress / 100f,

                    modifier =
                        Modifier
                            .fillMaxWidth()
                            .padding(
                                top = 8.dp
                            )

                )

            }



            Button(

                onClick = onUpdateClick,

                modifier =
                    Modifier
                        .fillMaxWidth()
                        .padding(
                            top = 12.dp
                        )

            ) {


                Text(

                    text =
                        "🔄 Обновить Jessica"

                )

            }


        }

    }

}

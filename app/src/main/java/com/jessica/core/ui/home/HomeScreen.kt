package com.jessica.core.ui.home


import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding

import androidx.compose.foundation.lazy.LazyColumn

import androidx.compose.material3.Button
import androidx.compose.material3.Text

import androidx.compose.runtime.Composable

import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

import com.jessica.core.modules.updater.UpdateState
import com.jessica.core.ui.status.JessicaStatusCard
import com.jessica.core.ui.update.UpdateCard



@Composable
fun HomeScreen(

    message: String,

    blockCount: Int,

    updateState: UpdateState,

    onUpdate: () -> Unit,

    onChat: () -> Unit,

    onBlocks: () -> Unit,

    onReports: () -> Unit,

    onMemory: () -> Unit,

    onTasks: () -> Unit,

    onAddBlock: () -> Unit

) {


    LazyColumn(

        modifier =
            Modifier
                .fillMaxSize()
                .padding(20.dp),


        verticalArrangement =
            Arrangement.spacedBy(10.dp)

    ) {



        item {


            JessicaStatusCard()


        }



        item {


            UpdateCard(

                state = updateState,

                onUpdateClick = onUpdate

            )


        }



        item {


            Text(

                text = message

            )


        }



        item {


            Text(

                text =
                    "Установлено блоков: $blockCount"

            )


        }



        item {


            Spacer(

                modifier =
                    Modifier.height(10.dp)

            )


        }



        item {


            Button(

                onClick = onChat,

                modifier =
                    Modifier.fillMaxWidth()

            ) {


                Text(

                    "Чат Jessica"

                )


            }


        }



        item {


            Button(

                onClick = onBlocks,

                modifier =
                    Modifier.fillMaxWidth()

            ) {


                Text(
                    "Блоки"
                )

            }


        }



        item {


            Button(

                onClick = onReports,

                modifier =
                    Modifier.fillMaxWidth()

            ) {


                Text(
                    "Отчёты"
                )

            }


        }



        item {


            Button(

                onClick = onMemory,

                modifier =
                    Modifier.fillMaxWidth()

            ) {


                Text(
                    "Память"
                )

            }


        }



        item {


            Button(

                onClick = onTasks,

                modifier =
                    Modifier.fillMaxWidth()

            ) {


                Text(
                    "Задачи"
                )

            }


        }



        item {


            Spacer(

                modifier =
                    Modifier.height(10.dp)

            )


        }



        item {


            Button(

                onClick = onAddBlock,

                modifier =
                    Modifier.fillMaxWidth()

            ) {


                Text(
                    "+ Добавить блок"
                )

            }


        }


    }


}

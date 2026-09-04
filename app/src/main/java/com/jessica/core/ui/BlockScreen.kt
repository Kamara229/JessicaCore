package com.jessica.core.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.Block
import com.jessica.core.modules.BlockManager


@Composable
fun BlockScreen(
    blockManager: BlockManager,
    onUpdate: () -> Unit
) {

    val blocks =
        blockManager.getBlocks()


    Column(
        modifier = Modifier
            .padding(20.dp)
    ) {


        Text(
            text = "Менеджер блоков",
            style = MaterialTheme.typography.titleLarge
        )


        Spacer(
            modifier = Modifier.height(20.dp)
        )


        if (blocks.isEmpty()) {

            Text(
                "Нет установленных блоков"
            )

        }


        blocks.forEach { block ->


            Card(

                modifier = Modifier
                    .fillMaxWidth()
                    .padding(5.dp)

            ){

                Column(
                    modifier = Modifier
                        .padding(15.dp)
                ){

                    Text(
                        text =
                        block.name
                    )


                    Text(
                        text =
                        "Версия: ${block.version}"
                    )


                    Text(
                        text =
                        "Статус: ${block.status}"
                    )


                    Spacer(
                        modifier = Modifier.height(10.dp)
                    )


                    Row {


                        Button(

                            onClick = {

                                // тест блока

                            }

                        ){

                            Text(
                                "Тест"
                            )

                        }


                        Spacer(
                            modifier = Modifier.width(10.dp)
                        )


                        Button(

                            onClick = {

                                blockManager.removeBlock(block)

                                onUpdate()

                            }

                        ){

                            Text(
                                "Удалить"
                            )

                        }

                    }

                }

            }

        }

    }

}

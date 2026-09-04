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

    val blocks = blockManager.getBlocks()

    var testResult by remember {
        mutableStateOf("")
    }

    var testedBlockName by remember {
        mutableStateOf("")
    }


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
                    .padding(vertical = 5.dp)

            ) {

                Column(
                    modifier = Modifier
                        .padding(15.dp)
                ) {


                    Text(
                        text = block.name,
                        style = MaterialTheme.typography.titleMedium
                    )


                    Text(
                        text = "Версия: ${block.version}"
                    )


                    Text(
                        text = "Статус: ${block.status}"
                    )


                    Spacer(
                        modifier = Modifier.height(10.dp)
                    )


                    Row {


                        Button(

                            onClick = {

                                testedBlockName = block.name

                                testResult = testBlock(
                                    block = block,
                                    blockManager = blockManager
                                )

                            }

                        ) {

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

                                if (testedBlockName == block.name) {
                                    testedBlockName = ""
                                    testResult = ""
                                }

                            }

                        ) {

                            Text(
                                "Удалить"
                            )

                        }

                    }


                    if (
                        testedBlockName == block.name &&
                        testResult.isNotEmpty()
                    ) {


                        Spacer(
                            modifier = Modifier.height(15.dp)
                        )


                        HorizontalDivider()


                        Spacer(
                            modifier = Modifier.height(10.dp)
                        )


                        Text(
                            text = testResult
                        )

                    }

                }

            }

        }

    }

}


fun testBlock(
    block: Block,
    blockManager: BlockManager
): String {


    val nameOk =
        block.name.isNotBlank()


    val versionOk =
        block.version.isNotBlank()


    val statusOk =
        block.status == "ACTIVE"


    val registeredOk =
        blockManager
            .getBlocks()
            .contains(block)


    val allOk =
        nameOk &&
        versionOk &&
        statusOk &&
        registeredOk


    return buildString {


        appendLine(
            if (allOk)
                "PASS"
            else
                "FAIL"
        )


        appendLine()


        appendLine(
            "Имя: " +
                if (nameOk)
                    "OK"
                else
                    "ERROR"
        )


        appendLine(
            "Версия: " +
                if (versionOk)
                    "OK"
                else
                    "ERROR"
        )


        appendLine(
            "Статус: " +
                if (statusOk)
                    "OK"
                else
                    "ERROR"
        )


        appendLine(
            "Регистрация: " +
                if (registeredOk)
                    "OK"
                else
                    "ERROR"
        )

    }

}

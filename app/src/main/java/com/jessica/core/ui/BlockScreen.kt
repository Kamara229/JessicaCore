package com.jessica.core.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.platform.LocalContext
import com.jessica.core.modules.BlockManager
import com.jessica.core.modules.BlockTester
import com.jessica.core.modules.CapabilityEngine
import com.jessica.core.modules.ReportStorage
import com.jessica.core.modules.TestReport
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale


@Composable
fun BlockScreen(
    blockManager: BlockManager,
    reportStorage: ReportStorage,
    onUpdate: () -> Unit,
    onBack: () -> Unit
) {

    val context =
        LocalContext.current


    val blocks =
        blockManager
            .getBlocks()
            .toList()


    var message by remember {
        mutableStateOf("")
    }


    var reports by remember {
        mutableStateOf(
            reportStorage.loadReports()
        )
    }


    LazyColumn(
        modifier =
            Modifier
                .fillMaxSize()
                .padding(20.dp),

        verticalArrangement =
            Arrangement.spacedBy(10.dp)
    ) {


        item {

            Button(
                onClick = onBack
            ) {
                Text("Назад")
            }

        }


        item {

            Text(
                text = "Менеджер блоков",
                style =
                    MaterialTheme.typography.titleLarge
            )

        }


        item {

            Text(
                text =
                    "Установлено блоков: ${blocks.size}"
            )

        }


        if (blocks.isEmpty()) {

            item {

                Text(
                    "Нет установленных блоков"
                )

            }

        }


        items(
            items = blocks,
            key = { block ->

                block.id.ifBlank {
                    block.name
                }

            }
        ) { block ->


            Card(
                modifier =
                    Modifier
                        .fillMaxWidth()
            ) {


                Column(
                    modifier =
                        Modifier
                            .padding(15.dp)
                ) {


                    Text(
                        text = block.name,
                        style =
                            MaterialTheme.typography.titleMedium
                    )


                    Spacer(
                        modifier =
                            Modifier.height(6.dp)
                    )


                    Text(
                        "ID: ${block.id}"
                    )


                    Text(
                        "Версия: ${block.version}"
                    )


                    Text(
                        "Автор: ${block.author}"
                    )


                    Text(
                        "Тип: ${block.type}"
                    )


                    Text(
                        "Статус: ${block.status}"
                    )


                    if (
                        block.description.isNotBlank()
                    ) {

                        Spacer(
                            modifier =
                                Modifier.height(8.dp)
                        )


                        Text(
                            block.description
                        )

                    }


                    Spacer(
                        modifier =
                            Modifier.height(10.dp)
                    )


                    Text(
                        text = "Возможности:",
                        style =
                            MaterialTheme.typography.labelLarge
                    )


                    if (
                        block.capabilities.isEmpty()
                    ) {

                        Text(
                            "Нет заявленных возможностей"
                        )

                    } else {

                        block.capabilities.forEach { capability ->

                            Text(
                                "• $capability"
                            )

                        }

                    }


                    Spacer(
                        modifier =
                            Modifier.height(15.dp)
                    )


                    Row(
                        modifier =
                            Modifier.fillMaxWidth()
                    ) {


                        Button(
                            onClick = {

                                val engine =
                                    CapabilityEngine(
                                        context
                                    )


                                val resultText =
                                    buildString {

                                        appendLine(
                                            "Запуск: ${block.name}"
                                        )

                                        appendLine()


                                        if (
                                            block.capabilities.isEmpty()
                                        ) {

                                            appendLine(
                                                "У блока нет возможностей для запуска"
                                            )

                                        } else {

                                            block.capabilities.forEach { capability ->

                                                val result =
                                                    engine.execute(
                                                        capability
                                                    )


                                                appendLine(
                                                    if (result.success)
                                                        "✅ $capability"
                                                    else
                                                        "❌ $capability"
                                                )


                                                appendLine(
                                                    result.message
                                                )


                                                appendLine()

                                            }

                                        }

                                    }


                                message =
                                    resultText

                            }
                        ) {

                            Text(
                                "Запустить"
                            )

                        }


                        Spacer(
                            modifier =
                                Modifier.width(8.dp)
                        )


                        Button(
                            onClick = {

                                val tester =
                                    BlockTester()


                                val result =
                                    tester.test(
                                        block,
                                        blockManager
                                    )


                                val report =
                                    TestReport(

                                        blockName =
                                            block.name,

                                        date =
                                            SimpleDateFormat(
                                                "dd.MM.yyyy HH:mm",
                                                Locale.getDefault()
                                            ).format(
                                                Date()
                                            ),

                                        result =
                                            result.report

                                    )


                                reports =
                                    reports + report


                                reportStorage.saveReports(
                                    reports
                                )


                                message =
                                    "Тест завершён: ${block.name}"

                            }
                        ) {

                            Text(
                                "Тест"
                            )

                        }

                    }


                    Spacer(
                        modifier =
                            Modifier.height(8.dp)
                    )


                    Button(
                        onClick = {

                            blockManager.removeBlock(
                                block
                            )


                            onUpdate()


                            message =
                                "Блок ${block.name} удалён"

                        }
                    ) {

                        Text(
                            "Удалить"
                        )

                    }

                }

            }

        }


        if (
            message.isNotEmpty()
        ) {

            item {

                Spacer(
                    modifier =
                        Modifier.height(10.dp)
                )


                HorizontalDivider()


                Spacer(
                    modifier =
                        Modifier.height(10.dp)
                )


                Text(
                    text = message
                )

            }

        }


        item {

            Spacer(
                modifier =
                    Modifier.height(30.dp)
            )

        }

    }

}

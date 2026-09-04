package com.jessica.core.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.BlockManager
import com.jessica.core.modules.BlockTester
import com.jessica.core.modules.ReportStorage
import com.jessica.core.modules.TestReport
import java.text.SimpleDateFormat
import java.util.Date


@Composable
fun BlockScreen(
    blockManager: BlockManager,
    reportStorage: ReportStorage,
    onUpdate: () -> Unit,
    onBack: () -> Unit
) {


    val blocks =
        blockManager.getBlocks()


    var testMessage by remember {

        mutableStateOf("")

    }


    var reports by remember {

        mutableStateOf(
            reportStorage.loadReports()
        )

    }



    Column(

        modifier =
        Modifier
            .fillMaxSize()
            .padding(20.dp)

    ) {


        Button(

            onClick = {

                onBack()

            }

        ) {

            Text(
                "Назад"
            )

        }


        Spacer(

            modifier =
            Modifier.height(20.dp)

        )



        Text(

            text = "Менеджер блоков",

            style =
            MaterialTheme.typography.titleLarge

        )



        Spacer(

            modifier =
            Modifier.height(20.dp)

        )



        if (blocks.isEmpty()) {


            Text(
                "Нет установленных блоков"
            )


        }



        blocks.forEach { block ->



            Card(

                modifier =
                Modifier
                    .fillMaxWidth()
                    .padding(vertical = 5.dp)

            ) {



                Column(

                    modifier =
                    Modifier
                        .padding(15.dp)

                ) {



                    Text(

                        text =
                        block.name,

                        style =
                        MaterialTheme.typography.titleMedium

                    )



                    Text(
                        "Версия: ${block.version}"
                    )



                    Text(
                        "Статус: ${block.status}"
                    )



                    Spacer(

                        modifier =
                        Modifier.height(10.dp)

                    )



                    Row {


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
                                            "dd.MM.yyyy HH:mm"
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



                                testMessage =
                                    "Тест завершён: ${block.name}"


                            }

                        ) {


                            Text(
                                "Тест"
                            )

                        }



                        Spacer(

                            modifier =
                            Modifier.width(10.dp)

                        )



                        Button(

                            onClick = {


                                blockManager.removeBlock(
                                    block
                                )


                                onUpdate()


                                testMessage =
                                    "Блок удалён"


                            }

                        ) {


                            Text(
                                "Удалить"
                            )

                        }


                    }


                }


            }


        }



        if (testMessage.isNotEmpty()) {


            Spacer(

                modifier =
                Modifier.height(20.dp)

            )


            HorizontalDivider()



            Spacer(

                modifier =
                Modifier.height(10.dp)

            )


            Text(
                testMessage
            )


        }


    }


}

package com.jessica.core.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.ReportStorage
import com.jessica.core.modules.TestReport


@Composable
fun ReportScreen(
    reportStorage: ReportStorage
) {


    var reports by remember {

        mutableStateOf(
            emptyList<TestReport>()
        )

    }


    LaunchedEffect(Unit) {

        reports =
            reportStorage.loadReports()

    }



    Column(

        modifier =
        Modifier
            .fillMaxSize()

    ) {


        Text(

            text = "История тестирования",

            style =
            MaterialTheme.typography.titleLarge,

            modifier =
            Modifier
                .padding(20.dp)

        )


        if (reports.isEmpty()) {


            Text(

                text = "Отчётов пока нет",

                modifier =
                Modifier
                    .padding(20.dp)

            )


        } else {


            LazyColumn(

                modifier =
                Modifier
                    .fillMaxSize()
                    .padding(horizontal = 20.dp),

                verticalArrangement =
                Arrangement.spacedBy(10.dp)

            ) {


                items(reports) { report ->


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

                                text =
                                report.blockName,

                                style =
                                MaterialTheme.typography.titleMedium

                            )


                            Spacer(

                                modifier =
                                Modifier.height(5.dp)

                            )


                            Text(

                                text =
                                "Дата: ${report.date}"

                            )


                            Spacer(

                                modifier =
                                Modifier.height(5.dp)

                            )


                            Text(

                                text =
                                "Результат:"

                            )


                            Text(

                                text =
                                report.result

                            )


                        }

                    }


                }


            }


        }


    }


}

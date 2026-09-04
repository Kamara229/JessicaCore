package com.jessica.core.ui

import androidx.compose.foundation.layout.*
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
            .padding(20.dp)

    ) {


        Text(

            text = "История тестирования",

            style =
            MaterialTheme.typography.titleLarge

        )


        Spacer(

            modifier =
            Modifier.height(20.dp)

        )



        if (reports.isEmpty()) {


            Text(
                "Отчётов пока нет"
            )


        }



        reports.forEach { report ->


            Card(

                modifier =
                Modifier
                    .fillMaxWidth()
                    .padding(5.dp)

            ) {


                Column(

                    modifier =
                    Modifier.padding(15.dp)

                ) {


                    Text(

                        text =
                        report.blockName,

                        style =
                        MaterialTheme.typography.titleMedium

                    )


                    Text(
                        "Дата: ${report.date}"
                    )


                    Text(
                        "Результат:"
                    )


                    Text(
                        report.result
                    )


                }

            }


        }


    }


}

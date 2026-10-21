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
import com.jessica.core.ui.language.uiText


@Composable
fun ReportScreen(
    reportStorage: ReportStorage,
    onBack: () -> Unit
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

                onClick = {
                    onBack()
                }

            ) {

                Text(uiText("Назад", "Back"))

            }

        }


        item {

            Spacer(
                modifier = Modifier.height(10.dp)
            )


            Text(

                text = uiText("История тестирования", "Test history"),

                style =
                    MaterialTheme.typography.titleLarge

            )

        }


        if (reports.isEmpty()) {

            item {

                Text(
                    uiText("Отчётов пока нет", "No reports yet")
                )

            }

        } else {


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

                            text = report.blockName,

                            style =
                                MaterialTheme.typography.titleMedium

                        )


                        Spacer(
                            modifier = Modifier.height(5.dp)
                        )


                        Text(
                            text = uiText("Дата: ", "Date: ") + report.date
                        )


                        Spacer(
                            modifier = Modifier.height(5.dp)
                        )


                        Text(
                            text = uiText("Результат:", "Result:")
                        )


                        Text(
                            text = report.result
                        )

                    }

                }

            }

        }


        item {

            Spacer(
                modifier = Modifier.height(30.dp)
            )

        }

    }

}

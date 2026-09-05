package com.jessica.core.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.EventStorage
import com.jessica.core.modules.JessicaEvent
import com.jessica.core.modules.MemoryItem
import com.jessica.core.modules.MemoryStorage
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale


@Composable
fun MemoryScreen(
    onBack: () -> Unit
) {

    val context =
        LocalContext.current


    val memoryStorage =
        remember {
            MemoryStorage(context)
        }


    val eventStorage =
        remember {
            EventStorage(context)
        }


    var memoryItems by remember {
        mutableStateOf(
            emptyList<MemoryItem>()
        )
    }


    var events by remember {
        mutableStateOf(
            emptyList<JessicaEvent>()
        )
    }


    fun refreshData() {

        memoryItems =
            memoryStorage
                .loadMemory()
                .sortedByDescending {
                    it.timestamp
                }


        events =
            eventStorage
                .loadEvents()
                .sortedByDescending {
                    it.timestamp
                }

    }


    LaunchedEffect(Unit) {

        refreshData()

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

                Text(
                    "Назад"
                )

            }

        }


        item {

            Text(
                text = "Память Jessica",
                style =
                    MaterialTheme.typography.titleLarge
            )

        }


        item {

            Button(
                onClick = {
                    refreshData()
                }
            ) {

                Text(
                    "Обновить"
                )

            }

        }


        item {

            HorizontalDivider()

        }


        item {

            Text(
                text = "Сохранённая память",
                style =
                    MaterialTheme.typography.titleMedium
            )

        }


        if (memoryItems.isEmpty()) {

            item {

                Text(
                    "Записей памяти пока нет"
                )

            }

        } else {

            items(
                items = memoryItems,
                key = {
                    it.key
                }
            ) { memory ->


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
                            text = memory.key,
                            style =
                                MaterialTheme.typography.titleSmall
                        )


                        Spacer(
                            modifier =
                                Modifier.height(5.dp)
                        )


                        Text(
                            text = memory.value
                        )


                        Spacer(
                            modifier =
                                Modifier.height(5.dp)
                        )


                        Text(
                            text =
                                formatTimestamp(
                                    memory.timestamp
                                )
                        )

                    }

                }

            }

        }


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
                text = "История событий",
                style =
                    MaterialTheme.typography.titleMedium
            )

        }


        if (events.isEmpty()) {

            item {

                Text(
                    "Событий пока нет"
                )

            }

        } else {

            items(
                items = events,
                key = {
                    "${it.timestamp}_${it.type}_${it.message}"
                }
            ) { event ->


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
                            text = event.type,
                            style =
                                MaterialTheme.typography.titleSmall
                        )


                        Spacer(
                            modifier =
                                Modifier.height(5.dp)
                        )


                        Text(
                            text = event.message
                        )


                        Spacer(
                            modifier =
                                Modifier.height(5.dp)
                        )


                        Text(
                            text =
                                formatTimestamp(
                                    event.timestamp
                                )
                        )

                    }

                }

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


private fun formatTimestamp(
    timestamp: Long
): String {

    return SimpleDateFormat(
        "dd.MM.yyyy HH:mm:ss",
        Locale.getDefault()
    ).format(
        Date(timestamp)
    )

}

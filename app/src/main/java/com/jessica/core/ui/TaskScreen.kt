package com.jessica.core.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.jessica.core.modules.CapabilityEngine
import com.jessica.core.modules.EventStorage
import com.jessica.core.modules.JessicaTask
import com.jessica.core.modules.TaskStorage
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale


@Composable
fun TaskScreen(
    onBack: () -> Unit
) {

    val context =
        LocalContext.current


    val taskStorage =
        remember {
            TaskStorage(context)
        }


    val eventStorage =
        remember {
            EventStorage(context)
        }


    val capabilityEngine =
        remember {
            CapabilityEngine(context)
        }


    val coroutineScope =
        rememberCoroutineScope()


    var taskText by remember {
        mutableStateOf("")
    }


    var tasks by remember {
        mutableStateOf(
            emptyList<JessicaTask>()
        )
    }


    var message by remember {
        mutableStateOf("")
    }


    var processingTaskId by remember {
        mutableStateOf<Long?>(null)
    }


    fun refreshTasks() {

        tasks =
            taskStorage
                .loadTasks()
                .sortedByDescending {
                    it.createdAt
                }

    }


    LaunchedEffect(Unit) {

        refreshTasks()

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
                text = "Задачи",
                style =
                    MaterialTheme.typography.titleLarge
            )

        }


        item {

            Text(
                text =
                    "Создайте задачу для Jessica Core"
            )

        }


        item {

            OutlinedTextField(

                value =
                    taskText,

                onValueChange = {
                    taskText = it
                },

                modifier =
                    Modifier.fillMaxWidth(),

                label = {
                    Text(
                        "Что нужно сделать?"
                    )
                },

                minLines = 3

            )

        }


        item {

            Button(

                onClick = {

                    val text =
                        taskText.trim()


                    if (text.isBlank()) {

                        message =
                            "Введите текст задачи"

                    } else {

                        val task =
                            taskStorage.createTask(
                                text
                            )


                        eventStorage.saveEvent(
                            type = "task",
                            message =
                                "Создана задача ${task.id}"
                        )


                        taskText = ""

                        message =
                            "Задача создана"


                        refreshTasks()

                    }

                },

                modifier =
                    Modifier.fillMaxWidth()

            ) {

                Text(
                    "Создать задачу"
                )

            }

        }


        if (message.isNotBlank()) {

            item {

                Text(
                    text = message
                )

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
                text = "История задач",
                style =
                    MaterialTheme.typography.titleMedium
            )

        }


        if (tasks.isEmpty()) {

            item {

                Text(
                    "Задач пока нет"
                )

            }

        } else {

            items(

                items = tasks,

                key = {
                    it.id
                }

            ) { task ->


                TaskCard(

                    task = task,

                    isProcessing =
                        processingTaskId == task.id,

                    onSolve = {

                        if (processingTaskId == null) {

                            processingTaskId =
                                task.id


                            message =
                                "Выполнение задачи..."


                            coroutineScope.launch {

                                try {

                                    val result =
                                        capabilityEngine.execute(
                                            capability =
                                                "solve_task",
                                            taskId =
                                                task.id
                                        )


                                    message =
                                        result.message

                                } catch (e: Exception) {

                                    message =
                                        e.message
                                            ?: "Ошибка выполнения задачи"

                                } finally {

                                    processingTaskId =
                                        null


                                    refreshTasks()

                                }

                            }

                        }

                    },

                    onDelete = {

                        if (processingTaskId != task.id) {

                            taskStorage.deleteTask(
                                task.id
                            )


                            eventStorage.saveEvent(
                                type = "task",
                                message =
                                    "Удалена задача ${task.id}"
                            )


                            message =
                                "Задача удалена"


                            refreshTasks()

                        }

                    }

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


@Composable
private fun TaskCard(
    task: JessicaTask,
    isProcessing: Boolean,
    onSolve: () -> Unit,
    onDelete: () -> Unit
) {

    Card(
        modifier =
            Modifier.fillMaxWidth()
    ) {

        Column(
            modifier =
                Modifier.padding(15.dp)
        ) {


            Text(
                text =
                    task.text,

                style =
                    MaterialTheme.typography.titleMedium
            )


            Spacer(
                modifier =
                    Modifier.height(8.dp)
            )


            Text(
                text =
                    "Статус: ${
                        when {
                            isProcessing ->
                                "PROCESSING"

                            else ->
                                task.status
                        }
                    }"
            )


            Text(
                text =
                    "Создана: ${
                        formatTaskDate(
                            task.createdAt
                        )
                    }"
            )


            if (task.result.isNotBlank()) {

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
                    text = "Результат:",
                    style =
                        MaterialTheme.typography.labelLarge
                )


                Text(
                    text =
                        task.result
                )

            }


            Spacer(
                modifier =
                    Modifier.height(15.dp)
            )


            if (
                task.status != "COMPLETED" &&
                !isProcessing
            ) {

                Button(
                    onClick =
                        onSolve,

                    modifier =
                        Modifier.fillMaxWidth()
                ) {

                    Text(
                        if (task.status == "FAILED") {
                            "Повторить"
                        } else {
                            "Решить"
                        }
                    )

                }


                Spacer(
                    modifier =
                        Modifier.height(8.dp)
                )

            }


            if (isProcessing) {

                LinearProgressIndicator(
                    modifier =
                        Modifier.fillMaxWidth()
                )


                Spacer(
                    modifier =
                        Modifier.height(8.dp)
                )


                Text(
                    "Jessica выполняет задачу..."
                )


                Spacer(
                    modifier =
                        Modifier.height(8.dp)
                )

            }


            OutlinedButton(

                onClick =
                    onDelete,

                enabled =
                    !isProcessing,

                modifier =
                    Modifier.fillMaxWidth()

            ) {

                Text(
                    "Удалить"
                )

            }

        }

    }

}


private fun formatTaskDate(
    timestamp: Long
): String {

    return SimpleDateFormat(
        "dd.MM.yyyy HH:mm",
        Locale.getDefault()
    ).format(
        Date(timestamp)
    )

}

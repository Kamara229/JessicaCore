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
import com.jessica.core.ui.language.uiText
import com.jessica.core.ui.language.LocalAppLanguage
import com.jessica.core.ui.language.localeText
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

    val appLanguage = LocalAppLanguage.current
    fun t(ru: String, en: String) = localeText(appLanguage, ru, en)


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

                Text(uiText("Назад", "Back"))

            }

        }


        item {

            Text(
                text = uiText("Задачи", "Tasks"),
                style =
                    MaterialTheme.typography.titleLarge
            )

        }


        item {

            Text(
                text =
                    uiText("Создайте задачу для Jessica Core", "Create a task for Jessica Core")
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
                    Text(uiText("Что нужно сделать?", "What needs to be done?"))
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
                            t("Введите текст задачи", "Enter a task")

                    } else {

                        val task =
                            taskStorage.createTask(
                                text
                            )


                        eventStorage.saveEvent(
                            type = "task",
                            message =
                                t("Создана задача ${task.id}", "Task ${task.id} created")
                        )


                        taskText = ""

                        message =
                            t("Задача создана", "Task created")


                        refreshTasks()

                    }

                },

                modifier =
                    Modifier.fillMaxWidth()

            ) {

                Text(
                    uiText("Создать задачу", "Create task")
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
                text = uiText("История задач", "Task history"),
                style =
                    MaterialTheme.typography.titleMedium
            )

        }


        if (tasks.isEmpty()) {

            item {

                Text(
                    uiText("Задач пока нет", "No tasks yet")
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
                                t("Выполнение задачи...", "Running task...")


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
                                            ?: t("Ошибка выполнения задачи", "Could not complete the task")

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
                                    t("Удалена задача ${task.id}", "Task ${task.id} deleted")
                            )


                            message =
                                t("Задача удалена", "Task deleted")


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
                    uiText("Статус: ", "Status: ") + "${
                        when {
                            isProcessing ->
                                uiText("Выполняется", "In progress")

                            else ->
                                when (task.status) {
                                    "COMPLETED" -> uiText("Завершена", "Completed")
                                    "FAILED" -> uiText("Ошибка", "Failed")
                                    "PROCESSING" -> uiText("Выполняется", "In progress")
                                    else -> uiText("Ожидает", "Pending")
                                }
                        }
                    }"
            )


            Text(
                text =
                    uiText("Создана: ", "Created: ") + "${
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
                    text = uiText("Результат:", "Result:"),
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
                            uiText("Повторить", "Retry")
                        } else {
                            uiText("Решить", "Solve")
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
                    uiText("Jessica выполняет задачу...", "Jessica is working on the task...")
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
                    uiText("Удалить", "Delete")
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
